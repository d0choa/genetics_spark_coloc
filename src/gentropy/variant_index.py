"""Step to generate variant index dataset."""

from __future__ import annotations

import math
from functools import reduce

from pyspark.sql import functions as f

from gentropy.common.session import Session
from gentropy.dataset.amino_acid_variants import AminoAcidVariants
from gentropy.dataset.variant_index import VariantIndex
from gentropy.datasource.ensembl.vep_parser import VariantEffectPredictorParser
from gentropy.datasource.open_targets.variants import OpenTargetsVariant


class VariantIndexStep:
    """Generate variant index based on a VEP output in json format.

    The variant index is a dataset that contains variant annotations extracted from VEP output. It is expected that all variants in the VEP output are present in the variant index.
    There's an option to provide extra variant annotations to be added to the variant index eg. allele frequencies from GnomAD.
    """

    def __init__(
        self: VariantIndexStep,
        session: Session,
        vep_output_json_path: str,
        variant_index_path: str,
        hash_threshold: int,
        variant_annotations_path: list[str] | None = None,
        amino_acid_change_annotations: list[str] | None = None,
    ) -> None:
        """Run VariantIndex step.

        Args:
            session (Session): Session object.
            vep_output_json_path (str): Variant effect predictor output path (in json format).
            variant_index_path (str): Variant index dataset path to save resulting data.
            hash_threshold (int): Hash threshold for variant identifier length.
            variant_annotations_path (list[str] | None): List of paths to extra variant annotation datasets.
            amino_acid_change_annotations (list[str] | None): list of paths to amino-acid based variant annotations.
        """
        # Extract variant annotations from VEP output:
        variant_index = VariantEffectPredictorParser.extract_variant_index_from_vep(
            session.spark, vep_output_json_path, hash_threshold
        )

        # Process variant annotations if provided:
        if variant_annotations_path:
            for annotation_path in variant_annotations_path:
                # Read variant annotations from parquet:
                annotations = VariantIndex.from_parquet(
                    session=session,
                    path=annotation_path,
                    recursiveFileLookup=True,
                    id_threshold=hash_threshold,
                )

                # Update index with extra annotations:
                variant_index = variant_index.add_annotation(annotations)

        # If provided read amino-acid based annotation and enrich variant index:
        if amino_acid_change_annotations:
            for annotation_path in amino_acid_change_annotations:
                annotation_data = AminoAcidVariants.from_parquet(
                    session, annotation_path
                )

                # Update index with extra annotations:
                variant_index = variant_index.annotate_with_amino_acid_consequences(
                    annotation_data
                )

        (
            variant_index.df.repartitionByRange(
                session.output_partitions, "chromosome", "position"
            )
            .sortWithinPartitions("chromosome", "position")
            .write.mode(session.write_mode)
            .parquet(variant_index_path)
        )


class ConvertToVcfStep:
    """Convert dataset with variant annotation to VCF step.

    This step converts in-house data source formats to VCF like format.

    NOTE! Due to the csv DataSourceWriter limitations we can not save the column name
    `#CHROM` as in vcf file. The column is replaced with `CHROM`.
    """

    def __init__(
        self,
        session: Session,
        source_paths: list[str],
        source_formats: list[str],
        output_path: str,
        partition_size: int,
    ) -> None:
        """Initialize step.

        Args:
            session (Session): Session object.
            source_paths (list[str]): Input dataset path.
            source_formats (list[str]): Format of the input dataset.
            output_path (str): Output VCF file path.
            partition_size (int): Approximate number of variants in each output partition.

        Raises:
            AssertionError: When the length of `source_paths` does not match the lenght of `source_formats`.
        """
        assert len(source_formats) == len(
            source_paths
        ), "Must provide format for each source path."

        # Load
        raw_variants = [
            session.load_data(p, f)
            for p, f in zip(source_paths, source_formats, strict=True)
        ]

        # Extract
        processed_variants = [
            OpenTargetsVariant.as_vcf_df(session, df) for df in raw_variants
        ]

        # Merge
        merged_variants = reduce(
            lambda x, y: x.unionByName(y), processed_variants
        ).drop_duplicates(["#CHROM", "POS", "REF", "ALT"])

        variant_count = merged_variants.count()
        n_partitions = int(math.ceil(variant_count / partition_size))
        partitioned_variants = (
            merged_variants.repartitionByRange(
                n_partitions, f.col("#CHROM"), f.col("POS")
            )
            .sortWithinPartitions(f.col("#CHROM").asc(), f.col("POS").asc())
            # Due to the large number of partitions ensure we do not lose the partitions before saving them
            .persist()
            # FIXME the #CHROM column is saved as "#CHROM" by pyspark which fails under VEP,
            # The native solution would be to implement the datasource with proper writer
            # see https://docs.databricks.com/en/pyspark/datasources.html.
            # Proposed solution will require adding # at the start of the first line of
            # vcf before processing it in orchestration.
            .withColumnRenamed("#CHROM", "CHROM")
        )
        # Write
        partitioned_variants.write.mode(session.write_mode).option("sep", "\t").option(
            "quote", ""
        ).option("quoteAll", False).option("header", True).csv(output_path)
