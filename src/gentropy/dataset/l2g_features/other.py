"""Methods to generate features which are not obviously categorised."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyspark.sql.functions as f

from gentropy.common.spark_helpers import convert_from_wide_to_long
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_locus import CredibleSetConfidenceClasses, StudyLocus
from gentropy.dataset.target_index import TargetIndex
from gentropy.dataset.variant_index import VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


def common_genecount_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    *,
    target_index: TargetIndex,
    feature_name: str,
    genomic_window: int,
    protein_coding_only: bool = False,
) -> DataFrame:
    """Computes the feature.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci
            that will be used for annotation
        target_index (TargetIndex): Dataset containing information related to all genes in release.
        feature_name (str): The name of the feature
        genomic_window (int): The maximum window size to consider
        protein_coding_only (bool): Whether to only consider protein coding genes in calculation.

    Returns:
            DataFrame: Feature dataset
    """
    study_loci_window = (
        study_loci_to_annotate.df.withColumn(
            "window_start", f.col("position") - (genomic_window / 2)
        )
        .withColumn("window_end", f.col("position") + (genomic_window / 2))
        .withColumnRenamed("chromosome", "SL_chromosome")
    )
    target_index_filter = target_index.df

    if protein_coding_only:
        target_index_filter = target_index_filter.filter(
            f.col("biotype") == "protein_coding"
        )

    distinct_gene_counts = (
        study_loci_window.join(
            target_index_filter.alias("genes"),
            on=(
                (f.col("SL_chromosome") == f.col("genes.genomicLocation.chromosome"))
                & (f.col("genes.tss") >= f.col("window_start"))
                & (f.col("genes.tss") <= f.col("window_end"))
            ),
            how="inner",
        )
        .groupBy("studyLocusId")
        .agg(f.approx_count_distinct(f.col("id").alias("geneId")).alias(feature_name))
    )

    return (
        study_loci_window.join(
            target_index_filter.alias("genes"),
            on=(
                (f.col("SL_chromosome") == f.col("genes.genomicLocation.chromosome"))
                & (f.col("genes.tss") >= f.col("window_start"))
                & (f.col("genes.tss") <= f.col("window_end"))
            ),
            how="inner",
        )
        .join(distinct_gene_counts, on="studyLocusId", how="inner")
        .select("studyLocusId", f.col("id").alias("geneId"), feature_name)
        .distinct()
    )


def is_protein_coding_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    *,
    variant_index: VariantIndex,
    feature_name: str,
    genomic_window: int = 500_000,
) -> DataFrame:
    """Computes the feature to indicate if a gene is protein-coding or not.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci
            that will be used for annotation
        variant_index (VariantIndex): Dataset containing information related to all overlapping genes within a genomic window.
        feature_name (str): The name of the feature
        genomic_window (int): The window size around the locus to consider. Defaults to its maximum value: 500kb up and downstream the locus

    Returns:
        DataFrame: Feature dataset, with 1 if the gene is protein-coding, 0 if not.

    Raises:
        AssertionError: when provided `genomic_window` is more or equal to 500kb.
    """
    assert genomic_window <= 500_000, "Genomic window must be less than 500kb."
    genes_in_window = (
        variant_index.df.withColumn(
            "transcriptConsequence", f.explode("transcriptConsequences")
        )
        .select(
            "variantId",
            f.col("transcriptConsequence.targetId").alias("geneId"),
            f.col("transcriptConsequence.biotype").alias("biotype"),
            f.col("transcriptConsequence.distanceFromFootprint").alias(
                "distanceFromFootprint"
            ),
        )
        .filter(f.col("distanceFromFootprint") <= genomic_window)
    )
    if isinstance(study_loci_to_annotate, StudyLocus):
        variants_df = study_loci_to_annotate.df.select(
            f.explode_outer("locus.variantId").alias("variantId"),
            "studyLocusId",
        ).filter(f.col("variantId").isNotNull())
    elif isinstance(study_loci_to_annotate, L2GGoldStandard):
        variants_df = study_loci_to_annotate.df.select("studyLocusId", "variantId")
    return (
        # Annotate all genes in the window of a locus
        variants_df.join(
            genes_in_window,
            on="variantId",
        )
        # Apply flag across all variants in the locus
        .withColumn(
            feature_name,
            f.when(f.col("biotype") == "protein_coding", f.lit(1.0)).otherwise(
                f.lit(0.0)
            ),
        )
        .select("studyLocusId", "geneId", feature_name)
        .distinct()
    )


class GeneCountFeature(L2GFeature):
    """Counts the number of genes within a specified window size from the study locus."""

    feature_dependency_type = TargetIndex
    feature_name = "geneCount500kb"

    @classmethod
    def compute(
        cls: type[GeneCountFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> GeneCountFeature:
        """Computes the gene count feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dictionary containing dependencies, with target index and window size

        Returns:
            GeneCountFeature: Feature dataset
        """
        genomic_window = 500000
        gene_count_df = common_genecount_feature_logic(
            study_loci_to_annotate=study_loci_to_annotate,
            feature_name=cls.feature_name,
            genomic_window=genomic_window,
            **feature_dependency,
        )

        return cls(
            _df=convert_from_wide_to_long(
                gene_count_df,
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class ProteinGeneCountFeature(L2GFeature):
    """Counts the number of protein coding genes within a specified window size from the study locus."""

    feature_dependency_type = TargetIndex
    feature_name = "proteinGeneCount500kb"

    @classmethod
    def compute(
        cls: type[ProteinGeneCountFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> ProteinGeneCountFeature:
        """Computes the gene count feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dictionary containing dependencies, with target index and window size

        Returns:
            ProteinGeneCountFeature: Feature dataset
        """
        genomic_window = 500000
        gene_count_df = common_genecount_feature_logic(
            study_loci_to_annotate=study_loci_to_annotate,
            feature_name=cls.feature_name,
            genomic_window=genomic_window,
            protein_coding_only=True,
            **feature_dependency,
        )

        return cls(
            _df=convert_from_wide_to_long(
                gene_count_df,
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class ProteinCodingFeature(L2GFeature):
    """Indicates whether a gene is protein-coding within a specified window size from the study locus."""

    feature_dependency_type = VariantIndex
    feature_name = "isProteinCoding"

    @classmethod
    def compute(
        cls: type[ProteinCodingFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> ProteinCodingFeature:
        """Computes the protein coding feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dictionary containing dependencies, including variant index

        Returns:
            ProteinCodingFeature: Feature dataset with 1 if the gene is protein-coding, 0 otherwise
        """
        genomic_window = 500_000
        protein_coding_df = is_protein_coding_feature_logic(
            study_loci_to_annotate=study_loci_to_annotate,
            feature_name=cls.feature_name,
            genomic_window=genomic_window,
            **feature_dependency,
        )

        return cls(
            _df=convert_from_wide_to_long(
                protein_coding_df,
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class CredibleSetConfidenceFeature(L2GFeature):
    """Distance of the sentinel variant to gene TSS. This is not weighted by the causal probability."""

    feature_dependency_type = [StudyLocus, VariantIndex]
    feature_name = "credibleSetConfidence"

    @classmethod
    def compute(
        cls: type[CredibleSetConfidenceFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> CredibleSetConfidenceFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            CredibleSetConfidenceFeature: Feature dataset
        """
        full_credible_set = feature_dependency["study_locus"].df.select(
            "studyLocusId",
            "studyId",
            f.explode("locus.variantId").alias("variantId"),
            cls.score_credible_set_confidence(f.col("confidence")).alias(
                cls.feature_name
            ),
        )

        return cls(
            _df=convert_from_wide_to_long(
                (
                    study_loci_to_annotate.df.drop("studyLocusId")
                    # Annotate genes
                    .join(
                        feature_dependency["variant_index"].df.select(
                            "variantId",
                            f.explode("transcriptConsequences.targetId").alias(
                                "geneId"
                            ),
                        ),
                        on="variantId",
                        how="inner",
                    )
                    # Annotate credible set confidence
                    .join(full_credible_set, ["variantId", "studyId"])
                    .select("studyLocusId", "geneId", cls.feature_name)
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )

    @classmethod
    def score_credible_set_confidence(
        cls: type[CredibleSetConfidenceFeature],
        confidence_column: Column,
    ) -> Column:
        """Expression that assigns a score to the credible set confidence.

        Args:
            confidence_column (Column): Confidence column in the StudyLocus object

        Returns:
            Column: A confidence score between 0 and 1
        """
        return (
            f.when(
                f.col("confidence")
                == CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD.value,
                f.lit(1.0),
            )
            .when(
                f.col("confidence")
                == CredibleSetConfidenceClasses.FINEMAPPED_OUT_OF_SAMPLE_LD.value,
                f.lit(0.75),
            )
            .when(
                f.col("confidence")
                == CredibleSetConfidenceClasses.PICSED_SUMMARY_STATS.value,
                f.lit(0.5),
            )
            .when(
                f.col("confidence")
                == CredibleSetConfidenceClasses.PICSED_TOP_HIT.value,
                f.lit(0.25),
            )
            .when(
                f.col("confidence") == CredibleSetConfidenceClasses.UNKNOWN.value,
                f.lit(0.0),
            )
        )
