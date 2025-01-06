"""Parser for OTPlatform target dataset."""
from __future__ import annotations

import pyspark.sql.functions as f
from pyspark.sql import Column, DataFrame

from gentropy.dataset.target_index import TargetIndex


class OpenTargetsTarget:
    """Parser for OTPlatform target dataset.

    Genomic data from Open Targets provides gene identification and genomic coordinates that are integrated into the target index of our ETL pipeline.

    The EMBL-EBI Ensembl database is used as a source for human targets in the Platform, with the Ensembl gene ID as the primary identifier. The criteria for target inclusion is:
    - Genes from all biotypes encoded in canonical chromosomes
    - Genes in alternative assemblies encoding for a reviewed protein product.
    """

    @staticmethod
    def _get_gene_tss(strand_col: Column, start_col: Column, end_col: Column) -> Column:
        """Returns the TSS of a gene based on its orientation.

        Args:
            strand_col (Column): Column containing 1 if the coding strand of the gene is forward, and -1 if it is reverse.
            start_col (Column): Column containing the start position of the gene.
            end_col (Column): Column containing the end position of the gene.

        Returns:
            Column: Column containing the TSS of the gene.

        Examples:
            >>> df = spark.createDataFrame([{"strand": 1, "start": 100, "end": 200}, {"strand": -1, "start": 100, "end": 200}])
            >>> df.withColumn("tss", OpenTargetsTarget._get_gene_tss(f.col("strand"), f.col("start"), f.col("end"))).show()
            +---+-----+------+---+
            |end|start|strand|tss|
            +---+-----+------+---+
            |200|  100|     1|100|
            |200|  100|    -1|200|
            +---+-----+------+---+
            <BLANKLINE>

        """
        return f.when(strand_col == 1, start_col).when(strand_col == -1, end_col)

    @classmethod
    def as_target_index(
        cls: type[OpenTargetsTarget], target_index: DataFrame
    ) -> TargetIndex:
        """Initialise TargetIndex from source dataset.

        Args:
            target_index (DataFrame): Target index dataframe

        Returns:
            TargetIndex: Target index dataset
        """
        return TargetIndex(
            _df=target_index.select(
                f.coalesce(f.col("id"), f.lit("unknown")).alias("geneId"),
                "approvedSymbol",
                "approvedName",
                "biotype",
                f.col("obsoleteSymbols.label").alias("obsoleteSymbols"),
                f.coalesce(f.col("genomicLocation.chromosome"), f.lit("unknown")).alias(
                    "chromosome"
                ),
                OpenTargetsTarget._get_gene_tss(
                    f.col("genomicLocation.strand"),
                    f.col("genomicLocation.start"),
                    f.col("genomicLocation.end"),
                ).alias("tss"),
                f.col("genomicLocation.start").alias("start"),
                f.col("genomicLocation.end").alias("end"),
                f.col("genomicLocation.strand").alias("strand"),
            ),
            _schema=TargetIndex.get_schema(),
        )
