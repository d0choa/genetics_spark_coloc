"""Interval dataset from Thurman et al. 2019."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from otg.dataset.intervals import Intervals

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from otg.common.Liftover import LiftOverSpark
    from otg.dataset.gene_index import GeneIndex


class IntervalsThurnman(Intervals):
    """Interval dataset from Thurman et al. 2012."""

    @staticmethod
    def read_thurnman(spark: SparkSession, path: str) -> DataFrame:
        """Read thurnman dataset.

        Args:
            spark (SparkSession): Spark session
            path (str): Path to dataset

        Returns:
            DataFrame: DataFrame with raw thurnman data
        """
        thurman_schema = t.StructType(
            [
                t.StructField("gene_chr", t.StringType(), False),
                t.StructField("gene_start", t.IntegerType(), False),
                t.StructField("gene_end", t.IntegerType(), False),
                t.StructField("gene_name", t.StringType(), False),
                t.StructField("chrom", t.StringType(), False),
                t.StructField("start", t.IntegerType(), False),
                t.StructField("end", t.IntegerType(), False),
                t.StructField("score", t.FloatType(), False),
            ]
        )
        return spark.read.csv(path, sep="\t", header=True, schema=thurman_schema)

    @classmethod
    def parse(
        cls: type[IntervalsThurnman],
        thurnman_raw: DataFrame,
        gene_index: GeneIndex,
        lift: LiftOverSpark,
    ) -> Intervals:
        """Parse the Thurman et al. 2012 dataset.

        Args:
            thurnman_raw (DataFrame): raw Thurman et al. 2019 dataset
            gene_index (GeneIndex): gene index
            lift (LiftOverSpark): LiftOverSpark instance

        Returns:
            Intervals: Interval dataset containing Thurnman et al. 2012 data
        """
        dataset_name = "thurman2012"
        experiment_type = "dhscor"
        pmid = "22955617"

        return cls(
            _df=(
                thurnman_raw.select(
                    f.regexp_replace(f.col("chrom"), "chr", "").alias("chrom"),
                    "start",
                    "end",
                    "gene_name",
                    "score",
                )
                # Lift over to the GRCh38 build:
                .transform(
                    lambda df: lift.convert_intervals(df, "chrom", "start", "end")
                )
                .alias("intervals")
                # Map gene names to gene IDs:
                .join(
                    gene_index.symbols_lut().alias("genes"),
                    on=[
                        f.col("intervals.gene_name") == f.col("genes.geneSymbol"),
                        f.col("intervals.chrom") == f.col("genes.chromosome"),
                    ],
                    how="inner",
                )
                # Select relevant columns and add constant columns:
                .select(
                    f.col("chrom").alias("chromosome"),
                    f.col("mapped_start").alias("start"),
                    f.col("mapped_end").alias("end"),
                    "geneId",
                    f.col("score").cast(t.DoubleType()).alias("resourceScore"),
                    f.lit(dataset_name).alias("datasourceId"),
                    f.lit(experiment_type).alias("datatypeId"),
                    f.lit(pmid).alias("pmid"),
                )
                .distinct()
            ),
            _schema=cls.get_schema(),
        )
