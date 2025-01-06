"""Interval dataset from Thurman et al. 2019."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from gentropy.dataset.intervals import Intervals

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from gentropy.common.Liftover import LiftOverSpark
    from gentropy.dataset.target_index import TargetIndex


class IntervalsThurman:
    """Interval dataset from Thurman et al. 2012."""

    @staticmethod
    def read(spark: SparkSession, path: str) -> DataFrame:
        """Read thurman dataset.

        Args:
            spark (SparkSession): Spark session
            path (str): Path to dataset

        Returns:
            DataFrame: DataFrame with raw thurman data
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
        return spark.read.csv(path, sep="\t", header=False, schema=thurman_schema)

    @classmethod
    def parse(
        cls: type[IntervalsThurman],
        thurman_raw: DataFrame,
        target_index: TargetIndex,
        lift: LiftOverSpark,
    ) -> Intervals:
        """Parse the Thurman et al. 2012 dataset.

        Args:
            thurman_raw (DataFrame): raw Thurman et al. 2019 dataset
            target_index (TargetIndex): Target index
            lift (LiftOverSpark): LiftOverSpark instance

        Returns:
            Intervals: Interval dataset containing Thurman et al. 2012 data
        """
        dataset_name = "thurman2012"
        experiment_type = "dhscor"
        pmid = "22955617"

        return Intervals(
            _df=(
                thurman_raw.select(
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
                    target_index.symbols_lut().alias("genes"),
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
            _schema=Intervals.get_schema(),
        )
