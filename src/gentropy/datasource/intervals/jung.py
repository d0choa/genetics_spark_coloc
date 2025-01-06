"""Interval dataset from Jung et al. 2019."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from gentropy.dataset.intervals import Intervals

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from gentropy.common.Liftover import LiftOverSpark
    from gentropy.dataset.target_index import TargetIndex


class IntervalsJung:
    """Interval dataset from Jung et al. 2019."""

    @staticmethod
    def read(spark: SparkSession, path: str) -> DataFrame:
        """Read jung dataset.

        Args:
            spark (SparkSession): Spark session
            path (str): Path to dataset

        Returns:
            DataFrame: DataFrame with raw jung data
        """
        return spark.read.csv(path, sep=",", header=True)

    @classmethod
    def parse(
        cls: type[IntervalsJung],
        jung_raw: DataFrame,
        target_index: TargetIndex,
        lift: LiftOverSpark,
    ) -> Intervals:
        """Parse the Jung et al. 2019 dataset.

        Args:
            jung_raw (DataFrame): raw Jung et al. 2019 dataset
            target_index (TargetIndex): Target index
            lift (LiftOverSpark): LiftOverSpark instance

        Returns:
            Intervals: Interval dataset containing Jung et al. 2019 data
        """
        dataset_name = "jung2019"
        experiment_type = "pchic"
        pmid = "31501517"

        # Lifting over the coordinates:
        return Intervals(
            _df=(
                jung_raw.withColumn(
                    "interval", f.split(f.col("Interacting_fragment"), r"\.")
                )
                .select(
                    # Parsing intervals:
                    f.regexp_replace(f.col("interval")[0], "chr", "").alias("chrom"),
                    f.col("interval")[1].cast(t.IntegerType()).alias("start"),
                    f.col("interval")[2].cast(t.IntegerType()).alias("end"),
                    # Extract other columns:
                    f.col("Promoter").alias("gene_name"),
                    f.col("Tissue_type").alias("tissue"),
                )
                # Lifting over to GRCh38 interval 1:
                .transform(
                    lambda df: lift.convert_intervals(df, "chrom", "start", "end")
                )
                .select(
                    "chrom",
                    f.col("mapped_start").alias("start"),
                    f.col("mapped_end").alias("end"),
                    f.explode(f.split(f.col("gene_name"), ";")).alias("gene_name"),
                    "tissue",
                )
                .alias("intervals")
                # Joining with genes:
                .join(
                    target_index.symbols_lut().alias("genes"),
                    on=[f.col("intervals.gene_name") == f.col("genes.geneSymbol")],
                    how="inner",
                )
                # Finalize dataset:
                .select(
                    "chromosome",
                    f.col("intervals.start").alias("start"),
                    f.col("intervals.end").alias("end"),
                    "geneId",
                    f.col("tissue").alias("biofeature"),
                    f.lit(1.0).alias("score"),
                    f.lit(dataset_name).alias("datasourceId"),
                    f.lit(experiment_type).alias("datatypeId"),
                    f.lit(pmid).alias("pmid"),
                )
                .drop_duplicates()
            ),
            _schema=Intervals.get_schema(),
        )
