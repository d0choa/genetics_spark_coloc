"""Summary statistics ingestion for FinnGen."""

from __future__ import annotations

from dataclasses import dataclass

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from gentropy.common.utils import parse_pvalue
from gentropy.dataset.summary_statistics import SummaryStatistics


@dataclass
class FinnGenSummaryStats:
    """Summary statistics dataset for FinnGen."""

    raw_schema: t.StructType = StructType(
        [
            StructField("#chrom", StringType(), True),
            StructField("pos", StringType(), True),
            StructField("ref", StringType(), True),
            StructField("alt", StringType(), True),
            StructField("rsids", StringType(), True),
            StructField("nearest_genes", StringType(), True),
            StructField("pval", StringType(), True),
            StructField("mlogp", StringType(), True),
            StructField("beta", StringType(), True),
            StructField("sebeta", StringType(), True),
            StructField("af_alt", StringType(), True),
            StructField("af_alt_cases", StringType(), True),
            StructField("af_alt_controls", StringType(), True),
        ]
    )

    @classmethod
    def from_source(
        cls: type[FinnGenSummaryStats],
        spark: SparkSession,
        raw_file: str,
    ) -> SummaryStatistics:
        """Ingests all summary statst for all FinnGen studies.

        Args:
            spark (SparkSession): Spark session object.
            raw_file (str): Path to raw summary statistics .gz files.

        Returns:
            SummaryStatistics: Processed summary statistics dataset
        """
        study_id = raw_file.split("/")[-1].split(".")[0].upper()
        processed_summary_stats_df = (
            spark.read.schema(cls.raw_schema)
            .option("delimiter", "\t")
            .csv(raw_file, header=True)
            # Drop rows which don't have proper position.
            .filter(f.col("pos").cast(t.IntegerType()).isNotNull())
            .select(
                # From the full path, extracts just the filename, and converts to upper case to get the study ID.
                f.lit(study_id).alias("studyId"),
                # Add variant information.
                f.concat_ws(
                    "_",
                    f.col("#chrom"),
                    f.col("pos"),
                    f.col("ref"),
                    f.col("alt"),
                ).alias("variantId"),
                f.col("#chrom").alias("chromosome"),
                f.col("pos").cast(t.IntegerType()).alias("position"),
                # Parse p-value into mantissa and exponent.
                *parse_pvalue(f.col("pval")),
                # Add beta, standard error, and allele frequency information.
                f.col("beta").cast("double"),
                f.col("sebeta").cast("double").alias("standardError"),
                f.col("af_alt").cast("float").alias("effectAlleleFrequencyFromSource"),
            )
            # Calculating the confidence intervals.
            .filter(
                f.col("pos").cast(t.IntegerType()).isNotNull() & (f.col("beta") != 0)
            )
            # Average ~20Mb partitions with 30 partitions per study
            .repartitionByRange(30, "chromosome", "position")
            .sortWithinPartitions("chromosome", "position")
        )

        # Initializing summary statistics object:
        return SummaryStatistics(
            _df=processed_summary_stats_df,
            _schema=SummaryStatistics.get_schema(),
        )
