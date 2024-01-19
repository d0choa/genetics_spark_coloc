"""Tests for study index dataset from FinnGen."""

from __future__ import annotations

from gentropy.dataset.summary_statistics import SummaryStatistics
from gentropy.datasource.finngen.summary_stats import FinnGenSummaryStats
from pyspark.sql import SparkSession


def test_finngen_summary_stats_from_source(spark: SparkSession) -> None:
    """Test summary statistics from source."""
    assert isinstance(
        FinnGenSummaryStats.from_source(
            spark=spark,
            raw_file="tests/data_samples/finngen_R9_AB1_ACTINOMYCOSIS.gz",
        ),
        SummaryStatistics,
    )
