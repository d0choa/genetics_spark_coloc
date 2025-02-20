"""Test locus-breaker clumping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyspark.sql import functions as f
from pyspark.sql import types as t

from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_locus_breaker_return_type(
    mock_summary_statistics: SummaryStatistics,
) -> None:
    """Test locus clumping."""
    assert isinstance(
        mock_summary_statistics.locus_breaker_clumping(),
        StudyLocus,
    )


class TestLocusBreakerClumping:
    """Test locus breaker clumping."""

    # Some constants for testing:
    distance_cutoff = 3
    pvalue_baseline_cutoff = 0.05
    pvalue_cutoff = 1e-3
    flanking = 2

    @pytest.fixture(scope="class")
    def mock_input(
        self: TestLocusBreakerClumping,
        spark: SparkSession,
    ) -> SummaryStatistics:
        """Prepare mock summary statistics for clumping."""
        data = [
            # Block 1: Study1, chromosome 1, expected boundaries: 0-5
            ("s1", "chr1", "v1", 1, -2),
            ("s1", "chr1", "v1", 2, -4),
            ("s1", "chr1", "top_loci", 3, -5),
            ("s1", "chr1", "v1", 4, -1),  # <- will be dropped: not reaching baseline
            # Block 2: Study1, chromosome 1, expected boundaries: 5-10
            ("s1", "chr1", "top_loci", 7, -4),
            ("s1", "chr1", "v1", 8, -2),
            # Block 3: Study1, chromosome 2, expected boundaries: 6-12
            ("s1", "chr2", "v1", 8, -2),
            ("s1", "chr2", "v1", 9, -3),
            ("s1", "chr2", "top_loci", 10, -5),
            # Block 4: Study1, chromosome 2
            ("s1", "chr2", "v1", 14, -2),  # <- will be dropped: low p-value
            # Block 5: Study2, chromosome 2, expected boundaries: 6-12
            ("s2", "chr2", "v1", 8, -2),
            ("s2", "chr2", "v1", 9, -3),
            ("s2", "chr2", "top_loci", 10, -6),
            # Block 6: Study2, chromosome 2, expected boundaries: 12-16
            ("s2", "chr2", "top_loci", 14, -5),
        ]
        df = spark.createDataFrame(
            data, ["studyId", "chromosome", "variantId", "position", "pValueExponent"]
        ).select(
            "studyId",
            "chromosome",
            "variantId",
            f.col("position").cast(t.IntegerType()),
            f.col("pValueExponent").cast(t.IntegerType()),
            f.lit(1.0).cast(t.FloatType()).alias("pValueMantissa"),
            f.lit(1.0).cast(t.DoubleType()).alias("beta"),
        )

        return SummaryStatistics(_df=df, _schema=SummaryStatistics.get_schema())

    @pytest.fixture(scope="class")
    def clumped_data(
        self: TestLocusBreakerClumping, mock_input: SummaryStatistics
    ) -> StudyLocus:
        """Apply method and store for clumped data."""
        return mock_input.locus_breaker_clumping(
            self.pvalue_baseline_cutoff,
            self.distance_cutoff,
            self.pvalue_cutoff,
            self.flanking,
        ).persist()

    def test_return_type(
        self: TestLocusBreakerClumping, clumped_data: StudyLocus
    ) -> None:
        """Testing return type."""
        assert isinstance(clumped_data, StudyLocus), (
            f"Unexpected return type: {type(clumped_data)}"
        )

    def test_number_of_loci(
        self: TestLocusBreakerClumping, clumped_data: StudyLocus
    ) -> None:
        """Testing return type."""
        assert clumped_data.df.count() == 5, (
            f"Unexpected number of loci: {clumped_data.df.count()}"
        )

    def test_top_loci(self: TestLocusBreakerClumping, clumped_data: StudyLocus) -> None:
        """Testing selected top-loci."""
        top_loci_variants = clumped_data.df.select("variantId").distinct().collect()

        assert len(top_loci_variants) == 1, (
            f"Unexpected number of top loci: {len(top_loci_variants)} ({top_loci_variants})"
        )

        assert top_loci_variants[0]["variantId"] == "top_loci", (
            f"Unexpected top locus: {top_loci_variants[0]['variantId']}"
        )

    def test_locus_boundaries(
        self: TestLocusBreakerClumping, clumped_data: StudyLocus
    ) -> None:
        """Testing locus boundaries."""
        locus_start = [
            row["locusStart"] for row in clumped_data.df.select("locusStart").collect()
        ]

        locus_end = [
            row["locusEnd"] for row in clumped_data.df.select("locusEnd").collect()
        ]

        assert locus_start == [0, 5, 6, 6, 12], f"Unexpected locus start: {locus_start}"

        assert locus_end == [5, 10, 12, 12, 16], f"Unexpected locus end: {locus_end}"
