"""Test study index dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import types as t

from gentropy.common.genomic_region import GenomicRegion
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_summary_statistics__creation(
    mock_summary_statistics: SummaryStatistics,
) -> None:
    """Test summary statistics creation with mock summary statistics."""
    assert isinstance(mock_summary_statistics, SummaryStatistics)


def test_summary_statistics__pval_filter__return_type(
    mock_summary_statistics: SummaryStatistics,
) -> None:
    """Test if the p-value filter indeed returns summary statistics object."""
    pval_threshold = 5e-3
    assert isinstance(
        mock_summary_statistics.pvalue_filter(pval_threshold), SummaryStatistics
    )


def test_summary_statistics__window_based_clumping__return_type(
    mock_summary_statistics: SummaryStatistics,
) -> None:
    """Test if the window-based clumping indeed returns study locus object."""
    assert isinstance(
        mock_summary_statistics.window_based_clumping(250_000), StudyLocus
    )


def test_summary_statistics__exclude_region__return_type(
    mock_summary_statistics: SummaryStatistics,
) -> None:
    """Testing if the exclude region method returns the right datatype."""
    assert isinstance(
        mock_summary_statistics.exclude_region(
            GenomicRegion.from_string("chr12:124-1245")
        ),
        SummaryStatistics,
    )


def test_summary_statistics__exclude_region__correctness(
    spark: SparkSession,
) -> None:
    """Testing if the exclude region method returns the right datatype."""
    data = [
        # Region needs to be dropped:
        ("s1", "c1", "v1", 1, 1.0, -2, 0.0),
        ("s1", "c1", "v1", 10, 1.0, -2, 0.0),
        ("s1", "c1", "v1", 15, 1.0, -2, 0.0),
        ("s1", "c1", "v1", 20, 1.0, -2, 0.0),
        # Same region on different chromosome - should stay
        ("s1", "c2", "v1", 1, 1.0, -2, 0.0),
        ("s1", "c2", "v1", 10, 1.0, -2, 0.0),
        ("s1", "c2", "v1", 15, 1.0, -2, 0.0),
        ("s1", "c2", "v1", 20, 1.0, -2, 0.0),
        # Same region on different study - should be filtered
        ("s2", "c1", "v1", 1, 1.0, -2, 0.0),
        ("s2", "c1", "v1", 10, 1.0, -2, 0.0),
        ("s2", "c1", "v1", 15, 1.0, -2, 0.0),
        ("s2", "c1", "v1", 20, 1.0, -2, 0.0),
    ]

    schema = t.StructType(
        [
            t.StructField("studyId", t.StringType(), False),
            t.StructField("chromosome", t.StringType(), False),
            t.StructField("variantId", t.StringType(), False),
            t.StructField("position", t.IntegerType(), False),
            t.StructField("pValueMantissa", t.FloatType(), False),
            t.StructField("pValueExponent", t.IntegerType(), False),
            t.StructField("beta", t.DoubleType(), False),
        ]
    )
    # Create dataframe and apply region based filter:
    df = spark.createDataFrame(data, schema=schema)
    filtered_sumstas = SummaryStatistics(
        _df=df, _schema=SummaryStatistics.get_schema()
    ).exclude_region(GenomicRegion.from_string("c1:9-16"))

    # Test for the correct number of rows returned:
    assert filtered_sumstas.df.count() == 8


# Look for medium size summary statistics -
