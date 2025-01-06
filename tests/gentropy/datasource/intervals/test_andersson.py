"""Test Andersson Intervals."""

from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, SparkSession

from gentropy.common.Liftover import LiftOverSpark
from gentropy.dataset.intervals import Intervals
from gentropy.dataset.target_index import TargetIndex
from gentropy.datasource.intervals.andersson import IntervalsAndersson


@pytest.fixture(scope="module")
def sample_intervals_andersson(spark: SparkSession) -> DataFrame:
    """Sample Andersson intervals."""
    return IntervalsAndersson.read(
        spark, "tests/gentropy/data_samples/andersson_sample.bed"
    )


def test_read_andersson(sample_intervals_andersson: DataFrame) -> None:
    """Test read Andersson data."""
    assert isinstance(sample_intervals_andersson, DataFrame)


def test_andersson_intervals_from_source(
    sample_intervals_andersson: DataFrame,
    mock_target_index: TargetIndex,
    liftover_chain_37_to_38: LiftOverSpark,
) -> None:
    """Test AnderssonIntervals creation with mock data."""
    assert isinstance(
        IntervalsAndersson.parse(
            sample_intervals_andersson, mock_target_index, liftover_chain_37_to_38
        ),
        Intervals,
    )
