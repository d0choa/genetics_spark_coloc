"""Test Thurman."""

from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, SparkSession

from gentropy.common.Liftover import LiftOverSpark
from gentropy.dataset.intervals import Intervals
from gentropy.dataset.target_index import TargetIndex
from gentropy.datasource.intervals.thurman import IntervalsThurman


@pytest.fixture(scope="module")
def sample_intervals_thurman(spark: SparkSession) -> DataFrame:
    """Sample Thurman intervals."""
    return IntervalsThurman.read(
        spark, "tests/gentropy/data_samples/thurman_sample.bed8"
    )


def test_read_thurman(sample_intervals_thurman: DataFrame) -> None:
    """Test read Thurman data."""
    assert isinstance(sample_intervals_thurman, DataFrame)


def test_thurman_intervals_from_source(
    sample_intervals_thurman: DataFrame,
    mock_target_index: TargetIndex,
    liftover_chain_37_to_38: LiftOverSpark,
) -> None:
    """Test IntervalsThurman creation with mock data."""
    assert isinstance(
        IntervalsThurman.parse(
            sample_intervals_thurman, mock_target_index, liftover_chain_37_to_38
        ),
        Intervals,
    )
