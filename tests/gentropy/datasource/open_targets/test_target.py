"""Test Open Targets target data source."""

from __future__ import annotations

from pyspark.sql import DataFrame

from gentropy.dataset.target_index import TargetIndex
from gentropy.datasource.open_targets.target import OpenTargetsTarget


def test_open_targets_as_target_index(sample_target_index: DataFrame) -> None:
    """Test target index from source."""
    assert isinstance(
        OpenTargetsTarget.as_target_index(sample_target_index), TargetIndex
    )
