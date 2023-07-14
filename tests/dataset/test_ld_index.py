"""Tests on LD index."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from otg.dataset.ld_index import LDIndex

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_ld_index_creation(mock_ld_index: LDIndex) -> None:
    """Test ld index creation with mock ld index."""
    assert isinstance(mock_ld_index, LDIndex)


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        # Normal scenario: the ld index contains all variants in the matrix
        (
            # Observed
            [
                (0, "varA"),
                (1, "varB"),
                (2, "varC"),
            ],
            # Expected
            [
                (1.0, "varA", "varA"),
                (0.7, "varA", "varB"),
                (-0.7, "varA", "varC"),
            ],
        ),
        # The ld index contains a subset of the variants in the matrix
        (
            # Observed
            [
                (0, "varA"),
                (1, "varB"),
            ],
            # Expected - the missing variant is ignored
            [
                (1.0, "varA", "varA"),
                (0.7, "varA", "varB"),
            ],
        ),
    ],
)
def test_resolve_variant_indices(
    spark: SparkSession, observed: list, expected: list
) -> None:
    """Test _resolve_variant_indices."""
    ld_matrix = spark.createDataFrame(
        [
            (0, 0, 1.0),
            (0, 1, 0.7),
            (0, 2, -0.7),
        ],
        ["i", "j", "r"],
    )
    ld_index = spark.createDataFrame(
        observed,
        ["idx", "variantId"],
    )
    expected_df = spark.createDataFrame(
        expected,
        ["r", "variantId_i", "variantId_j"],
    )
    observed_df = LDIndex._resolve_variant_indices(ld_index, ld_matrix)
    assert (
        observed_df.orderBy(observed_df["r"].desc()).collect() == expected_df.collect()
    )
