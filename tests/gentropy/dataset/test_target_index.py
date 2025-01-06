"""Tests on LD index."""

from __future__ import annotations

from pyspark.sql import DataFrame

from gentropy.dataset.target_index import TargetIndex


def test_target_index_creation(mock_target_index: TargetIndex) -> None:
    """Test target index creation with mock target index."""
    assert isinstance(mock_target_index, TargetIndex)


def test_target_index_location_lut(mock_target_index: TargetIndex) -> None:
    """Test target index location lut."""
    assert isinstance(mock_target_index.locations_lut(), DataFrame)


def test_target_index_symbols_lut(mock_target_index: TargetIndex) -> None:
    """Test target index symbols lut."""
    assert isinstance(mock_target_index.symbols_lut(), DataFrame)


def test_target_index_filter_by_biotypes(mock_target_index: TargetIndex) -> None:
    """Test target index filter by biotypes."""
    assert isinstance(
        mock_target_index.filter_by_biotypes(
            biotypes=["protein_coding", "3prime_overlapping_ncRNA", "antisense"]
        ),
        TargetIndex,
    )
