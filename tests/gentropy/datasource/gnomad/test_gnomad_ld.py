"""Tests on LD matrix from GnomAD."""

from __future__ import annotations

from math import sqrt
from typing import Any

import hail as hl
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from gentropy.datasource.gnomad.ld import GnomADLDMatrix


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        # Normal scenario: the ld index contains all variants in the matrix
        (
            # Observed
            [
                (0, "varA", "chr1"),
                (1, "varB", "chr1"),
                (2, "varC", "chr1"),
            ],
            # Expected
            [
                (1.0, "varA", "chr1", "varA"),
                (0.7, "varA", "chr1", "varB"),
                (-0.7, "varA", "chr1", "varC"),
            ],
        ),
        # LD index is missing a variant in the matrix
        (
            # Observed
            [
                (0, "varA", "chr1"),
                (1, "varB", "chr1"),
            ],
            # Expected - the missing variant is ignored
            [
                (1.0, "varA", "chr1", "varA"),
                (0.7, "varA", "chr1", "varB"),
            ],
        ),
    ],
)
def test_resolve_variant_indices(
    spark: SparkSession, observed: list[Any], expected: list[Any]
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
        ["idx", "variantId", "chromosome"],
    )
    expected_df = spark.createDataFrame(
        expected,
        ["r", "variantId_i", "chromosome", "variantId_j"],
    )
    observed_df = GnomADLDMatrix._resolve_variant_indices(ld_index, ld_matrix)
    assert (
        observed_df.orderBy(observed_df["r"].desc()).collect() == expected_df.collect()
    )


class TestGnomADLDMatrixVariants:
    """Test the resolved LD variant sets."""

    def test_get_ld_slice_type(self: TestGnomADLDMatrixVariants) -> None:
        """Testing if ld_slice has the right type."""
        assert isinstance(self.ld_slice, DataFrame)

    def test_get_ld_empty_slice_type(self: TestGnomADLDMatrixVariants) -> None:
        """Testing if ld_empty_slice is None."""
        assert self.ld_empty_slice is None

    def test_get_ld_variants__square(
        self: TestGnomADLDMatrixVariants,
    ) -> None:
        """Testing if the function returns a square matrix."""
        if self.ld_slice is not None:
            assert sqrt(self.ld_slice.count()) == int(sqrt(self.ld_slice.count()))

    @pytest.fixture(autouse=True)
    def _setup(self: TestGnomADLDMatrixVariants, spark: SparkSession) -> None:
        """Prepares fixtures for the test."""
        hl.init(sc=spark.sparkContext, log="/dev/null", idempotent=True)

        ld_test_population = "test-pop"

        gnomad_ld_matrix = GnomADLDMatrix(
            ld_matrix_template="tests/gentropy/data_samples/example_{POP}.bm",
            ld_index_raw_template="tests/gentropy/data_samples/example_{POP}.ht",
            grch37_to_grch38_chain_path="tests/gentropy/data_samples/grch37_to_grch38.over.chain",
        )
        self.ld_slice = gnomad_ld_matrix.get_ld_variants(
            gnomad_ancestry=ld_test_population,
            chromosome="1",
            start=10025,
            end=10075,
        )
        self.ld_empty_slice = gnomad_ld_matrix.get_ld_variants(
            gnomad_ancestry=ld_test_population,
            chromosome="1",
            start=0,
            end=1,
        )


class TestGnomADLDMatrixSlice:
    """Test GnomAD LD methods."""

    def test_get_ld_matrix_slice__diagonal(self: TestGnomADLDMatrixSlice) -> None:
        """Test LD matrix slice."""
        assert (
            self.matrix_slice.filter(f.col("idx_i") == f.col("idx_j"))
            .select("r")
            .distinct()
            .collect()[0]["r"]
            == 1.0
        ), "The matrix does not have ones in the diagonal."

    def test_get_ld_matrix_slice__count(self: TestGnomADLDMatrixSlice) -> None:
        """Test LD matrix slice."""
        # As the slicing of the matrix is inclusive, the total number of rows are calculated as follows:
        included_indices = self.slice_end_index - self.slice_start_index + 1
        expected_pariwise_count = included_indices**2

        assert self.matrix_slice.count() == expected_pariwise_count, (
            "The matrix is not complete."
        )

    def test_get_ld_matrix_slice__type(self: TestGnomADLDMatrixSlice) -> None:
        """Test LD matrix slice."""
        assert isinstance(self.matrix_slice, DataFrame), (
            "The returned data is not a dataframe."
        )

    def test_get_ld_matrix_slice__symmetry(
        self: TestGnomADLDMatrixSlice,
    ) -> None:
        """Test LD matrix slice."""
        # Testing square matrix completeness and symmetry:
        compared = self.matrix_slice.join(
            (
                self.matrix_slice.select(
                    f.col("idx_i").alias("idx_j"),
                    f.col("idx_j").alias("idx_i"),
                    f.col("r").alias("r_sym"),
                )
            ),
            on=["idx_i", "idx_j"],
            how="inner",
        )

        assert compared.count() == self.matrix_slice.count(), (
            "The matrix is not complete."
        )
        assert (
            compared.filter(f.col("r") == f.col("r_sym")).count() == compared.count()
        ), "The matrix is not symmetric."

    @pytest.fixture(autouse=True)
    def _setup(self: TestGnomADLDMatrixSlice, spark: SparkSession) -> None:
        """Prepares fixtures for the test."""
        hl.init(sc=spark.sparkContext, log="/dev/null", idempotent=True)
        gnomad_ld_matrix = GnomADLDMatrix(
            ld_matrix_template="tests/gentropy/data_samples/example_{POP}.bm"
        )
        test_ld_population: str = "test-pop"
        self.slice_start_index: int = 1
        self.slice_end_index: int = 2

        self.matrix_slice = gnomad_ld_matrix.get_ld_matrix_slice(
            gnomad_ancestry=test_ld_population,
            start_index=self.slice_start_index,
            end_index=self.slice_end_index,
        )
