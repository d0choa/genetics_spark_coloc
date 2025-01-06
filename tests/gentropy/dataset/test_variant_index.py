"""Tests on variant index generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyspark.sql import functions as f
from pyspark.sql import types as t

from gentropy.dataset.variant_index import VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_variant_index_creation(mock_variant_index: VariantIndex) -> None:
    """Test variant index creation with mock variant index."""
    assert isinstance(mock_variant_index, VariantIndex)


class TestVariantIndex:
    """Collection of tests around the functionality and shape of the variant index."""

    MOCK_ANNOTATION_DATA = [
        ("v1", "c1", 2, "T", "A", ["rs5"], "really bad consequence"),
        (
            "v4_long",
            "c1",
            5,
            "T",
            "A",
            ["rs6"],
            "mild consequence",
        ),  # should be hashed automatically
    ]

    MOCK_DATA = [
        ("v1", "c1", 2, "T", "A", ["rs1"]),
        ("v2", "c1", 3, "T", "A", ["rs2", "rs3"]),
        ("v3", "c1", 4, "T", "A", None),
        ("v4_long", "c1", 5, "T", "A", None),  # should be hashed automatically
    ]

    MOCK_SCHEMA = t.StructType(
        [
            t.StructField("variantId", t.StringType(), False),
            t.StructField("chromosome", t.StringType(), False),
            t.StructField("position", t.IntegerType(), False),
            t.StructField("referenceAllele", t.StringType(), False),
            t.StructField("alternateAllele", t.StringType(), False),
            t.StructField("rsIds", t.ArrayType(t.StringType(), True), True),
        ]
    )

    # The mock annotation has an extra column that needs to be propagated to the annotated data:
    MOCK_ANNOTATION_SCHEMA = t.StructType(
        [
            t.StructField("variantId", t.StringType(), False),
            t.StructField("chromosome", t.StringType(), False),
            t.StructField("position", t.IntegerType(), False),
            t.StructField("referenceAllele", t.StringType(), False),
            t.StructField("alternateAllele", t.StringType(), False),
            t.StructField("rsIds", t.ArrayType(t.StringType(), True), True),
            t.StructField("mostSevereConsequenceId", t.StringType(), False),
        ]
    )

    @pytest.fixture(autouse=True)
    def _setup(self: TestVariantIndex, spark: SparkSession) -> None:
        """Setting up the test.

        Args:
            spark (SparkSession): Spark session.
        """
        # Create dataframe:
        self.df = spark.createDataFrame(self.MOCK_DATA, schema=self.MOCK_SCHEMA)
        # Loading variant index:
        self.variant_index = VariantIndex(
            _df=self.df, _schema=VariantIndex.get_schema(), id_threshold=2
        )

        # Loading annotation variant index:
        self.annotation = VariantIndex(
            _df=spark.createDataFrame(
                self.MOCK_ANNOTATION_DATA, schema=self.MOCK_ANNOTATION_SCHEMA
            ),
            _schema=VariantIndex.get_schema(),
            id_threshold=2,
        )

    def test_init_type(self: TestVariantIndex) -> None:
        """Just make sure the right datatype is created."""
        assert isinstance(self.variant_index, VariantIndex)

    def test_removed_null_values(self: TestVariantIndex) -> None:
        """Making sure upon initialisation, the null values are removed and were replaced with empty arrays."""
        # RsIds column cannot be null:
        assert self.variant_index.df.filter(f.col("rsIds").isNull()).count() == 0

        # However can be empty array:
        assert self.variant_index.df.filter(f.size("rsIds") == 0).count() > 0

    def test_annotation_return_type(self: TestVariantIndex) -> None:
        """Make sure the annotation method returns a dataframe."""
        assert isinstance(
            self.variant_index.add_annotation(self.annotation), VariantIndex
        )

    def test_new_column_added(self: TestVariantIndex) -> None:
        """Make sure the annotation method adds a new column."""
        assert (
            "mostSevereConsequenceId"
            in self.variant_index.add_annotation(self.annotation).df.columns
        )

    def test_new_column_correct(self: TestVariantIndex) -> None:
        """Make sure the annotation method adds the correct values."""
        assert (
            self.variant_index.add_annotation(self.annotation)
            .df.filter(f.col("mostSevereConsequenceId").isNotNull())
            .count()
            == 2
        )

    def test_rsid_column_updated(self: TestVariantIndex) -> None:
        """Make sure the annotation method updates the rsId column."""
        # RsId added to a new row:
        assert (
            self.variant_index.add_annotation(self.annotation)
            .df.filter(f.size("rsIds") > 0)
            .count()
            == 3
        )

        # RsID added to an existing row:
        assert (
            self.variant_index.add_annotation(self.annotation)
            .df.filter(f.size("rsIds") > 1)
            .count()
            == 2
        )

    def test_variantid_column_hashed(self: TestVariantIndex) -> None:
        """Make sure the variantId column is hashed during initialisation. Threshold is set to 2, so var_4_long should be hashed."""
        assert (
            self.variant_index.df.filter(f.col("variantId").startswith("OTVAR")).count()
            != 0
        )

    @pytest.mark.parametrize(
        "distance_type", ["distanceFromTss", "distanceFromFootprint"]
    )
    def test_get_distance_to_gene(
        self: TestVariantIndex, mock_variant_index: VariantIndex, distance_type: str
    ) -> None:
        """Assert that the function returns a df with the requested columns."""
        expected_cols = ["variantId", "targetId", distance_type]
        observed = mock_variant_index.get_distance_to_gene(distance_type=distance_type)
        for col in expected_cols:
            assert col in observed.columns, f"Column {col} not in {observed.columns}"

    def test_get_loftee(
        self: TestVariantIndex, mock_variant_index: VariantIndex
    ) -> None:
        """Assert that the function returns a df with the requested columns."""
        expected_cols = [
            "variantId",
            "targetId",
            "lofteePrediction",
            "isHighQualityPlof",
        ]
        observed = mock_variant_index.get_loftee()
        for col in expected_cols:
            assert col in observed.columns, f"Column {col} not in {observed.columns}"
