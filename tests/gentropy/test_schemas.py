"""Tests on spark schemas."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import StructType

from gentropy.common.schemas import SchemaValidationError

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

    from gentropy.dataset.l2g_prediction import L2GPrediction
    from gentropy.dataset.target_index import TargetIndex

SCHEMA_DIR = "src/gentropy/assets/schemas"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Testing accross all schemas.

    Pytest hook to parametrise testing

    Args:
        metafunc (pytest.Metafunc): pytest metafunc
    """
    if "schema_json" in metafunc.fixturenames:
        schemas = [f for f in os.listdir(SCHEMA_DIR) if f.endswith(".json")]
        metafunc.parametrize("schema_json", schemas)


def test_schema(schema_json: str) -> None:
    """Test schema is a valid Spark schema.

    Args:
        schema_json (str): schema filename
    """
    core_schema = json.loads(Path(SCHEMA_DIR, schema_json).read_text(encoding="utf-8"))
    isinstance(StructType.fromJson(core_schema), StructType)


def is_camelcase(identifier: str) -> bool:
    """Use a regular expression to check if the identifier is in camelCase.

    CamelCase starts with a lowercase letter and has uppercase letters in between. A lowercase letter can also be followed by a number.
    """
    return re.match(r"^[a-z]+(?:[A-Z0-9][a-z0-9]*)*$", identifier) is not None


def test_schema_columns_camelcase(schema_json: str) -> None:
    """Test schema column names are in camelCase.

    Args:
        schema_json (str): schema filename
    """
    if schema_json == "vep_json_output.json":
        pytest.skip("VEP schema is exempt from camelCase check.")

    core_schema = json.loads(Path(SCHEMA_DIR, schema_json).read_text(encoding="utf-8"))
    schema = StructType.fromJson(core_schema)
    # Use a regular expression to check if the identifier is in camelCase
    # CamelCase starts with a lowercase letter and has uppercase letters in between.

    for field in schema.fields:
        assert is_camelcase(field.name), (
            f"Column name '{field.name}' is not in camelCase."
        )


class TestValidateSchema:
    """Test validate_schema method using L2GPrediction (unnested) and TargetIndex (nested) as a testing dataset."""

    @pytest.fixture()
    def mock_dataset_instance(
        self: TestValidateSchema, request: FixtureRequest
    ) -> L2GPrediction | TargetIndex:
        """Meta fixture to return the value of any requested fixture."""
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize(
        "mock_dataset_instance",
        ["mock_l2g_predictions", "mock_target_index"],
        indirect=True,
    )
    def test_validate_schema_extra_field(
        self: TestValidateSchema,
        mock_dataset_instance: L2GPrediction | TargetIndex,
    ) -> None:
        """Test that validate_schema raises an error if the observed schema has an extra field."""
        with pytest.raises(SchemaValidationError, match="extraField"):
            mock_dataset_instance.df = mock_dataset_instance.df.withColumn(
                "extraField", f.lit("extra")
            )

    @pytest.mark.parametrize(
        "mock_dataset_instance",
        ["mock_l2g_predictions"],
        indirect=True,
    )
    def test_validate_schema_missing_field(
        self: TestValidateSchema,
        mock_dataset_instance: L2GPrediction | TargetIndex,
    ) -> None:
        """Test that validate_schema raises an error if the observed schema is missing a required field, geneId in this case."""
        with pytest.raises(SchemaValidationError, match="geneId"):
            mock_dataset_instance.df = mock_dataset_instance.df.drop("geneId")

    @pytest.mark.parametrize(
        "mock_dataset_instance",
        ["mock_l2g_predictions", "mock_target_index"],
        indirect=True,
    )
    def test_validate_schema_duplicated_field(
        self: TestValidateSchema,
        mock_dataset_instance: L2GPrediction | TargetIndex,
    ) -> None:
        """Test that validate_schema raises an error if the observed schema has a duplicated field, geneId in this case."""
        with pytest.raises(SchemaValidationError, match="geneId"):
            mock_dataset_instance.df = mock_dataset_instance.df.select(
                "*", f.lit("A").alias("geneId")
            )

    @pytest.mark.parametrize(
        "mock_dataset_instance",
        ["mock_l2g_predictions", "mock_target_index"],
        indirect=True,
    )
    def test_validate_schema_different_datatype(
        self: TestValidateSchema,
        mock_dataset_instance: L2GPrediction | TargetIndex,
    ) -> None:
        """Test that validate_schema raises an error if any field in the observed schema has a different type than expected."""
        with pytest.raises(SchemaValidationError, match="geneId"):
            mock_dataset_instance.df = mock_dataset_instance.df.withColumn(
                "geneId", f.lit(1)
            )
