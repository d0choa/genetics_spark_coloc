"""LDIndex dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otg.common.schemas import parse_spark_schema
from otg.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class LDIndex(Dataset):
    """Dataset containing linkage desequilibrium information between variants."""

    @classmethod
    def get_schema(cls: type[LDIndex]) -> StructType:
        """Provides the schema for the LDIndex dataset.

        Returns:
            StructType: Schema for the LDIndex dataset
        """
        return parse_spark_schema("ld_index.json")
