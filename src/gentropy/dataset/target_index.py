"""Target index dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from gentropy.common.schemas import parse_spark_schema
from gentropy.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType


@dataclass
class TargetIndex(Dataset):
    """Target index dataset.

    Gene-based annotation.
    """

    @classmethod
    def get_schema(cls: type[TargetIndex]) -> StructType:
        """Provides the schema for the TargetIndex dataset.

        Returns:
            StructType: Schema for the TargetIndex dataset
        """
        return parse_spark_schema("target_index.json")

    def filter_by_biotypes(self: TargetIndex, biotypes: list[str]) -> TargetIndex:
        """Filter by approved biotypes.

        Args:
            biotypes (list[str]): List of Ensembl biotypes to keep.

        Returns:
            TargetIndex: Target index dataset filtered by biotypes.
        """
        self.df = self._df.filter(f.col("biotype").isin(biotypes))
        return self

    def locations_lut(self: TargetIndex) -> DataFrame:
        """Gene location information.

        Returns:
            DataFrame: Gene LUT including genomic location information.
        """
        return self.df.select(
            "geneId",
            "chromosome",
            "start",
            "end",
            "strand",
            "tss",
        )

    def symbols_lut(self: TargetIndex) -> DataFrame:
        """Gene symbol lookup table.

        Pre-processess gene/target dataset to create lookup table of gene symbols, including
        obsoleted gene symbols.

        Returns:
            DataFrame: Gene LUT for symbol mapping containing `geneId` and `geneSymbol` columns.
        """
        return self.df.select(
            f.explode(
                f.array_union(f.array("approvedSymbol"), f.col("obsoleteSymbols"))
            ).alias("geneSymbol"),
            "*",
        )
