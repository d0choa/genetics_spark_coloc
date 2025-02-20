"""Pairwise LD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
from pyspark.sql import functions as f
from pyspark.sql import types as t

from gentropy.common.schemas import parse_spark_schema
from gentropy.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class PairwiseLD(Dataset):
    """Pairwise variant correlation dataset.

    This class captures logic applied on pairwise linkage data + by validation ensuring data quality.
    """

    dimension: tuple[int, int] = field(init=False)

    def __post_init__(self: PairwiseLD) -> None:
        """Validating the dataset upon creation.

        - Besides the schema, a pairwise LD table is expected have rows being a square number.

        Raises:
            AssertionError: When the number of rows in the provided dataframe to construct the LD matrix is not even after applying square root.
        """
        row_count = self.df.count()

        assert (
            int(sqrt(row_count)) == sqrt(row_count)
        ), f"The number of rows in a pairwise LD table has to be square. Got: {row_count}"

        self.dimension = (int(sqrt(row_count)), int(sqrt(row_count)))
        super().__post_init__()

    @classmethod
    def get_schema(cls: type[PairwiseLD]) -> StructType:
        """Provide the schema for the StudyIndex dataset.

        Returns:
            StructType: The schema of the StudyIndex dataset.
        """
        return parse_spark_schema("pairwise_ld.json")

    def overlap_with_locus(self: PairwiseLD, locus_variants: list[str]) -> PairwiseLD:
        """Subset pairwise LD table with locus.

        Args:
            locus_variants (list[str]): List of variants found in the locus.

        Returns:
            PairwiseLD: _description_
        """
        return PairwiseLD(
            _df=(
                self.df.filter(
                    f.col("variantIdI").isin(locus_variants)
                    & f.col("variantIdJ").isin(locus_variants)
                )
            ),
            _schema=PairwiseLD.get_schema(),
        )

    def r_to_numpy_matrix(self) -> np.ndarray:
        """Convert pairwise LD to a numpy square matrix.

        Returns:
            np.ndarray: 2D square matrix with r values.
        """
        return np.array(
            self.df.select(
                f.split("variantIdI", "_")[1].cast(t.IntegerType()).alias("position_i"),
                f.split("variantIdJ", "_")[1].cast(t.IntegerType()).alias("position_j"),
                "r",
            )
            .orderBy(f.col("position_i").asc(), f.col("position_j").asc())
            .select("r")
            .collect()
        ).reshape(self.dimension)

    def get_variant_list(self) -> list[str]:
        """Return a list of unique variants from the dataset.

        Returns:
            list[str]: list of variant identifiers sorted by position.
        """
        return [
            row["variantId"]
            for row in (
                self.df.select(
                    f.col("variantIdI").alias("variantId"),
                    f.split(f.col("variantIdI"), "_")[1]
                    .cast(t.IntegerType())
                    .alias("position"),
                )
                .orderBy(f.col("position").asc())
                .collect()
            )
        ]
