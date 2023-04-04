"""Feature matrix of study locus pairs annotated with their functional genomics features."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial, reduce
from typing import TYPE_CHECKING, List, Optional, Type

from otg.common.schemas import parse_spark_schema
from otg.common.spark_helpers import _convert_from_long_to_wide
from otg.dataset.colocalisation import Colocalisation
from otg.dataset.dataset import Dataset
from otg.dataset.study_index import StudyIndex
from otg.dataset.study_locus import StudyLocus
from otg.dataset.v2g import V2G
from otg.method.l2g_utils.feature_factory import (
    ColocalisationFactory,
    StudyLocusFactory,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType

    from otg.common.session import Session


@dataclass
class L2GFeature(Dataset):
    """Property of a study locus pair."""

    # TODO: think about moving this to a trait id - so that we can extract the best study for that trait to train on

    _df: DataFrame
    _schema: StructType = parse_spark_schema("l2g_feature.json")


@dataclass
class L2GFeatureMatrix(Dataset):
    """Dataset with features for Locus to Gene prediction."""

    etl: Session
    _schema: StructType = parse_spark_schema("l2g_feature_matrix.json")
    _df: DataFrame = field(default_factory=partial(etl.spark.createDataFrame, [], schema=_schema))

    @staticmethod
    def fill_na(
        df: DataFrame, value: float = 0.0, subset: Optional[List[str]] = None
    ) -> DataFrame:
        """Fill missing values in a column with a given value."""
        return df.fillna(value, subset=subset)

    @classmethod
    def from_parquet(cls: type[L2GFeatureMatrix], etl: Session, path: str) -> Dataset:
        """Initialise L2GFeatureMatrix from parquet file.

        Args:
            etl (Session): ETL session
            path (str): Path to parquet file

        Returns:
            L2GFeatureMatrix: Locus to gene feature matrix
        """
        return super().from_parquet(session, path, cls._schema)

    @classmethod
    def generate_features(
        cls: Type[L2GFeatureMatrix],
        etl: Session,
        study_locus_path: str,
        study_index_path: str,
        variant_gene_path: str,
        colocalisation_path: str,
    ) -> L2GFeatureMatrix:
        """Generate features from the OTG datasets."""
        # Load datasets
        study_locus = StudyLocus.from_parquet(session, study_locus_path)
        studies = StudyIndex.from_parquet(session, study_index_path)
        distances = V2G.from_parquet(session, variant_gene_path)
        coloc = Colocalisation.from_parquet(session, colocalisation_path)

        # Extract features
        coloc_features = ColocalisationFactory._get_coloc_features_df(study_locus, studies, coloc)
        distance_features = StudyLocusFactory._get_tss_distance_features(distances)

        fm = reduce(
            lambda x, y: x.unionByName(y),
            [coloc_features._df, distance_features._df],
        )

        return cls(_df=_convert_from_long_to_wide(fm))

    def train_test_split(self: L2GFeatureMatrix, fraction: float) -> tuple[L2GFeatureMatrix, L2GFeatureMatrix]:
        """Split the dataset into training and test sets.

        Args:
            fraction (float): Fraction of the dataset to use for training

        Returns:
            tuple[L2GFeatureMatrix, L2GFeatureMatrix]: Training and test datasets
        """
        train, test = self._df.randomSplit([fraction, 1 - fraction], seed=42)
        return (
            L2GFeatureMatrix(_df=train).persist(),
            L2GFeatureMatrix(_df=test).persist(),
        )

    def fill_na(self: L2GFeatureMatrix) -> type[NotImplementedError]:
        """Fill NA values."""
        return NotImplementedError

    def get_distance_features(self: L2GFeatureMatrix, study_locus: StudyLocus, distances: V2G) -> L2GFeatureMatrix:
        """Get distance features."""
        distance_features = study_locus._get_tss_distance_features(distances)
        return L2GFeatureMatrix(_df=self._df.unionByName(distance_features, allowMissingColumns=True))

    def get_coloc_features(self: L2GFeatureMatrix, colocalisation: Colocalisation) -> type[NotImplementedError]:
        """Get colocalisation features."""
        return NotImplementedError


@classmethod
class L2G(Dataset):
    """Output L2G."""

    pass
