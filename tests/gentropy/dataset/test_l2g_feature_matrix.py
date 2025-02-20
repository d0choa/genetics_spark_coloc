# pylint: disable=too-few-public-methods
# isort: skip_file

"""Test L2G feature matrix methods."""

from __future__ import annotations

import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.target_index import TargetIndex
from gentropy.method.l2g.feature_factory import L2GFeatureInputLoader

from pyspark.sql import SparkSession


def test_select_features_inheritance(
    spark: SparkSession, mock_l2g_feature_matrix: L2GFeatureMatrix
) -> None:
    """Test L2GFeatureMatrix.select_features method inherits the instance attributes in the new instance."""
    new_instance = mock_l2g_feature_matrix.select_features(
        features_list=["distanceTssMean"]
    )
    assert new_instance.features_list == ["distanceTssMean"]
    # Because the feature matrix contains the gold standard flag information, the new fixed colums should be the same
    assert "goldStandardSet" in new_instance.fixed_cols


class TestFromFeaturesList:
    """Test L2GFeatureMatrix.from_features_list method.

    If the columns from the features list are there, it means that the business logic is working (the dataframe is not empty when converting from long to wide).
    """

    def test_study_locus(
        self: TestFromFeaturesList,
    ) -> None:
        """Test building feature matrix for a SL with the eQtlColocH4Maximum feature."""
        features_list = ["eQtlColocH4Maximum", "geneCount500kb"]
        loader = L2GFeatureInputLoader(
            colocalisation=self.sample_colocalisation,
            study_index=self.sample_study_index,
            study_locus=self.sample_study_locus,
            target_index=self.sample_target_index,
        )
        fm = L2GFeatureMatrix.from_features_list(
            self.sample_study_locus, features_list, loader
        )
        for feature in features_list:
            assert feature in fm._df.columns, (
                f"Feature {feature} not found in feature matrix."
            )

    def test_gold_standard(
        self: TestFromFeaturesList,
    ) -> None:
        """Test building feature matrix for a gold standard with the eQtlColocH4Maximum feature."""
        features_list = ["eQtlColocH4Maximum"]
        loader = L2GFeatureInputLoader(
            colocalisation=self.sample_colocalisation,
            study_index=self.sample_study_index,
            study_locus=self.sample_study_locus,
        )
        fm = L2GFeatureMatrix.from_features_list(
            self.sample_gold_standard, features_list, loader
        )
        for feature in features_list:
            assert feature in fm._df.columns, (
                f"Feature {feature} not found in feature matrix."
            )

    @pytest.fixture(autouse=True)
    def _setup(self: TestFromFeaturesList, spark: SparkSession) -> None:
        """Setup fixture."""
        self.sample_gold_standard = L2GGoldStandard(
            _df=spark.createDataFrame(
                [(1, "var1", "gwas1", "g1", "efo1", "positive", ["a_source"])],
                L2GGoldStandard.get_schema(),
            ),
            _schema=L2GGoldStandard.get_schema(),
        )
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    (
                        "1",
                        "var1",
                        "gwas1",
                        "X",
                        2,
                        [
                            {"variantId": "var1", "posteriorProbability": 0.8},
                            {"variantId": "var12", "posteriorProbability": 0.2},
                        ],
                    ),
                    (
                        "2",
                        "var2",
                        "eqtl1",
                        "X",
                        10,
                        [
                            {"variantId": "var2", "posteriorProbability": 1.0},
                        ],
                    ),
                ],
                schema=StructType(
                    [
                        StructField("studyLocusId", StringType(), True),
                        StructField("variantId", StringType(), True),
                        StructField("studyId", StringType(), True),
                        StructField("chromosome", StringType(), True),
                        StructField("position", IntegerType(), True),
                        StructField(
                            "locus",
                            ArrayType(
                                StructType(
                                    [
                                        StructField("variantId", StringType(), True),
                                        StructField(
                                            "posteriorProbability", DoubleType(), True
                                        ),
                                    ]
                                )
                            ),
                            True,
                        ),
                    ]
                ),
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_study_index = StudyIndex(
            _df=spark.createDataFrame(
                [("gwas1", "gwas", None, "p1"), ("eqtl1", "eqtl", "g1", "p2")],
                [
                    "studyId",
                    "studyType",
                    "geneId",
                    "projectId",
                ],
            ),
            _schema=StudyIndex.get_schema(),
        )
        self.sample_colocalisation = Colocalisation(
            _df=spark.createDataFrame(
                [("1", "2", "eqtl", "X", "COLOC", 1, 0.9)],
                [
                    "leftStudyLocusId",
                    "rightStudyLocusId",
                    "rightStudyType",
                    "chromosome",
                    "colocalisationMethod",
                    "numberColocalisingVariants",
                    "h4",
                ],
            ),
            _schema=Colocalisation.get_schema(),
        )
        self.sample_target_index = TargetIndex(
            _df=spark.createDataFrame(
                [
                    {
                        "id": "g1",
                        "genomicLocation": {
                            "chromosome": "X",
                        },
                        "tss": 200,
                        "biotype": "protein_coding",
                    },
                    {
                        "id": "g2",
                        "genomicLocation": {
                            "chromosome": "X",
                        },
                        "tss": 300,
                        "biotype": "protein_coding",
                    },
                ],
                TargetIndex.get_schema(),
            ),
            _schema=TargetIndex.get_schema(),
        )


def test_fill_na(spark: SparkSession) -> None:
    """Tests L2GFeatureMatrix.fill_na, particularly the imputation logic."""
    sample_fm = L2GFeatureMatrix(
        _df=spark.createDataFrame(
            [
                {
                    "studyLocusId": "1",
                    "geneId": "gene1",
                    "proteinGeneCount500kb": 3.0,
                    "geneCount500kb": 8.0,
                    "isProteinCoding": 1.0,
                    "anotherFeature": None,
                },
                {
                    "studyLocusId": "1",
                    "geneId": "gene2",
                    "proteinGeneCount500kb": 4.0,
                    "geneCount500kb": 10.0,
                    "isProteinCoding": 1.0,
                    "anotherFeature": None,
                },
                {
                    "studyLocusId": "1",
                    "geneId": "gene3",
                    "proteinGeneCount500kb": None,
                    "geneCount500kb": None,
                    "isProteinCoding": None,
                    "anotherFeature": None,
                },
            ],
            schema="studyLocusId STRING, geneId STRING, proteinGeneCount500kb DOUBLE, geneCount500kb DOUBLE, isProteinCoding DOUBLE, anotherFeature DOUBLE",
        ),
    )
    observed_df = sample_fm.fill_na()._df.filter(f.col("geneId") == "gene3")
    expected_df_missing_row = spark.createDataFrame(
        [
            {
                "studyLocusId": "1",
                "geneId": "gene3",
                "proteinGeneCount500kb": 3.5,
                "geneCount500kb": 9.0,
                "isProteinCoding": 0.0,
                "anotherFeature": 0.0,
            },
        ],
    ).select(
        "studyLocusId",
        "geneId",
        "proteinGeneCount500kb",
        "geneCount500kb",
        "isProteinCoding",
        "anotherFeature",
    )
    assert observed_df.collect() == expected_df_missing_row.collect()
