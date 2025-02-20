# pylint: disable=too-few-public-methods
# isort: skip_file

"""Test locus-to-gene feature generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.target_index import TargetIndex
from gentropy.dataset.l2g_features.colocalisation import (
    EQtlColocClppMaximumFeature,
    EQtlColocClppMaximumNeighbourhoodFeature,
    EQtlColocH4MaximumFeature,
    EQtlColocH4MaximumNeighbourhoodFeature,
    PQtlColocClppMaximumFeature,
    PQtlColocClppMaximumNeighbourhoodFeature,
    PQtlColocH4MaximumFeature,
    PQtlColocH4MaximumNeighbourhoodFeature,
    SQtlColocClppMaximumFeature,
    SQtlColocClppMaximumNeighbourhoodFeature,
    SQtlColocH4MaximumFeature,
    SQtlColocH4MaximumNeighbourhoodFeature,
    common_colocalisation_feature_logic,
    common_neighbourhood_colocalisation_feature_logic,
    extend_missing_colocalisation_to_neighbourhood_genes,
)
from gentropy.dataset.l2g_features.distance import (
    DistanceFootprintMeanFeature,
    DistanceFootprintMeanNeighbourhoodFeature,
    DistanceSentinelFootprintFeature,
    DistanceSentinelFootprintNeighbourhoodFeature,
    DistanceSentinelTssFeature,
    DistanceSentinelTssNeighbourhoodFeature,
    DistanceTssMeanFeature,
    DistanceTssMeanNeighbourhoodFeature,
    common_distance_feature_logic,
    common_neighbourhood_distance_feature_logic,
)
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_features.vep import (
    VepMaximumFeature,
    VepMaximumNeighbourhoodFeature,
    VepMeanFeature,
    VepMeanNeighbourhoodFeature,
    common_neighbourhood_vep_feature_logic,
    common_vep_feature_logic,
)
from gentropy.dataset.l2g_features.other import (
    common_genecount_feature_logic,
    is_protein_coding_feature_logic,
    GeneCountFeature,
    ProteinGeneCountFeature,
    CredibleSetConfidenceFeature,
    ProteinCodingFeature,
)
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.variant_index import VariantIndex
from gentropy.method.l2g.feature_factory import L2GFeatureInputLoader

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@pytest.mark.parametrize(
    "feature_class",
    [
        EQtlColocH4MaximumFeature,
        PQtlColocH4MaximumFeature,
        SQtlColocH4MaximumFeature,
        EQtlColocClppMaximumFeature,
        PQtlColocClppMaximumFeature,
        SQtlColocClppMaximumFeature,
        EQtlColocClppMaximumNeighbourhoodFeature,
        PQtlColocClppMaximumNeighbourhoodFeature,
        SQtlColocClppMaximumNeighbourhoodFeature,
        EQtlColocH4MaximumNeighbourhoodFeature,
        PQtlColocH4MaximumNeighbourhoodFeature,
        SQtlColocH4MaximumNeighbourhoodFeature,
        DistanceTssMeanFeature,
        DistanceTssMeanNeighbourhoodFeature,
        DistanceFootprintMeanFeature,
        DistanceFootprintMeanNeighbourhoodFeature,
        DistanceSentinelTssFeature,
        DistanceSentinelTssNeighbourhoodFeature,
        DistanceSentinelFootprintFeature,
        DistanceSentinelFootprintNeighbourhoodFeature,
        VepMaximumFeature,
        VepMeanFeature,
        VepMaximumNeighbourhoodFeature,
        VepMeanNeighbourhoodFeature,
        GeneCountFeature,
        ProteinGeneCountFeature,
        CredibleSetConfidenceFeature,
        ProteinCodingFeature,
    ],
)
def test_feature_factory_return_type(
    feature_class: Any,
    mock_study_locus: StudyLocus,
    mock_colocalisation: Colocalisation,
    mock_study_index: StudyIndex,
    mock_variant_index: VariantIndex,
    mock_target_index: TargetIndex,
) -> None:
    """Test that every feature factory returns a L2GFeature dataset."""
    loader = L2GFeatureInputLoader(
        colocalisation=mock_colocalisation,
        study_index=mock_study_index,
        variant_index=mock_variant_index,
        study_locus=mock_study_locus,
        target_index=mock_target_index,
    )
    feature_dataset = feature_class.compute(
        study_loci_to_annotate=mock_study_locus,
        feature_dependency=loader.get_dependency_by_type(
            feature_class.feature_dependency_type
        ),
    )
    assert isinstance(feature_dataset, L2GFeature)


@pytest.fixture(scope="module")
def sample_target_index(spark: SparkSession) -> TargetIndex:
    """Create a sample target index for testing."""
    return TargetIndex(
        _df=spark.createDataFrame(
            [
                {
                    "id": "gene1",
                    "genomicLocation": {
                        "chromosome": "1",
                    },
                    "biotype": "protein_coding",
                },
                {
                    "id": "gene2",
                    "genomicLocation": {
                        "chromosome": "1",
                    },
                    "biotype": "lncRNA",
                },
                {
                    "id": "gene3",
                    "genomicLocation": {
                        "chromosome": "1",
                    },
                    "biotype": "protein_coding",
                },
            ],
            TargetIndex.get_schema(),
        ),
        _schema=TargetIndex.get_schema(),
    )


@pytest.fixture(scope="module")
def sample_variant_index(spark: SparkSession) -> VariantIndex:
    """Create a sample variant index for testing."""
    return VariantIndex(
        _df=spark.createDataFrame(
            [
                (
                    "var1",
                    "chrom",
                    1,
                    "A",
                    "T",
                    [
                        {
                            "targetId": "gene1",
                            "consequenceScore": 0.66,
                            "isEnsemblCanonical": True,
                        },
                        {
                            "targetId": "gene2",
                            "consequenceScore": 1.0,
                            "isEnsemblCanonical": True,
                        },
                        {
                            "targetId": "gene3",
                            "consequenceScore": 0.0,
                            "isEnsemblCanonical": True,
                        },
                    ],
                ),
            ],
            schema=StructType(
                [
                    StructField("variantId", StringType(), True),
                    StructField("chromosome", StringType(), True),
                    StructField("position", IntegerType(), True),
                    StructField("referenceAllele", StringType(), True),
                    StructField("alternateAllele", StringType(), True),
                    StructField(
                        "transcriptConsequences",
                        ArrayType(
                            StructType(
                                [
                                    StructField("targetId", StringType(), True),
                                    StructField(
                                        "isEnsemblCanonical", BooleanType(), True
                                    ),
                                    StructField("consequenceScore", FloatType(), True),
                                ]
                            )
                        ),
                        True,
                    ),
                ]
            ),
        ),
        _schema=VariantIndex.get_schema(),
    )


@pytest.fixture(scope="module")
def sample_variant_index_schema() -> StructType:
    """Partial schema of the variant index."""
    return StructType(
        [
            StructField("variantId", StringType(), True),
            StructField("chromosome", StringType(), True),
            StructField("position", IntegerType(), True),
            StructField("referenceAllele", StringType(), True),
            StructField("alternateAllele", StringType(), True),
            StructField(
                "transcriptConsequences",
                ArrayType(
                    StructType(
                        [
                            StructField("distanceFromFootprint", LongType(), True),
                            StructField("distanceFromTss", LongType(), True),
                            StructField("targetId", StringType(), True),
                            StructField("isEnsemblCanonical", BooleanType(), True),
                            StructField("biotype", StringType(), True),
                        ]
                    )
                ),
                True,
            ),
        ]
    )


class TestCommonColocalisationFeatureLogic:
    """Test the common logic of the colocalisation features."""

    def test__common_colocalisation_feature_logic(
        self: TestCommonColocalisationFeatureLogic,
        spark: SparkSession,
    ) -> None:
        """Test the common logic of the colocalisation features.

        The test data associates studyLocusId1 with gene1 based on the colocalisation with studyLocusId2 and studyLocusId3.
        The H4 value of number 2 is higher, therefore the feature value should be based on that.
        """
        feature_name = "eQtlColocH4Maximum"
        observed_df = common_colocalisation_feature_logic(
            self.sample_study_loci_to_annotate,
            self.colocalisation_method,
            self.colocalisation_metric,
            feature_name,
            self.qtl_type,
            colocalisation=self.sample_colocalisation,
            study_index=self.sample_studies,
            study_locus=self.sample_study_locus,
        )
        expected_df = spark.createDataFrame(
            [
                {
                    "studyLocusId": "1",
                    "geneId": "gene1",
                    "eQtlColocH4Maximum": 0.81,
                },
                {
                    "studyLocusId": "1",
                    "geneId": "gene2",
                    "eQtlColocH4Maximum": 0.9,
                },
            ],
        ).select("studyLocusId", "geneId", "eQtlColocH4Maximum")
        assert observed_df.collect() == expected_df.collect(), (
            "The feature values are not as expected."
        )

    def test_extend_missing_colocalisation_to_neighbourhood_genes(
        self: TestCommonColocalisationFeatureLogic,
        spark: SparkSession,
        sample_target_index: TargetIndex,
        sample_variant_index: VariantIndex,
    ) -> None:
        """Test the extend_missing_colocalisation_to_neighbourhood_genes function."""
        local_features = spark.createDataFrame(
            [
                {
                    "studyLocusId": "1",
                    "geneId": "gene1",
                    "eQtlColocH4Maximum": 0.81,
                },
                {
                    "studyLocusId": "1",
                    "geneId": "gene2",
                    "eQtlColocH4Maximum": 0.9,
                },
            ],
        )
        observed_df = extend_missing_colocalisation_to_neighbourhood_genes(
            feature_name="eQtlColocH4Maximum",
            local_features=local_features,
            variant_index=sample_variant_index,
            target_index=sample_target_index,
            study_locus=self.sample_study_locus,
        ).select("studyLocusId", "geneId", "eQtlColocH4Maximum")
        expected_df = spark.createDataFrame(
            [{"geneId": "gene3", "studyLocusId": "1", "eQtlColocH4Maximum": 0.0}]
        ).select("studyLocusId", "geneId", "eQtlColocH4Maximum")
        assert observed_df.collect() == expected_df.collect(), (
            "The feature values are not as expected."
        )

    def test_common_neighbourhood_colocalisation_feature_logic(
        self: TestCommonColocalisationFeatureLogic,
        spark: SparkSession,
        sample_target_index: TargetIndex,
        sample_variant_index: VariantIndex,
    ) -> None:
        """Test the common logic of the neighbourhood colocalisation features."""
        feature_name = "eQtlColocH4MaximumNeighbourhood"
        observed_df = common_neighbourhood_colocalisation_feature_logic(
            self.sample_study_loci_to_annotate,
            self.colocalisation_method,
            self.colocalisation_metric,
            feature_name,
            self.qtl_type,
            colocalisation=self.sample_colocalisation,
            study_index=self.sample_studies,
            study_locus=self.sample_study_locus,
            target_index=sample_target_index,
            variant_index=sample_variant_index,
        ).withColumn(feature_name, f.round(f.col(feature_name), 3))
        # expected max is 0.81
        expected_df = spark.createDataFrame(
            [
                {
                    "studyLocusId": "1",
                    "geneId": "gene1",
                    "eQtlColocH4MaximumNeighbourhood": 1.0,  # 0.81 / 0.81
                },
                {
                    "studyLocusId": "1",
                    "geneId": "gene3",
                    "eQtlColocH4MaximumNeighbourhood": 0.0,  # 0.0 (no coloc with gene3) /0.81
                },
            ],
        ).select("geneId", "studyLocusId", "eQtlColocH4MaximumNeighbourhood")
        assert observed_df.collect() == expected_df.collect(), (
            "The expected and observed dataframes do not match."
        )

    @pytest.fixture(autouse=True)
    def _setup(self: TestCommonColocalisationFeatureLogic, spark: SparkSession) -> None:
        """Set up the test variables."""
        self.colocalisation_method = "Coloc"
        self.colocalisation_metric = "h4"
        self.qtl_type = "eqtl"

        self.sample_study_loci_to_annotate = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "var1",
                        "studyId": "study1",  # this is a GWAS
                        "chromosome": "1",
                    },
                ]
            ),
            _schema=StudyLocus.get_schema(),
        )

        self.sample_colocalisation = Colocalisation(
            _df=spark.createDataFrame(
                [
                    {
                        "leftStudyLocusId": "1",
                        "rightStudyLocusId": "2",
                        "chromosome": "1",
                        "colocalisationMethod": "COLOC",
                        "numberColocalisingVariants": 1,
                        "h4": 0.81,
                        "rightStudyType": "eqtl",
                    },
                    {
                        "leftStudyLocusId": "1",
                        "rightStudyLocusId": "3",  # qtl linked to the same gene as studyLocusId 2 with a lower score
                        "chromosome": "1",
                        "colocalisationMethod": "COLOC",
                        "numberColocalisingVariants": 1,
                        "h4": 0.50,
                        "rightStudyType": "eqtl",
                    },
                    {
                        "leftStudyLocusId": "1",
                        "rightStudyLocusId": "4",  # qtl linked to a diff gene and with the highest score
                        "chromosome": "1",
                        "colocalisationMethod": "COLOC",
                        "numberColocalisingVariants": 1,
                        "h4": 0.90,
                        "rightStudyType": "eqtl",
                    },
                ],
                schema=Colocalisation.get_schema(),
            ),
            _schema=Colocalisation.get_schema(),
        )
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "var1",
                        "studyId": "study1",  # this is a GWAS
                        "chromosome": "1",
                    },
                    {
                        "studyLocusId": "2",
                        "variantId": "var1",
                        "studyId": "study2",  # this is a QTL (same gene)
                        "chromosome": "1",
                    },
                    {
                        "studyLocusId": "3",
                        "variantId": "var1",
                        "studyId": "study3",  # this is another QTL (same gene)
                        "chromosome": "1",
                    },
                    {
                        "studyLocusId": "4",
                        "variantId": "var1",
                        "studyId": "study4",  # this is another QTL (diff gene)
                        "chromosome": "1",
                    },
                ]
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_studies = StudyIndex(
            _df=spark.createDataFrame(
                [
                    {
                        "studyId": "study1",
                        "studyType": "gwas",
                        "geneId": None,
                        "traitFromSource": "trait1",
                        "projectId": "project1",
                    },
                    {
                        "studyId": "study2",
                        "studyType": "eqtl",
                        "geneId": "gene1",
                        "traitFromSource": "trait2",
                        "projectId": "project2",
                    },
                    {
                        "studyId": "study3",
                        "studyType": "eqtl",
                        "geneId": "gene1",
                        "traitFromSource": "trait3",
                        "projectId": "project3",
                    },
                    {
                        "studyId": "study4",
                        "studyType": "eqtl",
                        "geneId": "gene2",
                        "traitFromSource": "trait4",
                        "projectId": "project4",
                    },
                ]
            ),
            _schema=StudyIndex.get_schema(),
        )


class TestCommonDistanceFeatureLogic:
    """Test the CommonDistanceFeatureLogic methods."""

    @pytest.mark.parametrize(
        ("feature_name", "expected_data"),
        [
            (
                "distanceSentinelTss",
                [
                    {
                        "studyLocusId": "1",
                        "geneId": "gene1",
                        "distanceSentinelTss": 0.0,
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene2",
                        "distanceSentinelTss": 0.92,
                    },
                ],
            ),
            (
                "distanceTssMean",
                [
                    {"studyLocusId": "1", "geneId": "gene1", "distanceTssMean": 0.52},
                    {"studyLocusId": "1", "geneId": "gene2", "distanceTssMean": 0.63},
                ],
            ),
        ],
    )
    def test_common_distance_feature_logic(
        self: TestCommonDistanceFeatureLogic,
        spark: SparkSession,
        feature_name: str,
        expected_data: dict[str, Any],
    ) -> None:
        """Test the logic of the function that extracts features from distance.

        2 tests:
        - distanceSentinelTss: distance of the sentinel is 10, the max distance is 10. In log scale, the score is 0.
        - distanceTssMean: avg distance of any variant in the credible set, weighted by its posterior.
        """
        observed_df = (
            common_distance_feature_logic(
                self.sample_study_locus,
                variant_index=self.sample_variant_index,
                feature_name=feature_name,
                distance_type=self.distance_type,
                genomic_window=10,
            )
            .withColumn(feature_name, f.round(f.col(feature_name), 2))
            .orderBy(feature_name)
        )
        expected_df = (
            spark.createDataFrame(expected_data)
            .select("studyLocusId", "geneId", feature_name)
            .orderBy(feature_name)
        )
        assert observed_df.collect() == expected_df.collect(), (
            f"Expected and observed dataframes are not equal for feature {feature_name}."
        )

    def test_common_neighbourhood_distance_feature_logic(
        self: TestCommonDistanceFeatureLogic,
        spark: SparkSession,
    ) -> None:
        """Test the logic of the function that extracts the distance between the sentinel of a credible set and the nearby genes."""
        feature_name = "distanceSentinelTssNeighbourhood"
        observed_df = (
            common_neighbourhood_distance_feature_logic(
                self.sample_study_locus,
                variant_index=self.sample_variant_index,
                target_index=self.sample_target_index,
                feature_name=feature_name,
                distance_type=self.distance_type,
                genomic_window=10,
            )
            .withColumn(feature_name, f.round(f.col(feature_name), 2))
            .orderBy(f.col(feature_name).asc())
        )
        expected_df = spark.createDataFrame(  # regional max is 0.91 from gene2
            (
                ["gene1", "1", 0.0],  # (10-10)/0.91
                ["gene2", "1", 1.0],
            ),  # 0.91/0.91
            ["geneId", "studyLocusId", feature_name],
        ).orderBy(feature_name)
        assert observed_df.collect() == expected_df.collect(), (
            "Output doesn't meet the expectation."
        )

    @pytest.fixture(autouse=True)
    def _setup(
        self: TestCommonDistanceFeatureLogic,
        spark: SparkSession,
        sample_variant_index_schema: StructType,
    ) -> None:
        """Set up testing fixtures."""
        self.distance_type = "distanceFromTss"
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "lead1",
                        "studyId": "study1",
                        "locus": [
                            {
                                "variantId": "lead1",
                                "posteriorProbability": 0.5,
                            },
                            {
                                "variantId": "tag1",
                                "posteriorProbability": 0.5,
                            },
                        ],
                        "chromosome": "1",
                    },
                ],
                StudyLocus.get_schema(),
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_variant_index = VariantIndex(
            _df=spark.createDataFrame(
                [
                    (
                        "lead1",
                        "chrom",
                        1,
                        "A",
                        "T",
                        [
                            {
                                "distanceFromTss": 10,
                                "distanceFromFootprint": 0,
                                "targetId": "gene1",
                                "isEnsemblCanonical": True,
                                "biotype": "protein_coding",
                            },
                            {
                                "distanceFromTss": 2,
                                "distanceFromFootprint": 0,
                                "targetId": "gene2",
                                "isEnsemblCanonical": True,
                                "biotype": "protein_coding",
                            },
                        ],
                    ),
                    (
                        "tag1",
                        "chrom",
                        1,
                        "A",
                        "T",
                        [
                            {
                                "distanceFromTss": 5,
                                "distanceFromFootprint": 0,
                                "targetId": "gene1",
                                "isEnsemblCanonical": True,
                                "biotype": "protein_coding",
                            },
                        ],
                    ),
                ],
                sample_variant_index_schema,
            ),
            _schema=VariantIndex.get_schema(),
        )
        self.sample_target_index = TargetIndex(
            _df=spark.createDataFrame(
                [
                    {
                        "id": "gene1",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 950000,
                        "biotype": "protein_coding",
                    },
                    {
                        "id": "gene2",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 1050000,
                        "biotype": "protein_coding",
                    },
                    {
                        "id": "gene3",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 1010000,
                        "biotype": "non_coding",
                    },
                ],
                TargetIndex.get_schema(),
            ),
            _schema=TargetIndex.get_schema(),
        )


class TestCommonVepFeatureLogic:
    """Test the common_vep_feature_logic methods."""

    @pytest.mark.parametrize(
        ("feature_name", "expected_data"),
        [
            (
                "vepMean",
                [
                    {
                        "studyLocusId": "1",
                        "geneId": "gene3",
                        "vepMean": "0.00",
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene1",
                        "vepMean": "0.33",
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene2",
                        "vepMean": "0.50",
                    },
                ],
            ),
            (
                "vepMaximum",
                [
                    {
                        "studyLocusId": "1",
                        "geneId": "gene3",
                        "vepMaximum": "0.00",
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene1",
                        "vepMaximum": "0.66",
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene2",
                        "vepMaximum": "1.00",
                    },
                ],
            ),
        ],
    )
    def test_common_vep_feature_logic(
        self: TestCommonVepFeatureLogic,
        spark: SparkSession,
        feature_name: str,
        expected_data: dict[str, Any],
        sample_variant_index: VariantIndex,
    ) -> None:
        """Test the logic of the function that extracts features from VEP's functional consequences."""
        observed_df = (
            common_vep_feature_logic(
                self.sample_study_locus,
                variant_index=sample_variant_index,
                feature_name=feature_name,
            )
            .orderBy(feature_name)
            .withColumn(
                feature_name, f.format_number(f.round(f.col(feature_name), 2), 2)
            )
        )
        expected_df = (
            spark.createDataFrame(expected_data)
            .orderBy(feature_name)
            .select("studyLocusId", "geneId", feature_name)
        )
        assert observed_df.collect() == expected_df.collect(), (
            f"Expected and observed dataframes are not equal for feature {feature_name}."
        )

        def test_common_neighbourhood_vep_feature_logic(
            self: TestCommonVepFeatureLogic,
            spark: SparkSession,
            sample_target_index: TargetIndex,
            sample_variant_index: VariantIndex,
        ) -> None:
            """Test the logic of the function that extracts the maximum severity score for a gene given the maximum of the maximum scores for all protein coding genes in the vicinity."""
            feature_name = "vepMaximumNeighbourhood"
            observed_df = (
                common_neighbourhood_vep_feature_logic(
                    self.sample_study_locus,
                    variant_index=sample_variant_index,
                    target_index=sample_target_index,
                    feature_name=feature_name,
                )
                .withColumn(feature_name, f.round(f.col(feature_name), 2))
                .orderBy(f.col(feature_name).asc())
            )
            expected_df = (
                spark.createDataFrame(
                    # regional max is 0.66
                    (
                        ["1", "gene1", 1.0],  # 0.66/0.66
                        ["1", "gene3", 0.0],  # 0/0.66
                    ),
                    ["studyLocusId", "geneId", feature_name],
                )
                .orderBy(feature_name)
                .select("studyLocusId", "geneId", feature_name)
            )
            assert observed_df.collect() == expected_df.collect(), (
                "Output doesn't meet the expectation."
            )

    @pytest.fixture(autouse=True)
    def _setup(self: TestCommonVepFeatureLogic, spark: SparkSession) -> None:
        """Set up testing fixtures."""
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "var1",
                        "studyId": "study1",
                        "locus": [
                            {
                                "variantId": "var1",
                                "posteriorProbability": 0.5,
                            },
                        ],
                        "chromosome": "1",
                    },
                ],
                StudyLocus.get_schema(),
            ),
            _schema=StudyLocus.get_schema(),
        )


class TestCommonGeneCountFeatureLogic:
    """Test the CommonGeneCountFeatureLogic methods."""

    @pytest.mark.parametrize(
        ("feature_name", "expected_data", "protein_coding_only"),
        [
            (
                "geneCount",
                [
                    {"studyLocusId": "1", "geneId": "gene1", "geneCount": 3},
                    {"studyLocusId": "1", "geneId": "gene2", "geneCount": 3},
                    {"studyLocusId": "1", "geneId": "gene3", "geneCount": 3},
                ],
                False,  # Test case for all genes
            ),
            (
                "geneCountProteinCoding",
                [
                    {
                        "studyLocusId": "1",
                        "geneId": "gene1",
                        "geneCountProteinCoding": 2,
                    },
                    {
                        "studyLocusId": "1",
                        "geneId": "gene2",
                        "geneCountProteinCoding": 2,
                    },
                ],
                True,  # Test case for protein-coding genes only
            ),
        ],
    )
    def test_common_genecount_feature_logic(
        self: TestCommonGeneCountFeatureLogic,
        spark: SparkSession,
        feature_name: str,
        expected_data: list[dict[str, Any]],
        protein_coding_only: bool,
    ) -> None:
        """Test the common logic of the gene count features."""
        observed_df = common_genecount_feature_logic(
            study_loci_to_annotate=self.sample_study_locus,
            target_index=self.sample_target_index,
            feature_name=feature_name,
            genomic_window=500000,
            protein_coding_only=protein_coding_only,
        ).orderBy("studyLocusId", "geneId")
        expected_df = (
            spark.createDataFrame(expected_data)
            .select("studyLocusId", "geneId", feature_name)
            .orderBy("studyLocusId", "geneId")
        )

        assert observed_df.collect() == expected_df.collect(), (
            f"Expected and observed dataframes do not match for feature {feature_name}."
        )

    @pytest.fixture(autouse=True)
    def _setup(self: TestCommonGeneCountFeatureLogic, spark: SparkSession) -> None:
        """Set up testing fixtures."""
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "var1",
                        "studyId": "study1",
                        "chromosome": "1",
                        "position": 1000000,
                    },
                ],
                StudyLocus.get_schema(),
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_target_index = TargetIndex(
            _df=spark.createDataFrame(
                [
                    {
                        "id": "gene1",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 950000,
                        "biotype": "protein_coding",
                    },
                    {
                        "id": "gene2",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 1050000,
                        "biotype": "protein_coding",
                    },
                    {
                        "id": "gene3",
                        "genomicLocation": {
                            "chromosome": "1",
                        },
                        "tss": 1010000,
                        "biotype": "non_coding",
                    },
                ],
                TargetIndex.get_schema(),
            ),
            _schema=TargetIndex.get_schema(),
        )


class TestCommonProteinCodingFeatureLogic:
    """Test the CommonGeneCountFeatureLogic methods."""

    @pytest.mark.parametrize(
        ("expected_data"),
        [
            (
                [
                    {"studyLocusId": "1", "geneId": "gene1", "isProteinCoding": 1.0},
                    {"studyLocusId": "1", "geneId": "gene2", "isProteinCoding": 0.0},
                ]
            ),
        ],
    )
    def test_is_protein_coding_feature_logic(
        self: TestCommonProteinCodingFeatureLogic,
        spark: SparkSession,
        expected_data: list[dict[str, Any]],
    ) -> None:
        """Test the logic of the is_protein_coding_feature_logic function."""
        observed_df = (
            is_protein_coding_feature_logic(
                study_loci_to_annotate=self.sample_study_locus,
                variant_index=self.sample_variant_index,
                feature_name="isProteinCoding",
            )
            .select("studyLocusId", "geneId", "isProteinCoding")
            .orderBy("studyLocusId", "geneId")
        )

        expected_df = (
            spark.createDataFrame(expected_data)
            .select("studyLocusId", "geneId", "isProteinCoding")
            .orderBy("studyLocusId", "geneId")
        )
        assert observed_df.collect() == expected_df.collect(), (
            "Expected and observed DataFrames do not match."
        )

    @pytest.fixture(autouse=True)
    def _setup(
        self: TestCommonProteinCodingFeatureLogic,
        spark: SparkSession,
        sample_variant_index_schema: StructType,
    ) -> None:
        """Set up sample data for the test."""
        # Sample study locus data
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "var1",
                        "studyId": "study1",
                        "chromosome": "1",
                        "position": 1000000,
                        "locus": [
                            {
                                "variantId": "var1",
                            },
                        ],
                    },
                ],
                StudyLocus.get_schema(),
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_variant_index = VariantIndex(
            _df=spark.createDataFrame(
                [
                    (
                        "var1",
                        "chrom",
                        1,
                        "A",
                        "T",
                        [
                            {
                                "distanceFromFootprint": 0,
                                "distanceFromTss": 10,
                                "targetId": "gene1",
                                "biotype": "protein_coding",
                                "isEnsemblCanonical": True,
                            },
                            {
                                "distanceFromFootprint": 0,
                                "distanceFromTss": 20,
                                "targetId": "gene2",
                                "biotype": "non_coding",
                                "isEnsemblCanonical": True,
                            },
                        ],
                    ),
                ],
                sample_variant_index_schema,
            ),
            _schema=VariantIndex.get_schema(),
        )


class TestCredibleSetConfidenceFeatureLogic:
    """Test the CredibleSetConfidenceFeature method."""

    def test_compute(
        self: TestCredibleSetConfidenceFeatureLogic,
        spark: SparkSession,
    ) -> None:
        """Test the logic of the function that scores a credible set's confidence."""
        sample_study_loci_to_annotate = self.sample_study_locus
        observed_df = CredibleSetConfidenceFeature.compute(
            study_loci_to_annotate=sample_study_loci_to_annotate,
            feature_dependency={
                "study_locus": self.sample_study_locus,
                "variant_index": self.sample_variant_index,
            },
        )
        assert observed_df.df.first()["featureValue"] == 0.25

    @pytest.fixture(autouse=True)
    def _setup(
        self: TestCredibleSetConfidenceFeatureLogic,
        spark: SparkSession,
        sample_variant_index_schema: StructType,
    ) -> None:
        """Set up testing fixtures."""
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    {
                        "studyLocusId": "1",
                        "variantId": "lead1",
                        "studyId": "study1",
                        "confidence": "PICS fine-mapped credible set based on reported top hit",
                        "chromosome": "1",
                        "locus": [
                            {
                                "variantId": "lead1",
                            },
                        ],
                    },
                ],
                StudyLocus.get_schema(),
            ),
            _schema=StudyLocus.get_schema(),
        )
        self.sample_variant_index = VariantIndex(
            _df=spark.createDataFrame(
                [
                    (
                        "lead1",
                        "chrom",
                        1,
                        "A",
                        "T",
                        [
                            {
                                "distanceFromTss": 10,
                                "distanceFromFootprint": 0,
                                "targetId": "gene1",
                                "isEnsemblCanonical": True,
                                "biotype": "protein_coding",
                            },
                        ],
                    )
                ],
                sample_variant_index_schema,
            ),
            _schema=VariantIndex.get_schema(),
        )
