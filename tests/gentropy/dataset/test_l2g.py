"""Tests on L2G datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.types import FloatType

from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.l2g_prediction import L2GPrediction
from gentropy.dataset.study_locus_overlap import StudyLocusOverlap

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def test_feature_matrix(mock_l2g_feature_matrix: L2GFeatureMatrix) -> None:
    """Test L2G Feature Matrix creation with mock data."""
    assert isinstance(mock_l2g_feature_matrix, L2GFeatureMatrix)


def test_gold_standard(mock_l2g_gold_standard: L2GFeatureMatrix) -> None:
    """Test L2G gold standard creation with mock data."""
    assert isinstance(mock_l2g_gold_standard, L2GGoldStandard)


def test_process_gene_interactions(sample_otp_interactions: DataFrame) -> None:
    """Tests processing of gene interactions from OTP."""
    expected_cols = ["geneIdA", "geneIdB", "score"]
    observed_df = L2GGoldStandard.process_gene_interactions(sample_otp_interactions)
    assert observed_df.columns == expected_cols, (
        "Gene interactions has a different schema."
    )


def test_predictions(mock_l2g_predictions: L2GPrediction) -> None:
    """Test L2G predictions creation with mock data."""
    assert isinstance(mock_l2g_predictions, L2GPrediction)


def test_filter_unique_associations(spark: SparkSession) -> None:
    """Test filter_unique_associations."""
    mock_l2g_gs_df = spark.createDataFrame(
        [
            ("1", "variant1", "study1", "gene1", "positive"),
            (
                "2",
                "variant2",
                "study1",
                "gene1",
                "negative",
            ),  # in the same locus as sl1 and pointing to same gene, has to be dropped
            (
                "3",
                "variant3",
                "study1",
                "gene1",
                "positive",
            ),  # in diff locus as sl1 and pointing to same gene, has to be kept
            (
                "4",
                "variant4",
                "study1",
                "gene2",
                "positive",
            ),  # in same locus as sl1 and pointing to diff gene, has to be kept
        ],
        "studyLocusId STRING, variantId STRING, studyId STRING, geneId STRING, goldStandardSet STRING",
    )

    mock_sl_overlap_df = spark.createDataFrame(
        [
            ("1", "2", "eqtl", "CHROM1", "variant2", None),
            ("1", "4", "eqtl", "CHROM1", "variant4", None),
        ],
        StudyLocusOverlap.get_schema(),
    )

    expected_df = spark.createDataFrame(
        [
            ("1", "variant1", "study1", "gene1", "positive"),
            ("3", "variant3", "study1", "gene1", "positive"),
            ("4", "variant4", "study1", "gene2", "positive"),
        ],
        "studyLocusId STRING, variantId STRING, studyId STRING, geneId STRING, goldStandardSet STRING",
    )

    mock_l2g_gs = L2GGoldStandard(
        _df=mock_l2g_gs_df, _schema=L2GGoldStandard.get_schema()
    )
    mock_sl_overlap = StudyLocusOverlap(
        _df=mock_sl_overlap_df, _schema=StudyLocusOverlap.get_schema()
    )._convert_to_square_matrix()

    observed_df = mock_l2g_gs.filter_unique_associations(mock_sl_overlap).df

    assert observed_df.collect() == expected_df.collect()


def test_remove_false_negatives(spark: SparkSession) -> None:
    """Test `remove_false_negatives`."""
    mock_l2g_gs_df = spark.createDataFrame(
        [
            ("1", "variant1", "study1", "gene1", "positive"),
            (
                "2",
                "variant2",
                "study1",
                "gene2",
                "negative",
            ),  # gene2 is a partner of gene1, has to be dropped
            (
                "3",
                "variant3",
                "study1",
                "gene3",
                "negative",
            ),  # gene 3 is not a partner of gene1, has to be kept
            (
                "4",
                "variant4",
                "study1",
                "gene4",
                "positive",
            ),  # gene 4 is a partner of gene1, has to be kept because it's positive
        ],
        "studyLocusId STRING, variantId STRING, studyId STRING, geneId STRING, goldStandardSet STRING",
    )

    mock_interactions_df = spark.createDataFrame(
        [
            ("gene1", "gene2", 0.8),
            ("gene1", "gene3", 0.5),
            ("gene1", "gene4", 0.8),
        ],
        "geneIdA STRING, geneIdB STRING, score DOUBLE",
    )

    expected_df = spark.createDataFrame(
        [
            ("1", "variant1", "study1", "gene1", "positive"),
            ("3", "variant3", "study1", "gene3", "negative"),
            ("4", "variant4", "study1", "gene4", "positive"),
        ],
        "studyLocusId STRING, variantId STRING, studyId STRING, geneId STRING, goldStandardSet STRING",
    )

    mock_l2g_gs = L2GGoldStandard(
        _df=mock_l2g_gs_df, _schema=L2GGoldStandard.get_schema()
    )

    observed_df = mock_l2g_gs.remove_false_negatives(mock_interactions_df).df.orderBy(
        "studyLocusId"
    )

    assert observed_df.collect() == expected_df.collect()


def test_l2g_feature_constructor_with_schema_mismatch(
    spark: SparkSession,
) -> None:
    """Test if provided schema mismatch is converted to right type in the L2GFeatureMatrix constructor."""
    fm = L2GFeatureMatrix(
        _df=spark.createDataFrame(
            [
                ("1", "gene1", 100.0),
                ("2", "gene2", 1000.0),
            ],
            "studyLocusId STRING, geneId STRING, distanceTssMean DOUBLE",
        ),
        with_gold_standard=False,
    )
    assert fm._df.schema["distanceTssMean"].dataType == FloatType(), (
        "Feature `distanceTssMean` is not being casted to FloatType. Check L2GFeatureMatrix constructor."
    )


def test_calculate_feature_missingness_rate(
    spark: SparkSession, mock_l2g_feature_matrix: L2GFeatureMatrix
) -> None:
    """Test L2GFeatureMatrix.calculate_feature_missingness_rate."""
    expected_missingness = {"distanceTssMean": 0.0, "distanceSentinelTssMinimum": 1.0}
    observed_missingness = mock_l2g_feature_matrix.calculate_feature_missingness_rate()
    assert isinstance(observed_missingness, dict)
    assert mock_l2g_feature_matrix.features_list is not None and len(
        observed_missingness
    ) == len(mock_l2g_feature_matrix.features_list), (
        "Missing features in the missingness rate dictionary."
    )
    assert observed_missingness == expected_missingness, (
        "Missingness rate is incorrect."
    )
