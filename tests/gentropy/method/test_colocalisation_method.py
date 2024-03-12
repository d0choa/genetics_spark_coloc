"""Test colocalisation methods."""

from __future__ import annotations

from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.study_locus_overlap import StudyLocusOverlap
from gentropy.method.colocalisation import Coloc, ECaviar
from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def test_coloc(mock_study_locus_overlap: StudyLocusOverlap) -> None:
    """Test coloc."""
    assert isinstance(Coloc.colocalise(mock_study_locus_overlap), Colocalisation)


def test_coloc_colocalise(
    spark: SparkSession,
    threshold: float = 1e-5,
) -> None:
    """Compare COLOC results with R implementation, using provided sample dataset from R package (StudyLocusOverlap)."""
    test_overlap_df = spark.read.parquet(
        "tests/gentropy/data_samples/coloc_test_data.snappy.parquet", header=True
    )
    test_overlap = StudyLocusOverlap(test_overlap_df, StudyLocusOverlap.get_schema())
    test_result = Coloc.colocalise(test_overlap)

    expected = spark.createDataFrame(
        [
            {
                "h0": 1.3769995397857477e-18,
                "h1": 2.937336451601565e-10,
                "h2": 8.593226431647826e-12,
                "h3": 8.338916748775843e-4,
                "h4": 0.9991661080227981,
            }
        ]
    )
    difference = test_result.df.select("h0", "h1", "h2", "h3", "h4").subtract(expected)
    for col in difference.columns:
        assert difference.filter(f.abs(f.col(col)) > threshold).count() == 0


def test_single_snp_coloc(
    spark: SparkSession,
    threshold: float = 1e-5,
) -> None:
    """Test edge case of coloc where only one causal SNP is present in the StudyLocusOverlap."""
    test_overlap_df = spark.createDataFrame(
        [
            {
                "leftStudyLocusId": 1,
                "rightStudyLocusId": 2,
                "chromosome": "1",
                "tagVariantId": "snp",
                "left_logBF": 10.3,
                "right_logBF": 10.5,
            }
        ]
    )
    test_overlap = StudyLocusOverlap(
        test_overlap_df.select(
            "leftStudyLocusId",
            "rightStudyLocusId",
            "chromosome",
            "tagVariantId",
            f.struct(f.col("left_logBF"), f.col("right_logBF")).alias("statistics"),
        ),
        StudyLocusOverlap.get_schema(),
    )
    test_result = Coloc.colocalise(test_overlap)

    expected = spark.createDataFrame(
        [
            {
                "h0": 9.254841951638903e-5,
                "h1": 2.7517068829182966e-4,
                "h2": 3.3609423764447284e-4,
                "h3": 9.254841952564387e-13,
                "h4": 0.9992961866536217,
            }
        ]
    )
    difference = test_result.df.select("h0", "h1", "h2", "h3", "h4").subtract(expected)
    for col in difference.columns:
        assert difference.filter(f.abs(f.col(col)) > threshold).count() == 0


def test_single_snp_coloc_one_negative(
    spark: SparkSession,
    threshold: float = 1e-5,
) -> None:
    """Test edge case of coloc where only one causal SNP is present (On one side!) in the StudyLocusOverlap."""
    test_overlap_df = spark.createDataFrame(
        [
            {
                "leftStudyLocusId": 1,
                "rightStudyLocusId": 2,
                "chromosome": "1",
                "tagVariantId": "snp",
                "left_logBF": 18.3,
                "right_logBF": 0.01,
            }
        ]
    )
    test_overlap = StudyLocusOverlap(
        test_overlap_df.select(
            "leftStudyLocusId",
            "rightStudyLocusId",
            "chromosome",
            "tagVariantId",
            f.struct(f.col("left_logBF"), f.col("right_logBF")).alias("statistics"),
        ),
        StudyLocusOverlap.get_schema(),
    )
    test_result = Coloc.colocalise(test_overlap)
    test_result.df.show(1, False)
    expected = spark.createDataFrame(
        [
            {
                "h0": 1.0246538505087709e-4,
                "h1": 0.9081680002273896,
                "h2": 1.0349517929098209e-8,
                "h3": 1.0246538506112363e-12,
                "h4": 0.09172952403701702,
            }
        ]
    )
    difference = test_result.df.select("h0", "h1", "h2", "h3", "h4").subtract(expected)
    for col in difference.columns:
        assert difference.filter(f.abs(f.col(col)) > threshold).count() == 0


def test_single_snp_coloc_both_negative(
    spark: SparkSession,
    threshold: float = 1e-5,
) -> None:
    """Test edge case of coloc where only one non-causal SNP overlaps in the StudyLocusOverlap."""
    test_overlap_df = spark.createDataFrame(
        [
            {
                "leftStudyLocusId": 1,
                "rightStudyLocusId": 2,
                "chromosome": "1",
                "tagVariantId": "snp",
                "left_logBF": 0.03,
                "right_logBF": 0.01,
            }
        ]
    )
    test_overlap = StudyLocusOverlap(
        test_overlap_df.select(
            "leftStudyLocusId",
            "rightStudyLocusId",
            "chromosome",
            "tagVariantId",
            f.struct(f.col("left_logBF"), f.col("right_logBF")).alias("statistics"),
        ),
        StudyLocusOverlap.get_schema(),
    )
    test_result = Coloc.colocalise(test_overlap)
    expected = spark.createDataFrame(
        [
            {
                "h0": 0.9997855774090624,
                "h1": 1.0302335812225042e-4,
                "h2": 1.0098335895103664e-4,
                "h3": 9.9978557750904e-9,
                "h4": 1.0405876008495098e-5,
            }
        ]
    )
    difference = test_result.df.select("h0", "h1", "h2", "h3", "h4").subtract(expected)
    for col in difference.columns:
        assert difference.filter(f.abs(f.col(col)) > threshold).count() == 0


def test_ecaviar(mock_study_locus_overlap: StudyLocusOverlap) -> None:
    """Test eCAVIAR."""
    assert isinstance(ECaviar.colocalise(mock_study_locus_overlap), Colocalisation)
