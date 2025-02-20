"""Test colocalisation methods."""

from __future__ import annotations

from typing import Any

import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.study_locus_overlap import StudyLocusOverlap
from gentropy.method.colocalisation import Coloc, ECaviar


def test_coloc(mock_study_locus_overlap: StudyLocusOverlap) -> None:
    """Test coloc."""
    assert isinstance(Coloc.colocalise(mock_study_locus_overlap), Colocalisation)
    assert isinstance(
        Coloc.colocalise(
            mock_study_locus_overlap, priorc1=1e-4, priorc2=1e-4, priorc12=1e-5
        ),
        Colocalisation,
    )


@pytest.mark.parametrize(
    "observed_data, expected_data",
    [
        # associations with a single overlapping SNP
        (
            # observed overlap
            [
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp",
                    "statistics": {
                        "left_logBF": 10.3,
                        "right_logBF": 10.5,
                        "left_beta": 0.1,
                        "right_beta": 0.2,
                        "left_posteriorProbability": 0.91,
                        "right_posteriorProbability": 0.92,
                    },
                },
            ],
            # expected coloc
            [
                {
                    "h0": 9.254841951638903e-5,
                    "h1": 2.7517068829182966e-4,
                    "h2": 3.3609423764447284e-4,
                    "h3": 9.254841952564387e-13,
                    "h4": 0.9992961866536217,
                },
            ],
        ),
        # Case with mismatched posterior probabilities:
        (
            # observed overlap
            [
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp1",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": 10.5,
                        "left_beta": 0.001,
                        "right_beta": 0.2,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.92,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp2",
                    "statistics": {
                        "left_logBF": 10.3,
                        "right_logBF": 3.8,
                        "left_beta": 0.3,
                        "right_beta": 0.005,
                        "left_posteriorProbability": 0.91,
                        "right_posteriorProbability": 0.01,
                    },
                },
            ],
            # expected coloc
            [],
        ),
        # Case of an overlap with significant PP overlap:
        (
            # observed overlap
            [
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp1",
                    "statistics": {
                        "left_logBF": 10.2,
                        "right_logBF": 10.5,
                        "left_beta": 0.5,
                        "right_beta": 0.2,
                        "left_posteriorProbability": 0.91,
                        "right_posteriorProbability": 0.92,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp2",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": 3.8,
                        "left_beta": 0.003,
                        "right_beta": 0.005,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.01,
                    },
                },
            ],
            # expected coloc
            [
                {
                    "h0": 1.02277006860577e-4,
                    "h1": 2.7519169183135977e-4,
                    "h2": 3.718812819512325e-4,
                    "h3": 1.3533048074295033e-6,
                    "h4": 0.9992492967145488,
                },
            ],
        ),
        # Case where the overlap source is ["left", "both", "both"]:
        (
            # observed overlap
            [
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp1",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": None,
                        "left_beta": 0.003,
                        "right_beta": None,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.01,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp2",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": 3.8,
                        "left_beta": 0.003,
                        "right_beta": 0.005,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.01,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp3",
                    "statistics": {
                        "left_logBF": 10.2,
                        "right_logBF": 10.5,
                        "left_beta": 0.5,
                        "right_beta": 0.2,
                        "left_posteriorProbability": 0.91,
                        "right_posteriorProbability": 0.92,
                    },
                },
            ],
            # expected coloc
            [
                {
                    "h0": 1.02277006860577e-4,
                    "h1": 2.752255943423052e-4,
                    "h2": 3.718914358059273e-4,
                    "h3": 1.5042926116520848e-6,
                    "h4": 0.9992491016906891,
                },
            ],
        ),
        # Case where PPs are high on the left, but low on the right:
        (
            # observed overlap
            [
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp1",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": None,
                        "left_beta": 0.003,
                        "right_beta": None,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.01,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp2",
                    "statistics": {
                        "left_logBF": 1.2,
                        "right_logBF": 3.8,
                        "left_beta": 0.003,
                        "right_beta": 0.005,
                        "left_posteriorProbability": 0.001,
                        "right_posteriorProbability": 0.01,
                    },
                },
                {
                    "leftStudyLocusId": "1",
                    "rightStudyLocusId": "2",
                    "rightStudyType": "eqtl",
                    "chromosome": "1",
                    "tagVariantId": "snp3",
                    "statistics": {
                        "left_logBF": 10.2,
                        "right_logBF": 10.5,
                        "left_beta": 0.5,
                        "right_beta": 0.2,
                        "left_posteriorProbability": 0.36,
                        "right_posteriorProbability": 0.92,
                    },
                },
            ],
            # expected coloc
            [],
        ),
    ],
)
def test_coloc_semantic(
    spark: SparkSession,
    observed_data: list[Any],
    expected_data: list[Any],
) -> None:
    """Test our COLOC with the implementation in R."""
    observed_overlap = StudyLocusOverlap(
        _df=spark.createDataFrame(observed_data, schema=StudyLocusOverlap.get_schema()),
        _schema=StudyLocusOverlap.get_schema(),
    )

    observed_coloc_df = Coloc.colocalise(observed_overlap).df

    # Define schema for the expected DataFrame
    result_schema = StructType(
        [
            StructField("h0", DoubleType(), True),
            StructField("h1", DoubleType(), True),
            StructField("h2", DoubleType(), True),
            StructField("h3", DoubleType(), True),
            StructField("h4", DoubleType(), True),
        ]
    )

    if not expected_data:
        expected_coloc_df = spark.createDataFrame([], schema=result_schema)
    else:
        expected_coloc_df = spark.createDataFrame(expected_data, schema=result_schema)

    if observed_coloc_df.rdd.isEmpty():
        observed_coloc_df = spark.createDataFrame([], schema=result_schema)

    observed_coloc_df = observed_coloc_df.select("h0", "h1", "h2", "h3", "h4")

    observed_coloc_pdf = observed_coloc_df.toPandas()
    expected_coloc_pdf = expected_coloc_df.toPandas()

    if expected_coloc_pdf.empty:
        assert observed_coloc_pdf.empty, (
            f"Expected an empty DataFrame, but got:\n{observed_coloc_pdf}"
        )
    else:
        assert_frame_equal(
            observed_coloc_pdf,
            expected_coloc_pdf,
            check_exact=False,
            check_dtype=True,
        )


def test_coloc_no_logbf(
    spark: SparkSession,
    minimum_expected_h0: float = 0.99,
    maximum_expected_h4: float = 1e-5,
) -> None:
    """Test COLOC output when the input data has irrelevant logBF."""
    observed_overlap = StudyLocusOverlap(
        (
            spark.createDataFrame(
                [
                    {
                        "leftStudyLocusId": "1",
                        "rightStudyLocusId": "2",
                        "rightStudyType": "eqtl",
                        "chromosome": "1",
                        "tagVariantId": "snp",
                        "statistics": {
                            "left_logBF": None,
                            "right_logBF": None,
                            "left_beta": 0.1,
                            "right_beta": 0.2,
                            "left_posteriorProbability": 0.91,
                            "right_posteriorProbability": 0.92,
                        },  # irrelevant for COLOC
                    }
                ],
                schema=StructType(
                    [
                        StructField("leftStudyLocusId", StringType(), False),
                        StructField("rightStudyLocusId", StringType(), False),
                        StructField("rightStudyType", StringType(), False),
                        StructField("chromosome", StringType(), False),
                        StructField("tagVariantId", StringType(), False),
                        StructField(
                            "statistics",
                            StructType(
                                [
                                    StructField("left_logBF", DoubleType(), True),
                                    StructField("right_logBF", DoubleType(), True),
                                    StructField("left_beta", DoubleType(), False),
                                    StructField("right_beta", DoubleType(), False),
                                    StructField(
                                        "left_posteriorProbability", DoubleType(), True
                                    ),
                                    StructField(
                                        "right_posteriorProbability", DoubleType(), True
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            )
        ),
        StudyLocusOverlap.get_schema(),
    )
    observed_coloc_df = Coloc.colocalise(observed_overlap).df
    assert observed_coloc_df.select("h0").collect()[0]["h0"] > minimum_expected_h0, (
        "COLOC should return a high h0 (no association) when the input data has irrelevant logBF."
    )
    assert observed_coloc_df.select("h4").collect()[0]["h4"] < maximum_expected_h4, (
        "COLOC should return a low h4 (traits are associated) when the input data has irrelevant logBF."
    )


def test_coloc_no_betas(spark: SparkSession) -> None:
    """Test COLOC output when the input data has no betas."""
    observed_overlap = StudyLocusOverlap(
        (
            spark.createDataFrame(
                [
                    {
                        "leftStudyLocusId": "1",
                        "rightStudyLocusId": "2",
                        "rightStudyType": "eqtl",
                        "chromosome": "1",
                        "tagVariantId": "snp",
                        "statistics": {
                            "left_logBF": 10.5,
                            "right_logBF": 10.3,
                            "left_beta": None,
                            "right_beta": None,
                            "left_posteriorProbability": 0.91,
                            "right_posteriorProbability": 0.92,
                        },  # irrelevant for COLOC
                    }
                ],
                schema=StructType(
                    [
                        StructField("leftStudyLocusId", StringType(), False),
                        StructField("rightStudyLocusId", StringType(), False),
                        StructField("rightStudyType", StringType(), False),
                        StructField("chromosome", StringType(), False),
                        StructField("tagVariantId", StringType(), False),
                        StructField(
                            "statistics",
                            StructType(
                                [
                                    StructField("left_logBF", DoubleType(), False),
                                    StructField("right_logBF", DoubleType(), False),
                                    StructField("left_beta", DoubleType(), True),
                                    StructField("right_beta", DoubleType(), True),
                                    StructField(
                                        "left_posteriorProbability", DoubleType(), True
                                    ),
                                    StructField(
                                        "right_posteriorProbability", DoubleType(), True
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            )
        ),
        StudyLocusOverlap.get_schema(),
    )
    observed_coloc_df = Coloc.colocalise(observed_overlap).df
    assert (
        observed_coloc_df.select("betaRatioSignAverage").collect()[0][
            "betaRatioSignAverage"
        ]
        is None
    ), "No betas results in None type."


def test_ecaviar(mock_study_locus_overlap: StudyLocusOverlap) -> None:
    """Test eCAVIAR."""
    assert isinstance(ECaviar.colocalise(mock_study_locus_overlap), Colocalisation)
