"""Test suite for the variants module of Open Targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gentropy.dataset.study_locus import StudyLocus
from gentropy.datasource.open_targets.variants import OpenTargetsVariant

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from gentropy.common.session import Session


class TestOpenTargetsVariant:
    """Test suite for the OpenTargetsVariant class."""

    def test_as_vcf_df_credible_set(
        self: TestOpenTargetsVariant,
        spark: SparkSession,
        session: Session,
    ) -> None:
        """Test the as_vcf_df method."""
        df_credible_set_df = spark.createDataFrame(
            [
                {
                    "studyLocusId": "1",
                    "variantId": "1_2_C_G",
                    "studyId": "study1",
                    "locus": [
                        {
                            "variantId": "1_3_A_T",
                        },
                    ],
                },
            ],
            StudyLocus.get_schema(),
        )
        observed_df = OpenTargetsVariant.as_vcf_df(session, df_credible_set_df).orderBy(
            *["#CHROM", "POS", "REF", "ALT"]
        )

        vcf_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
        df_credible_set_expected_df = spark.createDataFrame(
            [
                ("1", 2, ".", "C", "G", ".", ".", "."),
                ("1", 3, ".", "A", "T", ".", ".", "."),
            ],
            vcf_cols,
        )
        assert observed_df.collect() == df_credible_set_expected_df.collect(), (
            "Unexpected VCF dataframe."
        )

    def test_as_vcf_df_without_variant_id(
        self: TestOpenTargetsVariant,
        spark: SparkSession,
        session: Session,
    ) -> None:
        """Test the as_vcf_df method."""
        df_without_variant_id_df = spark.createDataFrame(
            [("rs75493593",)], ["variantRsId"]
        )
        observed_df = OpenTargetsVariant.as_vcf_df(
            session, df_without_variant_id_df
        ).orderBy(*["#CHROM", "POS", "REF", "ALT"])

        assert observed_df.count() == 0, "A variant ID should be present for VCF step."

    def test_as_vcf_df_without_rs_id(
        self: TestOpenTargetsVariant,
        spark: SparkSession,
        session: Session,
    ) -> None:
        """Test the as_vcf_df method with a dataframe of variants without an annotated variantRsId."""
        df_without_rs_id_df = spark.createDataFrame([("1_2_G_GA",)], ["variantId"])
        observed_df = OpenTargetsVariant.as_vcf_df(session, df_without_rs_id_df)

        vcf_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
        df_without_rs_id_expected_df = spark.createDataFrame(
            [
                ("1", 2, ".", "G", "GA", ".", ".", "."),
            ],
            vcf_cols,
        )

        assert observed_df.collect() == df_without_rs_id_expected_df.collect(), (
            "Unexpected VCF dataframe."
        )
