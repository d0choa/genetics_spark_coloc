"""Testing VEP parsing and variant index extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from gentropy.dataset.variant_index import VariantIndex
from gentropy.datasource.ensembl.vep_parser import VariantEffectPredictorParser

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class TestVEPParserVariantEffectExtractor:
    """Testing the _vep_variant_effect_extractor method of the VEP parser class.

    These tests assumes that the _get_most_severe_transcript() method works correctly, as it's not tested.

    The test cases try to cover the following scenarios:
    - Transcripts with no assessments.
    - Transcripts without assessments flag.
    - Transcripts with no score.
    - Testing cases, where no score column is provided.
    """

    # Data prototype:
    SAMPLE_DATA = [
        # Complete dataset:
        ("v1", "deleterious", 0.1, "gene1", "flag"),
        # No assessment:
        ("v2", None, 0.1, "gene1", "flag"),
        # No flag:
        ("v3", "deleterious", 0.1, "gene1", None),
        # No score:
        ("v4", "deleterious", None, "gene1", "flag"),
    ]

    SAMPLE_COLUMNS = ["variantId", "assessment", "score", "gene_id", "flag"]

    @pytest.fixture(autouse=True)
    def _setup(self: TestVEPParserVariantEffectExtractor, spark: SparkSession) -> None:
        """Setup fixture."""
        parsed_df = (
            spark.createDataFrame(self.SAMPLE_DATA, self.SAMPLE_COLUMNS)
            .groupBy("variantId")
            .agg(
                f.collect_list(
                    f.struct(
                        f.col("assessment").alias("assessment"),
                        f.col("score").alias("score"),
                        f.col("flag").alias("flag"),
                        f.col("gene_id").alias("gene_id"),
                    )
                ).alias("transcripts")
            )
            .select(
                "variantId",
                VariantEffectPredictorParser._vep_variant_effect_extractor(
                    "transcripts", "method_name", "score", "assessment", "flag"
                ).alias("variant_effect"),
            )
        ).persist()

        self.df = parsed_df

    def test_variant_effect_missing_value(
        self: TestVEPParserVariantEffectExtractor,
    ) -> None:
        """Test if the variant effect count is correct."""
        variant_with_missing_score = [
            x[0] for x in filter(lambda x: x[2] is None, self.SAMPLE_DATA)
        ]
        # Assert that the correct variants return null:
        assert [
            x["variantId"]
            for x in self.df.filter(f.col("variant_effect").isNull()).collect()
        ] == variant_with_missing_score, (
            "Not the right variants got nullified in variant effect object."
        )


class TestVEPParser:
    """Testing VEP parser class.

    Some of the input data:
    - 6_151445307_C_T - complicated variant with numerous annotations.
    - 2_140699625_G_GT - simple variant with no annotations whatsoever.
    """

    SAMPLE_VEP_DATA_PATH = "tests/gentropy/data_samples/vep_sample.jsonl"

    @pytest.fixture(autouse=True)
    def _setup(self: TestVEPParser, spark: SparkSession) -> None:
        """Setup fixture."""
        self.raw_vep_output = spark.read.json(
            self.SAMPLE_VEP_DATA_PATH,
            schema=VariantEffectPredictorParser.get_schema(),
        )
        self.processed_vep_output = VariantEffectPredictorParser.process_vep_output(
            self.raw_vep_output, 200
        )

    def test_process(self: TestVEPParser) -> None:
        """Test process method."""
        df = VariantEffectPredictorParser.process_vep_output(self.raw_vep_output)
        assert isinstance(df, DataFrame), "Processed VEP output is not a DataFrame."
        assert df.count() > 0, "No variant data in processed VEP dataframe."

    def test_conversion(self: TestVEPParser) -> None:
        """Test if processed data can be converted into a VariantIndex object."""
        variant_index = VariantIndex(
            _df=self.processed_vep_output,
            _schema=VariantIndex.get_schema(),
        )

        assert isinstance(variant_index, VariantIndex), (
            "VariantIndex object not created."
        )

    def test_variant_count(self: TestVEPParser) -> None:
        """Test if the number of variants is correct.

        It is expected that all rows from the parsed VEP output are present in the processed VEP output.
        """
        assert self.raw_vep_output.count() == self.processed_vep_output.count(), (
            f"Incorrect number of variants in processed VEP output: expected {self.raw_vep_output.count()}, got {self.processed_vep_output.count()}."
        )

    def test_collection(self: TestVEPParser) -> None:
        """Test if the collection of VEP variantIndex runs without failures."""
        assert (
            len(self.processed_vep_output.collect())
            == self.processed_vep_output.count()
        ), "Collection performed incorrectly."

    def test_ensembl_transcripts_no_duplicates(self: TestVEPParser) -> None:
        """Test if in single row all ensembl target ids (gene ids) do not have duplicates."""
        targets = (
            self.processed_vep_output.limit(1)
            .select(f.explode("transcriptConsequences").alias("t"))
            .select("t.targetId")
            .collect()
        )

        asserted_targets = [t["targetId"] for t in targets]
        assert len(asserted_targets) == len(set(asserted_targets)), (
            "Duplicate ensembl transcripts in a single row."
        )
