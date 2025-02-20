"""Test GWAS Catalog study curation logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t
import pytest
from pyspark.sql import DataFrame

from gentropy.datasource.gwas_catalog.study_index import StudyIndexGWASCatalog

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class TestGWASCatalogStudyCuration:
    """Test suite for GWAS Catalog study curation logic."""

    @pytest.fixture(scope="class")
    def mock_gwas_study_index(
        self: TestGWASCatalogStudyCuration,
        spark: SparkSession,
    ) -> StudyIndexGWASCatalog:
        """Generate a dataset with a given set of studies with minimalistic schema."""
        # Generating a study index with 7 studies where the first study (s0) has no summary statistics:
        study_data = [
            (f"s{count}", False) if count == 0 else (f"s{count}", True)
            for count in range(0, 7)
        ]
        columns = [
            "projectId",
            "studyType",
            "traitFromSource",
            "publicationTitle",
            "publicationFirstAuthor",
            "pubmedId",
        ]
        return StudyIndexGWASCatalog(
            _df=(
                spark.createDataFrame(study_data, ["studyId", "hasSumstats"]).select(
                    "*", *[f.lit("foo").alias(colname) for colname in columns]
                )
            ),
            _schema=StudyIndexGWASCatalog.get_schema(),
        )

    @pytest.fixture(scope="class")
    def mock_study_curation(
        self: TestGWASCatalogStudyCuration, spark: SparkSession
    ) -> DataFrame:
        """Generate a mocked curation dataset with matching studies."""
        curation_data = [
            ("s2", None, None, None, True),  # Good study
            ("s3", "pQTL", None, None, True),  # Update type
            ("s4", None, ["analysis 1"], None, True),  # Add analysis flag
            ("s5", None, None, ["QC flag"], True),  # Add analysis flag
        ]

        curation_columns = [
            "studyId",
            "studyType",
            "analysisFlags",
            "qualityControls",
            "isCurated",
        ]
        return (
            spark.createDataFrame(curation_data, curation_columns)
            .withColumn("isCurated", f.col("isCurated").cast(t.BooleanType()))
            .persist()
        )

    @staticmethod
    def test_curation__return_type(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Test return types."""
        # Method should work on empty curation:
        assert isinstance(
            mock_gwas_study_index.annotate_from_study_curation(None),
            StudyIndexGWASCatalog,
        ), (
            f"Applying curation without curation table should yield a study table, but got: {type(mock_gwas_study_index.annotate_from_study_curation(None))}"
        )

        # Return type should work:
        assert isinstance(
            mock_gwas_study_index.annotate_from_study_curation(mock_study_curation),
            StudyIndexGWASCatalog,
        ), (
            f"Applying curation should return a study table, however got: {type(mock_gwas_study_index.annotate_from_study_curation(mock_study_curation))}"
        )

    @staticmethod
    def test_curation__returned_rows(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Test return types."""
        expected_count = mock_gwas_study_index.df.count()
        zero_return_count = mock_gwas_study_index.annotate_from_study_curation(
            None
        ).df.count()
        return_count = mock_gwas_study_index.annotate_from_study_curation(
            mock_study_curation
        ).df.count()
        # Method should work on empty curation:
        assert zero_return_count == expected_count, (
            f"When applied None to curation function, the size of the returned data was not as expected ({zero_return_count} vs {expected_count})."
        )

        # Return type should work:
        assert return_count == expected_count, (
            f"When applied curation data, the size of the returned data was not as expected ({return_count} vs {expected_count})."
        )

    # Test updated type
    @staticmethod
    def test_curation__study_type_update(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Test for making sure the study type got updated."""
        curated = mock_gwas_study_index.annotate_from_study_curation(
            mock_study_curation
        )

        # Expected studyIds:
        expected = [
            row["studyId"]
            for row in (
                mock_study_curation.filter(f.col("studyType").isNotNull())
                .select("studyId")
                .distinct()
                .collect()
            )
        ]

        observed = [
            row["studyId"]
            for row in (
                curated.df.filter(f.col("studyType") != "foo")
                .select("studyId")
                .distinct()
                .collect()
            )
        ]

        assert expected == observed

    # Test updated method flag
    @staticmethod
    def test_curation__analysis_flags(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Test for making sure the study type got updated."""
        curated = mock_gwas_study_index.annotate_from_study_curation(
            mock_study_curation
        )

        # Expected studyIds:
        expected = [
            row["studyId"]
            for row in (
                mock_study_curation.filter(f.col("analysisFlags").isNotNull())
                .select("studyId")
                .distinct()
                .collect()
            )
        ]

        observed = [
            row["studyId"]
            for row in (
                curated.df.filter(f.size(f.col("analysisFlags")) > 0)
                .select("studyId")
                .distinct()
                .collect()
            )
        ]

        assert expected == observed

    # Test curation extraction:
    @staticmethod
    def test_extract_curation__return_type(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Testing if the extracted curation table has the right type."""
        assert isinstance(
            mock_gwas_study_index.extract_studies_for_curation(mock_study_curation),
            DataFrame,
        )
        # we don't need to have existing curation:
        assert isinstance(
            mock_gwas_study_index.extract_studies_for_curation(None),
            DataFrame,
        )

    @staticmethod
    def test_extract_curation__rows(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Testing if the extracted curation table has the right type."""
        new_curation_count = mock_gwas_study_index.extract_studies_for_curation(
            mock_study_curation
        ).count()
        new_empty_count = mock_gwas_study_index.extract_studies_for_curation(
            mock_study_curation
        ).count()
        expected_count = mock_gwas_study_index.df.filter(f.col("hasSumstats")).count()
        assert new_curation_count == expected_count
        assert new_empty_count == expected_count

    @staticmethod
    def test_extract_curation__new_studies_to_curate(
        mock_gwas_study_index: StudyIndexGWASCatalog, mock_study_curation: DataFrame
    ) -> None:
        """Testing if the extracted curation table has the right type."""
        new_curation_count = (
            mock_gwas_study_index.extract_studies_for_curation(mock_study_curation)
            .filter(f.lower(f.col("isCurated")) == "false")
            .count()
        )
        new_empty_count = (
            mock_gwas_study_index.extract_studies_for_curation(None)
            .filter(f.lower(f.col("isCurated")) == "false")
            .count()
        )
        assert new_curation_count == 2
        assert new_empty_count == 6
