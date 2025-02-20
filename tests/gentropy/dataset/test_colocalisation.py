"""Test colocalisation dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_colocalisation_creation(mock_colocalisation: Colocalisation) -> None:
    """Test colocalisation creation with mock data."""
    assert isinstance(mock_colocalisation, Colocalisation)


def test_append_study_metadata_study_locus(
    mock_colocalisation: Colocalisation,
    mock_study_locus: StudyLocus,
    mock_study_index: StudyIndex,
    metadata_cols: list[str] | None = None,
) -> None:
    """Test appending right study metadata."""
    if metadata_cols is None:
        metadata_cols = ["studyType"]
    expected_extra_col = ["rightStudyType", "rightStudyId"]
    res_df = mock_colocalisation.append_study_metadata(
        mock_study_locus,
        mock_study_index,
        metadata_cols=metadata_cols,
        colocalisation_side="right",
    )
    for col in expected_extra_col:
        assert col in res_df.columns, f"Column {col} not found in result DataFrame."


class TestAppendStudyMetadata:
    """Test Colocalisation.append_study_metadata method."""

    @pytest.mark.parametrize(
        ("colocalisation_side", "expected_geneId"), [("right", "g1"), ("left", None)]
    )
    def test_append_study_metadata_right(
        self: TestAppendStudyMetadata,
        colocalisation_side: str,
        expected_geneId: str | None,
        metadata_cols: list[str] | None = None,
    ) -> None:
        """Test appending right study metadata."""
        if metadata_cols is None:
            metadata_cols = ["geneId"]
        observed_df = self.sample_colocalisation.append_study_metadata(
            self.sample_study_locus,
            self.sample_study_index,
            metadata_cols=metadata_cols,
            colocalisation_side=colocalisation_side,
        )
        assert (
            observed_df.select(f"{colocalisation_side}GeneId").collect()[0][0]
            == expected_geneId
        ), (
            f"Expected {colocalisation_side}GeneId {expected_geneId}, but got {observed_df.select(f'{colocalisation_side}GeneId').collect()[0][0]}"
        )

    @pytest.fixture(autouse=True)
    def _setup(self: TestAppendStudyMetadata, spark: SparkSession) -> None:
        """Setup fixture."""
        self.sample_study_locus = StudyLocus(
            _df=spark.createDataFrame(
                [
                    (
                        "1",
                        "var1",
                        "gwas1",
                    ),
                    (
                        "2",
                        "var2",
                        "eqtl1",
                    ),
                ],
                ["studyLocusId", "variantId", "studyId"],
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


def test_extract_maximum_coloc_probability_per_region_and_gene(
    mock_colocalisation: Colocalisation,
    mock_study_locus: StudyLocus,
    mock_study_index: StudyIndex,
    filter_by_colocalisation_method: str | None = None,
) -> None:
    """Test extracting maximum coloc probability per region and gene returns a dataframe with the correct columns: studyLocusId, geneId, h4."""
    filter_by_colocalisation_method = filter_by_colocalisation_method or "Coloc"
    res_df = mock_colocalisation.extract_maximum_coloc_probability_per_region_and_gene(
        mock_study_locus,
        mock_study_index,
        filter_by_colocalisation_method=filter_by_colocalisation_method,
    )
    expected_cols = ["studyLocusId", "geneId", "h4"]
    for col in expected_cols:
        assert col in res_df.columns, f"Column {col} not found in result DataFrame."
