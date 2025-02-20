"""Tests for study index dataset from FinnGen."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import types as T

from gentropy.dataset.study_index import StudyIndex
from gentropy.datasource.finngen.study_index import (
    FinngenPrefixMatch,
    FinnGenStudyIndex,
)
from gentropy.finngen_studies import FinnGenStudiesStep

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyspark.sql import SparkSession

    from gentropy.common.session import Session


@pytest.fixture()
def finngen_study_index_mock(spark: SparkSession) -> StudyIndex:
    """Finngen minimal example for mocking join to the efo mappings."""
    data = [
        # NOTE: Study maps to a single EFO trait
        (
            "FINNGEN_R11_STUDY_1",
            "Actinomycosis",
            "FINNGEN_R11",
            "gwas",
        ),
        # NOTE: Study does not map to EFO traits
        (
            "FINNGEN_R11_STUDY_2",
            "Some other trait",
            "FINNGEN_R11",
            "gwas",
        ),
        # NOTE: Study maps to two EFO traits
        (
            "FINNGEN_R11_STUDY_3",
            "Glucose",
            "FINNGEN_R11",
            "gwas",
        ),
    ]
    schema = T.StructType(
        [
            T.StructField("studyId", T.StringType(), nullable=False),
            T.StructField("traitFromSource", T.StringType(), nullable=False),
            T.StructField("projectId", T.StringType(), nullable=False),
            T.StructField("studyType", T.StringType(), nullable=False),
        ]
    )
    df = spark.createDataFrame(data=data, schema=schema)
    return StudyIndex(_df=df, _schema=StudyIndex.get_schema())


@pytest.fixture()
def finngen_phenotype_table_mock() -> str:
    """This is the data extracted from https://r11.finngen.fi/api/phenos."""
    data = json.dumps(
        [
            # NOTE: Study maps to single EFO trait.
            {
                "assoc_files": [
                    "/cromwell_root/pheweb/generated-by-pheweb/pheno_gz/AB1_ACTINOMYCOSIS.gz"
                ],
                "category": "I Certain infectious and parasitic diseases (AB1_)",
                "category_index": 1,
                "gc_lambda": {
                    "0.001": 0.93878,
                    "0.01": 0.96727,
                    "0.1": 0.85429,
                    "0.5": 0.52544,
                },
                "num_cases": 113,
                "num_cases_prev": 101,
                "num_controls": 399149,
                "num_controls_prev": 363227,
                "num_gw_significant": 0,
                "num_gw_significant_prev": 0,
                "phenocode": "AB1_ACTINOMYCOSIS",
                "phenostring": "Actinomycosis",
            },
            # NOTE: Study maps to multiple EFO traits.
            {
                "assoc_files": [
                    "/cromwell_root/pheweb/generated-by-pheweb/pheno_gz/GLUCOSE.gz"
                ],
                "category": "Glucose",
                "category_index": 28,
                "gc_lambda": {
                    "0.001": 1.1251,
                    "0.01": 1.062,
                    "0.1": 1.0531,
                    "0.5": 1.0599,
                },
                "num_cases": 43764,
                "num_cases_prev": 39231,
                "num_controls": 409969,
                "num_controls_prev": 372950,
                "num_gw_significant": 3,
                "num_gw_significant_prev": 3,
                "phenocode": "GLUCOSE",
                "phenostring": "Glucose",
            },
            # NOTE: Study does not map to EFO traits
            {
                "assoc_files": [
                    "/cromwell_root/pheweb/generated-by-pheweb/pheno_gz/SOME_OTHER_TRAIT.gz"
                ],
                "category": "SomeOtherTrait",
                "category_index": 28,
                "gc_lambda": {
                    "0.001": 1.1251,
                    "0.01": 1.062,
                    "0.1": 1.0531,
                    "0.5": 1.0599,
                },
                "num_cases": 43764,
                "num_cases_prev": 39231,
                "num_controls": 409969,
                "num_controls_prev": 372950,
                "num_gw_significant": 3,
                "num_gw_significant_prev": 3,
                "phenocode": "SOME_OTHER_TRAIT",
                "phenostring": "Some other trait",
            },
        ]
    )
    return data


@pytest.fixture()
def efo_mappings_mock() -> list[tuple[str, str, str]]:
    """EFO mappings mock based on https://raw.githubusercontent.com/opentargets/curation/24.09.1/mappings/disease/manual_string.tsv.

    Only required fields are extracted.
    """
    data = [
        (
            "STUDY",
            "PROPERTY_VALUE",
            "SEMANTIC_TAG",
        ),
        ("FinnGen r11", "Actinomycosis", "http://www.ebi.ac.uk/efo/EFO_0007128"),
        # NOTE: EFO does not map, as it's missing from the StudyIndex - hypothetical example.
        ("FinnGen r11", "Bleeding", "http://purl.obolibrary.org/obo/MP_0001914"),
        # NOTE: Two EFO traits for one disease should be collected to array - hypothetical example:
        # Glucose tolerance test & NMR Glucose
        ("FinnGen r11", "Glucose", "http://www.ebi.ac.uk/efo/EFO_0002571"),
        ("FinnGen r11", "Glucose", "http://www.ebi.ac.uk/efo/EFO_0004468"),
        # NOTE: EFO that does not map, due to study not from Finngen - hypothetical example.
        ("PheWAS 2024", "Glucose", "http://www.ebi.ac.uk/efo/EFO_0000001"),
    ]
    return data


@pytest.fixture()
def efo_mappings_df_mock(
    spark: SparkSession, efo_mappings_mock: list[tuple[str, str, str]]
) -> DataFrame:
    """EFO mappings dataframe mock."""
    schema = T.StructType(
        [
            T.StructField("STUDY", T.StringType(), nullable=False),
            T.StructField("PROPERTY_VALUE", T.StringType(), nullable=False),
            T.StructField("SEMANTIC_TAG", T.StringType(), nullable=False),
        ]
    )
    data = spark.createDataFrame(data=efo_mappings_mock, schema=schema)
    return data


@pytest.fixture()
def urlopen_mock(
    efo_mappings_mock: list[tuple[str, str, str, str]],
    finngen_phenotype_table_mock: str,
) -> Callable[[str], MagicMock]:
    """Mock object for requesting urlopen objects with proper encoding.

    This mock object allows to call `read` and `readlines` methods on two endpoints:
    - https://finngen_phenotypes -> finngen_phenotype_table_mock
    - https://efo_mappings -> efo_mappings_mock

    The return values are mocks of the source data respectively.
    """

    def mock_response(url: str) -> MagicMock:
        """Mock urllib.request.urlopen."""
        match url:
            case "https://finngen_phenotypes":
                value = finngen_phenotype_table_mock
            case "https://efo_mappings":
                value = "\n".join(["\t".join(row) for row in efo_mappings_mock])
            case _:
                value = ""
        mock_open = MagicMock()
        mock_open.read.return_value = value.encode()
        mock_open.readlines.return_value = value.encode().splitlines(keepends=True)
        return mock_open

    return mock_response


@pytest.mark.step_test
def test_finngen_study_index_step(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    tmp_path: Path,
    urlopen_mock: Callable[[str], MagicMock],
) -> None:
    """Test step that generates finngen study index.

    FIXME: Currently we miss following columns when reading from source.
    'biosampleFromSourceId'
    'publicationTitle'
    'diseaseIds'
    'publicationDate'
    'geneId'
    'backgroundDiseaseIds'
    'pubmedId'
    'publicationJournal'
    'qualityControls'
    'backgroundTraitFromSourceMappedIds'
    'publicationFirstAuthor'
    'replicationSamples'
    'analysisFlags'
    'condition'
    """
    with monkeypatch.context() as m:
        m.setattr("gentropy.datasource.finngen.study_index.urlopen", urlopen_mock)
        output_path = str(tmp_path / "study_index")
        FinnGenStudiesStep(
            session=session,
            finngen_study_index_out=output_path,
            finngen_phenotype_table_url="https://finngen_phenotypes",
            finngen_release_prefix="FINNGEN_R11",
            finngen_summary_stats_url_prefix="gs://finngen_data/sumstats",
            finngen_summary_stats_url_suffix=".gz",
            efo_curation_mapping_url="https://efo_mappings",
            sample_size=5_000_000,
        )
        study_index = StudyIndex.from_parquet(session=session, path=output_path)
        # fmt: off
        assert study_index.df.count() == 3, "Expected 3 rows that come from the input table."
        assert "traitFromSourceMappedIds" in study_index.df.columns, "Expected that EFO terms were joined to the study_index table."
        # fmt: on


def test_finngen_study_index_read_efo_curation(
    monkeypatch: pytest.MonkeyPatch,
    spark: SparkSession,
    urlopen_mock: Callable[[str], MagicMock],
) -> None:
    """Test reading efo curation."""
    with monkeypatch.context() as m:
        m.setattr("gentropy.datasource.finngen.study_index.urlopen", urlopen_mock)
        efo_df = FinnGenStudyIndex.read_efo_curation(spark, "https://efo_mappings")
        assert isinstance(efo_df, DataFrame)
        efo_df.show()
        assert efo_df.count() == 5


def test_finngen_study_index_from_source(
    monkeypatch: pytest.MonkeyPatch,
    spark: SparkSession,
    urlopen_mock: Callable[[str], MagicMock],
) -> None:
    """Test study index from source."""
    with monkeypatch.context() as m:
        m.setattr("gentropy.datasource.finngen.study_index.urlopen", urlopen_mock)
        expected_sample_size = 5_000_000
        expected_project_id = "FINNGEN_R11"
        study_index = FinnGenStudyIndex.from_source(
            spark,
            finngen_phenotype_table_url="https://finngen_phenotypes",
            finngen_release_prefix=expected_project_id,
            finngen_summary_stats_url_prefix="gs://finngen-public-data-r11/summary_stats/finngen_R11_",
            finngen_summary_stats_url_suffix=".gz",
            sample_size=expected_sample_size,
        )
        # fmt: off
        assert isinstance(study_index, StudyIndex), "Expect that we deal with StudyIndex object."

        all_columns = StudyIndex.get_schema().fieldNames()
        assert set(all_columns).issuperset(set(study_index.df.columns)), "Expect all columns can be found in the schema of StudyIndex."
        assert study_index.df.count() == 3, "Expect two rows at the study_index, as in the input."

        rows = study_index.df.collect()
        expected_study_ids = ["FINNGEN_R11_AB1_ACTINOMYCOSIS", "FINNGEN_R11_GLUCOSE", "FINNGEN_R11_SOME_OTHER_TRAIT"]
        assert "studyId" in study_index.df.columns, "Expect that studyId column exists."
        assert sorted([v["studyId"] for v in rows]) == expected_study_ids, "Expect that studyIds are populated from input."

        assert "projectId" in study_index.df.columns, "Expect that projectId column exists."
        assert {v["projectId"] for v in rows} == {expected_project_id}, "Expect projectId column is correctly populated."

        expected_sumstat_locations = [
            "gs://finngen-public-data-r11/summary_stats/finngen_R11_AB1_ACTINOMYCOSIS.gz",
            "gs://finngen-public-data-r11/summary_stats/finngen_R11_GLUCOSE.gz",
            "gs://finngen-public-data-r11/summary_stats/finngen_R11_SOME_OTHER_TRAIT.gz",
        ]
        assert "summarystatsLocation" in study_index.df.columns, "Expect that summarystatsLocation column exists."
        sumstat_locations = sorted([v["summarystatsLocation"] for v in rows])
        assert sumstat_locations == expected_sumstat_locations, "Expect that summarystatsLocation is populated."
        assert "ldPopulationStructure" in study_index.df.columns, "Expect that ldPopulationStructure column exists."
        for row in rows:
            ld_struct = row["ldPopulationStructure"][0]
            assert ld_struct["ldPopulation"] == "fin", "Expect fin ld population structure."
            assert ld_struct["relativeSampleSize"] == pytest.approx(1.0), "Expect relative sample size if fixed to be 1.0."

        assert "discoverySamples" in study_index.df.columns, "Expect that discoverySamples column exists."
        for row in rows:
            ds_struct = row["discoverySamples"][0]
            assert ds_struct["ancestry"] == "Finnish", "Expect Finnish ancestry."
            assert ds_struct["sampleSize"] == expected_sample_size, "Expect sample size to be fixed."
        # fmt: on


@pytest.mark.parametrize(
    ["prefix", "expected_output", "xfail"],
    [
        pytest.param(
            "FINNGEN_R11",
            FinngenPrefixMatch(prefix="FINNGEN_R11", release="R11"),
            False,
            id="Correct prefix passed.",
        ),
        pytest.param(
            "FINNGEN_R11_",
            FinngenPrefixMatch(prefix="FINNGEN_R11", release="R11"),
            False,
            id="Underscore is removed from the prefix.",
        ),
        pytest.param(
            "R11",
            FinngenPrefixMatch(prefix="FINNGEN_R11", release="R11"),
            True,
            id="Incorrect prefix raises ValueError.",
        ),
    ],
)
def test_finngen_validate_release_prefix(
    prefix: str, expected_output: FinngenPrefixMatch, xfail: bool
) -> None:
    """Test validate_release_prefix."""
    if not xfail:
        assert FinnGenStudyIndex.validate_release_prefix(prefix) == expected_output, (
            "Incorrect match object"
        )
    else:
        with pytest.raises(ValueError):
            FinnGenStudyIndex.validate_release_prefix(prefix)


def test_finngen_study_index_add_efos(
    finngen_study_index_mock: StudyIndex,
    efo_mappings_df_mock: DataFrame,
) -> None:
    """Test finngen study index add efo ids."""
    efo_column_name = "traitFromSourceMappedIds"
    # Expect that EFO column is not present when study index is generated.
    assert efo_column_name not in finngen_study_index_mock.df.columns
    study_index = FinnGenStudyIndex.join_efo_mapping(
        finngen_study_index_mock,
        finngen_release="R11",
        efo_curation_mapping=efo_mappings_df_mock,
    )
    # fmt: off
    assert isinstance(study_index, StudyIndex), "Expect we have the StudyIndex object after joining EFOs."
    assert efo_column_name in study_index.df.columns, "Expect that EFO column is present after joining EFOs."
    assert study_index.df.count() == 3, "Expect we do not drop any studies, even if no EFO has been found."
    # fmt: on
    efos = {
        row["studyId"]: sorted(row[efo_column_name])
        for row in study_index.df.select(efo_column_name, "studyId").collect()
    }
    expected_efos = {
        "FINNGEN_R11_STUDY_1": ["EFO_0007128"],
        "FINNGEN_R11_STUDY_2": [],
        "FINNGEN_R11_STUDY_3": ["EFO_0002571", "EFO_0004468"],
    }
    assert expected_efos == efos, "Expect that EFOs are correctly assigned."
