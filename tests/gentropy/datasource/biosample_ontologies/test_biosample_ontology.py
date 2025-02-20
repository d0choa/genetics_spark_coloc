"""Tests for biosample index dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gentropy.dataset.biosample_index import BiosampleIndex
from gentropy.datasource.biosample_ontologies.utils import extract_ontology_from_json

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class TestOntologyParger:
    """Testing ontology parser."""

    SAMPLE_CELL_ONTOLOGY_PATH = "tests/gentropy/data_samples/cell_ontology_sample.json"
    SAMPLE_UBERON_PATH = "tests/gentropy/data_samples/uberon_sample.json"
    SAMPLE_EFO_PATH = "tests/gentropy/data_samples/efo_biosample_sample.json"

    def test_ontology_parser(self: TestOntologyParger, spark: SparkSession) -> None:
        """Test all ontology parsers."""
        cell_ontology = extract_ontology_from_json(
            self.SAMPLE_CELL_ONTOLOGY_PATH, spark
        )
        uberon = extract_ontology_from_json(self.SAMPLE_UBERON_PATH, spark)
        efo_cell_line = extract_ontology_from_json(
            self.SAMPLE_EFO_PATH, spark
        ).retain_rows_with_ancestor_id(["CL_0000000"])

        assert isinstance(cell_ontology, BiosampleIndex), (
            "Cell ontology subset is not parsed correctly to BiosampleIndex."
        )
        assert isinstance(uberon, BiosampleIndex), (
            "Uberon subset is not parsed correctly to BiosampleIndex."
        )
        assert isinstance(efo_cell_line, BiosampleIndex), (
            "EFO cell line subset is not parsed correctly to BiosampleIndex."
        )

    def test_merge_biosample_indices(
        self: TestOntologyParger, spark: SparkSession
    ) -> None:
        """Test merging of biosample indices."""
        cell_ontology = extract_ontology_from_json(
            self.SAMPLE_CELL_ONTOLOGY_PATH, spark
        )
        uberon = extract_ontology_from_json(self.SAMPLE_UBERON_PATH, spark)
        efo = extract_ontology_from_json(self.SAMPLE_EFO_PATH, spark)

        merged = cell_ontology.merge_indices([uberon, efo])
        assert isinstance(merged, BiosampleIndex), (
            "Merging of biosample indices is not correct."
        )
