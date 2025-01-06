"""Step to generate biosample index dataset."""

from __future__ import annotations

from gentropy.common.session import Session
from gentropy.datasource.biosample_ontologies.utils import extract_ontology_from_json


class BiosampleIndexStep:
    """Biosample index step.

    This step generates a Biosample index dataset from the various ontology sources. Currently Cell Ontology and Uberon are supported.
    """

    def __init__(
        self,
        session: Session,
        cell_ontology_input_path: str,
        uberon_input_path: str,
        efo_input_path: str,
        biosample_index_path: str,
    ) -> None:
        """Run Biosample index generation step.

        Args:
            session (Session): Session object.
            cell_ontology_input_path (str): Input cell ontology dataset path.
            uberon_input_path (str): Input uberon dataset path.
            efo_input_path (str): Input efo dataset path.
            biosample_index_path (str): Output biosample index dataset path.
        """
        cell_ontology_index = extract_ontology_from_json(
            cell_ontology_input_path, session.spark
        )
        uberon_index = extract_ontology_from_json(uberon_input_path, session.spark)
        efo_index = extract_ontology_from_json(
            efo_input_path, session.spark
        ).retain_rows_with_ancestor_id(["CL_0000000"])

        biosample_index = cell_ontology_index.merge_indices([uberon_index, efo_index])

        biosample_index.df.coalesce(session.output_partitions).write.mode(
            session.write_mode
        ).parquet(biosample_index_path)
