"""Step to generate colocalisation results."""

from __future__ import annotations

from functools import partial
from typing import Any

from pyspark.sql.functions import col

from gentropy.common.session import Session
from gentropy.dataset.study_locus import FinemappingMethod, StudyLocus
from gentropy.method.colocalisation import Coloc, ColocalisationMethodInterface


class ColocalisationStep:
    """Colocalisation step.

    This workflow runs colocalisation analyses that assess the degree to which independent signals of the association share the same causal variant in a region of the genome, typically limited by linkage disequilibrium (LD).
    """

    __coloc_methods__ = {
        method.METHOD_NAME.lower(): method
        for method in ColocalisationMethodInterface.__subclasses__()
    }

    def __init__(
        self,
        session: Session,
        credible_set_path: str,
        coloc_path: str,
        colocalisation_method: str,
        colocalisation_method_params: dict[str, Any] | None = None,
    ) -> None:
        """Run Colocalisation step.

        This step allows for running two colocalisation methods: ecaviar and coloc.

        Args:
            session (Session): Session object.
            credible_set_path (str): Input credible sets path.
            coloc_path (str): Output Colocalisation path.
            colocalisation_method (str): Colocalisation method.
            colocalisation_method_params (dict[str, Any] | None): Keyword arguments passed to the colocalise method of Colocalisation class. Defaults to None

        Keyword Args:
            priorc1 (float): Prior on variant being causal for trait 1. Defaults to 1e-4. For coloc method only.
            priorc2 (float): Prior on variant being causal for trait 2. Defaults to 1e-4. For coloc method only.
            priorc12 (float): Prior on variant being causal for both traits. Defaults to 1e-5. For coloc method only.
        """
        colocalisation_method = colocalisation_method.lower()
        colocalisation_class = self._get_colocalisation_class(colocalisation_method)

        # Extract
        credible_set = StudyLocus.from_parquet(
            session, credible_set_path, recusiveFileLookup=True
        )
        if colocalisation_method == Coloc.METHOD_NAME.lower():
            credible_set = credible_set.filter(
                col("finemappingMethod").isin(
                    FinemappingMethod.SUSIE.value, FinemappingMethod.SUSIE_INF.value
                )
            )

        # Transform
        overlaps = credible_set.find_overlaps()

        # Make a partial caller to ensure that colocalisation_method_params are added to the call only when dict is not empty
        coloc = colocalisation_class.colocalise
        if colocalisation_method_params:
            coloc = partial(coloc, **colocalisation_method_params)
        colocalisation_results = coloc(overlaps)
        # Load
        colocalisation_results.df.coalesce(session.output_partitions).write.mode(
            session.write_mode
        ).parquet(f"{coloc_path}/{colocalisation_method.lower()}")

    @classmethod
    def _get_colocalisation_class(
        cls, method: str
    ) -> type[ColocalisationMethodInterface]:
        """Get colocalisation class.

        Args:
            method (str): Colocalisation method.

        Returns:
            type[ColocalisationMethodInterface]: Class that implements the ColocalisationMethodInterface.

        Raises:
            ValueError: if method not available.

        Examples:
            >>> ColocalisationStep._get_colocalisation_class("ECaviar")
            <class 'gentropy.method.colocalisation.ECaviar'>
        """
        method = method.lower()
        if method not in cls.__coloc_methods__:
            raise ValueError(f"Colocalisation method {method} not available.")
        return cls.__coloc_methods__[method]
