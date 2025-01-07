# isort: skip_file
"""Factory that computes features based on an input list."""

from __future__ import annotations

from typing import Any
from collections.abc import Iterator, Mapping

from gentropy.dataset.l2g_features.colocalisation import (
    EQtlColocClppMaximumFeature,
    EQtlColocClppMaximumNeighbourhoodFeature,
    EQtlColocH4MaximumFeature,
    EQtlColocH4MaximumNeighbourhoodFeature,
    PQtlColocClppMaximumFeature,
    PQtlColocClppMaximumNeighbourhoodFeature,
    PQtlColocH4MaximumFeature,
    PQtlColocH4MaximumNeighbourhoodFeature,
    SQtlColocClppMaximumFeature,
    SQtlColocClppMaximumNeighbourhoodFeature,
    SQtlColocH4MaximumFeature,
    SQtlColocH4MaximumNeighbourhoodFeature,
)
from gentropy.dataset.l2g_features.distance import (
    DistanceFootprintMeanFeature,
    DistanceFootprintMeanNeighbourhoodFeature,
    DistanceSentinelFootprintFeature,
    DistanceSentinelFootprintNeighbourhoodFeature,
    DistanceSentinelTssFeature,
    DistanceSentinelTssNeighbourhoodFeature,
    DistanceTssMeanFeature,
    DistanceTssMeanNeighbourhoodFeature,
)
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_features.other import (
    CredibleSetConfidenceFeature,
    GeneCountFeature,
    ProteinGeneCountFeature,
    ProteinCodingFeature,
)
from gentropy.dataset.l2g_features.vep import (
    VepMaximumFeature,
    VepMaximumNeighbourhoodFeature,
    VepMeanFeature,
    VepMeanNeighbourhoodFeature,
)
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_locus import StudyLocus


class L2GFeatureInputLoader:
    """Loads all input datasets required for the L2GFeature dataset."""

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initializes L2GFeatureInputLoader with the provided inputs and returns loaded dependencies as a dictionary.

        Args:
            **kwargs (Any): keyword arguments with the name of the dependency and the dependency itself.
        """
        self.input_dependencies = {k: v for k, v in kwargs.items() if v is not None}

    def get_dependency_by_type(
        self, dependency_type: list[Any] | Any
    ) -> dict[str, Any]:
        """Returns the dependency that matches the provided type.

        Args:
            dependency_type (list[Any] | Any): type(s) of the dependency to return.

        Returns:
            dict[str, Any]: dictionary of dependenci(es) that match the provided type(s).
        """
        if not isinstance(dependency_type, list):
            dependency_type = [dependency_type]
        return {
            k: v
            for k, v in self.input_dependencies.items()
            if isinstance(v, tuple(dependency_type))
        }

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Make the class iterable, returning an iterator over key-value pairs.

        Returns:
            Iterator[tuple[str, Any]]: iterator over the dictionary's key-value pairs.
        """
        return iter(self.input_dependencies.items())

    def __repr__(self) -> str:
        """Return a string representation of the input dependencies.

        Useful for understanding the loader content without having to print the object attribute.

        Returns:
            str: string representation of the input dependencies.
        """
        return repr(self.input_dependencies)


class FeatureFactory:
    """Factory class for creating features."""

    feature_mapper: Mapping[str, type[L2GFeature]] = {
        "distanceSentinelTss": DistanceSentinelTssFeature,
        "distanceSentinelTssNeighbourhood": DistanceSentinelTssNeighbourhoodFeature,
        "distanceSentinelFootprint": DistanceSentinelFootprintFeature,
        "distanceSentinelFootprintNeighbourhood": DistanceSentinelFootprintNeighbourhoodFeature,
        "distanceTssMean": DistanceTssMeanFeature,
        "distanceTssMeanNeighbourhood": DistanceTssMeanNeighbourhoodFeature,
        "distanceFootprintMean": DistanceFootprintMeanFeature,
        "distanceFootprintMeanNeighbourhood": DistanceFootprintMeanNeighbourhoodFeature,
        "eQtlColocClppMaximum": EQtlColocClppMaximumFeature,
        "eQtlColocClppMaximumNeighbourhood": EQtlColocClppMaximumNeighbourhoodFeature,
        "pQtlColocClppMaximum": PQtlColocClppMaximumFeature,
        "pQtlColocClppMaximumNeighbourhood": PQtlColocClppMaximumNeighbourhoodFeature,
        "sQtlColocClppMaximum": SQtlColocClppMaximumFeature,
        "sQtlColocClppMaximumNeighbourhood": SQtlColocClppMaximumNeighbourhoodFeature,
        "eQtlColocH4Maximum": EQtlColocH4MaximumFeature,
        "eQtlColocH4MaximumNeighbourhood": EQtlColocH4MaximumNeighbourhoodFeature,
        "pQtlColocH4Maximum": PQtlColocH4MaximumFeature,
        "pQtlColocH4MaximumNeighbourhood": PQtlColocH4MaximumNeighbourhoodFeature,
        "sQtlColocH4Maximum": SQtlColocH4MaximumFeature,
        "sQtlColocH4MaximumNeighbourhood": SQtlColocH4MaximumNeighbourhoodFeature,
        "vepMean": VepMeanFeature,
        "vepMeanNeighbourhood": VepMeanNeighbourhoodFeature,
        "vepMaximum": VepMaximumFeature,
        "vepMaximumNeighbourhood": VepMaximumNeighbourhoodFeature,
        "geneCount500kb": GeneCountFeature,
        "proteinGeneCount500kb": ProteinGeneCountFeature,
        "isProteinCoding": ProteinCodingFeature,
        "credibleSetConfidence": CredibleSetConfidenceFeature,
    }

    def __init__(
        self: FeatureFactory,
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        features_list: list[str],
    ) -> None:
        """Initializes the factory.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            features_list (list[str]): list of features to compute.
        """
        self.study_loci_to_annotate = study_loci_to_annotate
        self.features_list = features_list

    def generate_features(
        self: FeatureFactory,
        features_input_loader: L2GFeatureInputLoader,
    ) -> list[L2GFeature]:
        """Generates a feature matrix by reading an object with instructions on how to create the features.

        Args:
            features_input_loader (L2GFeatureInputLoader): object with required features dependencies.

        Returns:
            list[L2GFeature]: list of computed features.

        Raises:
            ValueError: If feature not found.
        """
        computed_features = []
        for feature in self.features_list:
            if feature in self.feature_mapper:
                computed_features.append(
                    self.compute_feature(feature, features_input_loader)
                )
            else:
                raise ValueError(f"Feature {feature} not found.")
        return computed_features

    def compute_feature(
        self: FeatureFactory,
        feature_name: str,
        features_input_loader: L2GFeatureInputLoader,
    ) -> L2GFeature:
        """Instantiates feature class.

        Args:
            feature_name (str): name of the feature
            features_input_loader (L2GFeatureInputLoader): Object that contais features input.

        Returns:
            L2GFeature: instantiated feature object
        """
        # Extract feature class and dependency type
        feature_cls = self.feature_mapper[feature_name]
        feature_dependency_type = feature_cls.feature_dependency_type
        return feature_cls.compute(
            study_loci_to_annotate=self.study_loci_to_annotate,
            feature_dependency=features_input_loader.get_dependency_by_type(
                feature_dependency_type
            ),
        )
