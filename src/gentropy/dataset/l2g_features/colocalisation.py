"""Collection of methods that extract features from the colocalisation datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyspark.sql.functions as f
from pyspark.sql import Window

from gentropy.common.spark_helpers import convert_from_wide_to_long
from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.target_index import TargetIndex
from gentropy.dataset.variant_index import VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def common_colocalisation_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    colocalisation_method: str,
    colocalisation_metric: str,
    feature_name: str,
    qtl_types: list[str] | str,
    *,
    colocalisation: Colocalisation,
    study_index: StudyIndex,
    study_locus: StudyLocus,
) -> DataFrame:
    """Wrapper to call the logic that creates a type of colocalisation features.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
        colocalisation_method (str): The colocalisation method to filter the data by
        colocalisation_metric (str): The colocalisation metric to use
        feature_name (str): The name of the feature to create
        qtl_types (list[str] | str): The types of QTL to filter the data by
        colocalisation (Colocalisation): Dataset with the colocalisation results
        study_index (StudyIndex): Study index to fetch study type and gene
        study_locus (StudyLocus): Study locus to traverse between colocalisation and study index

    Returns:
        DataFrame: Feature annotation in long format with the columns: studyLocusId, geneId, featureName, featureValue
    """
    joining_cols = (
        ["studyLocusId", "geneId"]
        if isinstance(study_loci_to_annotate, L2GGoldStandard)
        else ["studyLocusId"]
    )
    return (
        study_loci_to_annotate.df.join(
            colocalisation.extract_maximum_coloc_probability_per_region_and_gene(
                study_locus,
                study_index,
                filter_by_colocalisation_method=colocalisation_method,
                filter_by_qtls=qtl_types,
            ),
            on=joining_cols,
        )
        .selectExpr(
            "studyLocusId",
            "geneId",
            f"{colocalisation_metric} as {feature_name}",
        )
        .distinct()
    )


def extend_missing_colocalisation_to_neighbourhood_genes(
    feature_name: str,
    local_features: DataFrame,
    variant_index: VariantIndex,
    target_index: TargetIndex,
    study_locus: StudyLocus,
) -> DataFrame:
    """This function creates an artificial dataset of features that represents the missing colocalisation to the neighbourhood genes.

    Args:
        feature_name (str): The name of the feature to extend
        local_features (DataFrame): The dataframe of features to extend
        variant_index (VariantIndex): Variant index containing all variant/gene relationships
        target_index (TargetIndex): Target index to fetch the gene information
        study_locus (StudyLocus): Study locus to traverse between colocalisation and variant index

    Returns:
        DataFrame: Dataframe of features that include genes in the neighbourhood not present in the colocalisation results. For these genes, the feature value is set to 0.
    """
    coding_variant_gene_lut = (
        variant_index.df.select(
            "variantId", f.explode("transcriptConsequences").alias("tc")
        )
        .select(f.col("tc.targetId").alias("geneId"), "variantId")
        .join(target_index.df.select("geneId", "biotype"), "geneId", "left")
        .filter(f.col("biotype") == "protein_coding")
        .drop("biotype")
        .distinct()
    )
    local_features_w_variant = local_features.join(
        study_locus.df.select("studyLocusId", "variantId"), "studyLocusId"
    )
    return (
        # Get the genes that are not present in the colocalisation results
        coding_variant_gene_lut.join(
            local_features_w_variant, ["variantId", "geneId"], "left_anti"
        )
        # We now link the missing variant/gene to the study locus from the original dataframe
        .join(
            local_features_w_variant.select("studyLocusId", "variantId").distinct(),
            "variantId",
        )
        .drop("variantId")
        # Fill the information for missing genes with 0
        .withColumn(feature_name, f.lit(0.0))
    )


def common_neighbourhood_colocalisation_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    colocalisation_method: str,
    colocalisation_metric: str,
    feature_name: str,
    qtl_types: list[str] | str,
    *,
    colocalisation: Colocalisation,
    study_index: StudyIndex,
    target_index: TargetIndex,
    study_locus: StudyLocus,
    variant_index: VariantIndex,
) -> DataFrame:
    """Wrapper to call the logic that creates a type of colocalisation features.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
        colocalisation_method (str): The colocalisation method to filter the data by
        colocalisation_metric (str): The colocalisation metric to use
        feature_name (str): The name of the feature to create
        qtl_types (list[str] | str): The types of QTL to filter the data by
        colocalisation (Colocalisation): Dataset with the colocalisation results
        study_index (StudyIndex): Study index to fetch study type and gene
        target_index (TargetIndex): Target index to add gene type
        study_locus (StudyLocus): Study locus to traverse between colocalisation and study index
        variant_index (VariantIndex): Variant index to annotate all overlapping genes

    Returns:
        DataFrame: Feature annotation in long format with the columns: studyLocusId, geneId, featureName, featureValue
    """
    # First maximum colocalisation score for each studylocus, gene
    local_feature_name = feature_name.replace("Neighbourhood", "")
    local_max = common_colocalisation_feature_logic(
        study_loci_to_annotate,
        colocalisation_method,
        colocalisation_metric,
        local_feature_name,
        qtl_types,
        colocalisation=colocalisation,
        study_index=study_index,
        study_locus=study_locus,
    )
    extended_local_max = local_max.unionByName(
        extend_missing_colocalisation_to_neighbourhood_genes(
            local_feature_name,
            local_max,
            variant_index,
            target_index,
            study_locus,
        )
    )
    return (
        extended_local_max.join(
            # Compute average score in the vicinity (feature will be the same for any gene associated with a studyLocus)
            # (non protein coding genes in the vicinity are excluded see #3552)
            target_index.df.filter(f.col("biotype") == "protein_coding").select("geneId"),
            "geneId",
            "inner",
        )
        .withColumn(
            "regional_max",
            f.max(local_feature_name).over(Window.partitionBy("studyLocusId")),
        )
        .withColumn(
            feature_name,
            f.when(
                (f.col("regional_max").isNotNull()) & (f.col("regional_max") != 0.0),
                f.col(local_feature_name)
                / f.coalesce(f.col("regional_max"), f.lit(0.0)),
            ).otherwise(f.lit(0.0)),
        )
        .drop("regional_max", local_feature_name)
    )


class EQtlColocClppMaximumFeature(L2GFeature):
    """Max CLPP for each (study, locus, gene) aggregating over all eQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "eQtlColocClppMaximum"

    @classmethod
    def compute(
        cls: type[EQtlColocClppMaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> EQtlColocClppMaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dictionary with the dependencies required. They are passed as keyword arguments.

        Returns:
            EQtlColocClppMaximumFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_type = ["eqtl", "sceqtl"]

        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class EQtlColocClppMaximumNeighbourhoodFeature(L2GFeature):
    """Max CLPP for each (study, locus) aggregating over all eQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "eQtlColocClppMaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[EQtlColocClppMaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> EQtlColocClppMaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dictionary with the dependencies required. They are passed as keyword arguments.

        Returns:
            EQtlColocClppMaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_type = ["eqtl", "sceqtl"]

        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class PQtlColocClppMaximumFeature(L2GFeature):
    """Max CLPP for each (study, locus, gene) aggregating over all pQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "pQtlColocClppMaximum"

    @classmethod
    def compute(
        cls: type[PQtlColocClppMaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> PQtlColocClppMaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            PQtlColocClppMaximumFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_type = "pqtl"
        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class PQtlColocClppMaximumNeighbourhoodFeature(L2GFeature):
    """Max CLPP for each (study, locus, gene) aggregating over all pQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "pQtlColocClppMaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[PQtlColocClppMaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> PQtlColocClppMaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            PQtlColocClppMaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_type = "pqtl"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class SQtlColocClppMaximumFeature(L2GFeature):
    """Max CLPP for each (study, locus, gene) aggregating over all sQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "sQtlColocClppMaximum"

    @classmethod
    def compute(
        cls: type[SQtlColocClppMaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> SQtlColocClppMaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            SQtlColocClppMaximumFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_types = ["sqtl", "tuqtl", "scsqtl", "sctuqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_types,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class SQtlColocClppMaximumNeighbourhoodFeature(L2GFeature):
    """Max CLPP for each (study, locus, gene) aggregating over all sQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "sQtlColocClppMaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[SQtlColocClppMaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> SQtlColocClppMaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            SQtlColocClppMaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "ECaviar"
        colocalisation_metric = "clpp"
        qtl_types = ["sqtl", "tuqtl", "scsqtl", "sctuqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_types,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class EQtlColocH4MaximumFeature(L2GFeature):
    """Max H4 for each (study, locus, gene) aggregating over all eQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "eQtlColocH4Maximum"

    @classmethod
    def compute(
        cls: type[EQtlColocH4MaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> EQtlColocH4MaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            EQtlColocH4MaximumFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_type = ["eqtl", "sceqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class EQtlColocH4MaximumNeighbourhoodFeature(L2GFeature):
    """Max H4 for each (study, locus) aggregating over all eQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "eQtlColocH4MaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[EQtlColocH4MaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> EQtlColocH4MaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            EQtlColocH4MaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_type = ["eqtl", "sceqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class PQtlColocH4MaximumFeature(L2GFeature):
    """Max H4 for each (study, locus, gene) aggregating over all pQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "pQtlColocH4Maximum"

    @classmethod
    def compute(
        cls: type[PQtlColocH4MaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> PQtlColocH4MaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            PQtlColocH4MaximumFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_type = "pqtl"
        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class PQtlColocH4MaximumNeighbourhoodFeature(L2GFeature):
    """Max H4 for each (study, locus) aggregating over all pQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "pQtlColocH4MaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[PQtlColocH4MaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> PQtlColocH4MaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            PQtlColocH4MaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_type = "pqtl"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class SQtlColocH4MaximumFeature(L2GFeature):
    """Max H4 for each (study, locus, gene) aggregating over all sQTLs."""

    feature_dependency_type = [Colocalisation, StudyIndex, StudyLocus]
    feature_name = "sQtlColocH4Maximum"

    @classmethod
    def compute(
        cls: type[SQtlColocH4MaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> SQtlColocH4MaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            SQtlColocH4MaximumFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_types = ["sqtl", "tuqtl", "scsqtl", "sctuqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_types,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class SQtlColocH4MaximumNeighbourhoodFeature(L2GFeature):
    """Max H4 for each (study, locus) aggregating over all sQTLs."""

    feature_dependency_type = [
        Colocalisation,
        StudyIndex,
        TargetIndex,
        StudyLocus,
        VariantIndex,
    ]
    feature_name = "sQtlColocH4MaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[SQtlColocH4MaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> SQtlColocH4MaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset with the colocalisation results

        Returns:
            SQtlColocH4MaximumNeighbourhoodFeature: Feature dataset
        """
        colocalisation_method = "Coloc"
        colocalisation_metric = "h4"
        qtl_types = ["sqtl", "tuqtl", "scsqtl", "sctuqtl"]
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_colocalisation_feature_logic(
                    study_loci_to_annotate,
                    colocalisation_method,
                    colocalisation_metric,
                    cls.feature_name,
                    qtl_types,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )
