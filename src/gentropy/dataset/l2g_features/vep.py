"""Collection of methods that extract distance features from the variant index dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyspark.sql.functions as f
from pyspark.sql import Window

from gentropy.common.spark_helpers import convert_from_wide_to_long
from gentropy.dataset.gene_index import GeneIndex
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.variant_index import VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def common_vep_feature_logic(
    study_loci_to_annotate: L2GGoldStandard | StudyLocus,
    *,
    variant_index: VariantIndex,
    feature_name: str,
) -> DataFrame:
    """Extracts variant severity score computed from VEP.

    Args:
        study_loci_to_annotate (L2GGoldStandard | StudyLocus): The dataset containing study loci that will be used for annotation
        variant_index (VariantIndex): The dataset containing functional consequence information
        feature_name (str): The name of the feature

    Returns:
        DataFrame: Feature dataset
    """
    # Variant/Target/Severity dataframe
    consequences_dataset = variant_index.get_most_severe_gene_consequence()
    if isinstance(study_loci_to_annotate, StudyLocus):
        variants_df = (
            study_loci_to_annotate.df.withColumn(
                "variantInLocus", f.explode_outer("locus")
            )
            .select(
                "studyLocusId",
                f.col("variantInLocus.variantId").alias("variantId"),
                f.col("variantInLocus.posteriorProbability").alias(
                    "posteriorProbability"
                ),
            )
            .join(consequences_dataset, "variantId")
        )
    elif isinstance(study_loci_to_annotate, L2GGoldStandard):
        variants_df = study_loci_to_annotate.df.select(
            "studyLocusId", "variantId", f.lit(1.0).alias("posteriorProbability")
        ).join(consequences_dataset, "variantId")

    if "Maximum" in feature_name:
        agg_expr = f.max("severityScore")
    elif "Mean" in feature_name:
        variants_df = variants_df.withColumn(
            "weightedScore", f.col("severityScore") * f.col("posteriorProbability")
        )
        agg_expr = f.mean("weightedScore")
    return variants_df.groupBy("studyLocusId", "geneId").agg(
        agg_expr.alias(feature_name)
    )


def common_neighbourhood_vep_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    *,
    variant_index: VariantIndex,
    gene_index: GeneIndex,
    feature_name: str,
) -> DataFrame:
    """Extracts variant severity score computed from VEP for any gene, based on what is the mean score for protein coding genes that are nearby the locus.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
        variant_index (VariantIndex): The dataset containing functional consequence information
        gene_index (GeneIndex): The dataset containing the gene biotype
        feature_name (str): The name of the feature

    Returns:
        DataFrame: Feature dataset
    """
    local_feature_name = feature_name.replace("Neighbourhood", "")
    # First compute mean distances to a gene
    local_metric = common_vep_feature_logic(
        study_loci_to_annotate,
        feature_name=local_feature_name,
        variant_index=variant_index,
    )
    return (
        # Then compute mean distance in the vicinity (feature will be the same for any gene associated with a studyLocus)
        local_metric.join(
            # Bring gene classification
            gene_index.df.select("geneId", "biotype"),
            "geneId",
            "inner",
        )
        .withColumn(
            "regional_metric",
            f.coalesce(
                # Calculate mean based on protein coding genes
                f.mean(
                    f.when(
                        f.col("biotype") == "protein_coding", f.col(local_feature_name)
                    )
                ).over(Window.partitionBy("studyLocusId")),
                # Default to 0 if there are no protein coding genes
                f.lit(0),
            ),
        )
        .withColumn(feature_name, f.col(local_feature_name) - f.col("regional_metric"))
        .drop("regional_metric", local_feature_name, "biotype")
    )


class VepMaximumFeature(L2GFeature):
    """Maximum functional consequence score among all variants in a credible set for a studyLocus/gene."""

    feature_dependency_type = VariantIndex
    feature_name = "vepMaximum"

    @classmethod
    def compute(
        cls: type[VepMaximumFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> VepMaximumFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the functional consequence information

        Returns:
            VepMaximumFeature: Feature dataset
        """
        return cls(
            _df=convert_from_wide_to_long(
                common_vep_feature_logic(
                    study_loci_to_annotate=study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class VepMaximumNeighbourhoodFeature(L2GFeature):
    """Maximum functional consequence score among all variants in a credible set for a studyLocus/gene relative to the mean VEP score across all protein coding genes in the vicinity."""

    feature_dependency_type = [VariantIndex, GeneIndex]
    feature_name = "vepMaximumNeighbourhood"

    @classmethod
    def compute(
        cls: type[VepMaximumNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> VepMaximumNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the functional consequence information

        Returns:
            VepMaximumNeighbourhoodFeature: Feature dataset
        """
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_vep_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class VepMeanFeature(L2GFeature):
    """Average functional consequence score among all variants in a credible set for a studyLocus/gene.

    The mean severity score is weighted by the posterior probability of each variant.
    """

    feature_dependency_type = VariantIndex
    feature_name = "vepMean"

    @classmethod
    def compute(
        cls: type[VepMeanFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> VepMeanFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the functional consequence information

        Returns:
            VepMeanFeature: Feature dataset
        """
        return cls(
            _df=convert_from_wide_to_long(
                common_vep_feature_logic(
                    study_loci_to_annotate=study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class VepMeanNeighbourhoodFeature(L2GFeature):
    """Mean functional consequence score among all variants in a credible set for a studyLocus/gene relative to the mean VEP score across all protein coding genes in the vicinity.

    The mean severity score is weighted by the posterior probability of each variant.
    """

    feature_dependency_type = [VariantIndex, GeneIndex]
    feature_name = "vepMeanNeighbourhood"

    @classmethod
    def compute(
        cls: type[VepMeanNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> VepMeanNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the functional consequence information

        Returns:
            VepMeanNeighbourhoodFeature: Feature dataset
        """
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_vep_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )
