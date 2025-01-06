"""Collection of methods that extract distance features from the variant index dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyspark.sql.functions as f
from pyspark.sql import Window

from gentropy.common.spark_helpers import convert_from_wide_to_long
from gentropy.dataset.l2g_features.l2g_feature import L2GFeature
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.target_index import TargetIndex
from gentropy.dataset.variant_index import VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def common_distance_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    *,
    variant_index: VariantIndex,
    feature_name: str,
    distance_type: str,
    genomic_window: int = 500_000,
) -> DataFrame:
    """Calculate the distance feature that correlates a variant in a credible set with a gene.

    The distance is weighted by the posterior probability of the variant to factor in its contribution to the trait when we look at the average distance score for all variants in the credible set.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
        variant_index (VariantIndex): The dataset containing distance to gene information
        feature_name (str): The name of the feature
        distance_type (str): The type of distance to gene
        genomic_window (int): The maximum window size to consider

    Returns:
        DataFrame: Feature dataset
    """
    distances_dataset = variant_index.get_distance_to_gene(distance_type=distance_type)
    if "Mean" in feature_name:
        # Weighting by the SNP contribution is only applied when we are averaging all distances
        df = study_loci_to_annotate.df.withColumn(
            "variantInLocus", f.explode_outer("locus")
        ).select(
            "studyLocusId",
            f.col("variantInLocus.variantId").alias("variantId"),
            f.col("variantInLocus.posteriorProbability").alias("posteriorProbability"),
        )
        distance_score_expr = (
            f.lit(genomic_window) - f.col(distance_type) + f.lit(1)
        ) * f.col("posteriorProbability")
        agg_expr = f.sum(f.col("distance_score"))
    elif "Sentinel" in feature_name:
        df = study_loci_to_annotate.df.select("studyLocusId", "variantId")
        # For minimum distances we calculate the unweighted distance between the sentinel (lead) and the gene.
        distance_score_expr = f.lit(genomic_window) - f.col(distance_type) + f.lit(1)
        agg_expr = f.first(f.col("distance_score"))
    return (
        df.join(
            distances_dataset.withColumnRenamed("targetId", "geneId"),
            on="variantId",
            how="inner",
        )
        .withColumn(
            "distance_score",
            distance_score_expr,
        )
        .groupBy("studyLocusId", "geneId")
        .agg(agg_expr.alias("distance_score_agg"))
        .withColumn(
            feature_name,
            f.log10(f.col("distance_score_agg")) / f.log10(f.lit(genomic_window + 1)),
        )
        .drop("distance_score_agg")
    )


def common_neighbourhood_distance_feature_logic(
    study_loci_to_annotate: StudyLocus | L2GGoldStandard,
    *,
    variant_index: VariantIndex,
    feature_name: str,
    distance_type: str,
    target_index: TargetIndex,
    genomic_window: int = 500_000,
) -> DataFrame:
    """Calculate the distance feature that correlates any variant in a credible set with any protein coding gene nearby the locus. The distance is weighted by the posterior probability of the variant to factor in its contribution to the trait.

    Args:
        study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
        variant_index (VariantIndex): The dataset containing distance to gene information
        feature_name (str): The name of the feature
        distance_type (str): The type of distance to gene
        target_index (TargetIndex): The dataset containing gene information
        genomic_window (int): The maximum window size to consider

    Returns:
        DataFrame: Feature dataset
    """
    local_feature_name = feature_name.replace("Neighbourhood", "")
    # First compute mean distances to a gene
    local_metric = common_distance_feature_logic(
        study_loci_to_annotate,
        feature_name=local_feature_name,
        distance_type=distance_type,
        variant_index=variant_index,
        genomic_window=genomic_window,
    )
    return (
        # Then compute mean distance in the vicinity (feature will be the same for any gene associated with a studyLocus)
        local_metric.join(
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
        .withColumn(
            feature_name,
            f.when(f.col(feature_name) < 0, f.lit(0.0))
            .when(f.col(feature_name) > 1, f.lit(1.0))
            .otherwise(f.col(feature_name)),
        )
        .drop("regional_max", local_feature_name)
    )


class DistanceTssMeanFeature(L2GFeature):
    """Average distance of all tagging variants to gene TSS."""

    feature_dependency_type = VariantIndex
    feature_name = "distanceTssMean"

    @classmethod
    def compute(
        cls: type[DistanceTssMeanFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceTssMeanFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceTssMeanFeature: Feature dataset
        """
        distance_type = "distanceFromTss"
        return cls(
            _df=convert_from_wide_to_long(
                common_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ).withColumn(
                    cls.feature_name,
                    f.when(f.col(cls.feature_name) < 0, f.lit(0.0)).otherwise(
                        f.col(cls.feature_name)
                    ),
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceTssMeanNeighbourhoodFeature(L2GFeature):
    """Minimum mean distance to TSS for all genes in the vicinity of a studyLocus."""

    feature_dependency_type = [VariantIndex, TargetIndex]
    feature_name = "distanceTssMeanNeighbourhood"

    @classmethod
    def compute(
        cls: type[DistanceTssMeanNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceTssMeanNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceTssMeanNeighbourhoodFeature: Feature dataset
        """
        distance_type = "distanceFromTss"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceSentinelTssFeature(L2GFeature):
    """Distance of the sentinel variant to gene TSS. This is not weighted by the causal probability."""

    feature_dependency_type = VariantIndex
    feature_name = "distanceSentinelTss"

    @classmethod
    def compute(
        cls: type[DistanceSentinelTssFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceSentinelTssFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceSentinelTssFeature: Feature dataset
        """
        distance_type = "distanceFromTss"
        return cls(
            _df=convert_from_wide_to_long(
                common_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceSentinelTssNeighbourhoodFeature(L2GFeature):
    """Distance between the sentinel variant and a gene TSS as a relation of the distnace with all the genes in the vicinity of a studyLocus. This is not weighted by the causal probability."""

    feature_dependency_type = [VariantIndex, TargetIndex]
    feature_name = "distanceSentinelTssNeighbourhood"

    @classmethod
    def compute(
        cls: type[DistanceSentinelTssNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceSentinelTssNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceSentinelTssNeighbourhoodFeature: Feature dataset
        """
        distance_type = "distanceFromTss"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceFootprintMeanFeature(L2GFeature):
    """Average distance of all tagging variants to the footprint of a gene."""

    feature_dependency_type = VariantIndex
    feature_name = "distanceFootprintMean"

    @classmethod
    def compute(
        cls: type[DistanceFootprintMeanFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceFootprintMeanFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceFootprintMeanFeature: Feature dataset
        """
        distance_type = "distanceFromFootprint"
        return cls(
            _df=convert_from_wide_to_long(
                common_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ).withColumn(
                    cls.feature_name,
                    f.when(f.col(cls.feature_name) < 0, f.lit(0.0)).otherwise(
                        f.col(cls.feature_name)
                    ),
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceFootprintMeanNeighbourhoodFeature(L2GFeature):
    """Minimum mean distance to footprint for all genes in the vicinity of a studyLocus."""

    feature_dependency_type = [VariantIndex, TargetIndex]
    feature_name = "distanceFootprintMeanNeighbourhood"

    @classmethod
    def compute(
        cls: type[DistanceFootprintMeanNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceFootprintMeanNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceFootprintMeanNeighbourhoodFeature: Feature dataset
        """
        distance_type = "distanceFromFootprint"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceSentinelFootprintFeature(L2GFeature):
    """Distance between the sentinel variant and the footprint of a gene."""

    feature_dependency_type = VariantIndex
    feature_name = "distanceSentinelFootprint"

    @classmethod
    def compute(
        cls: type[DistanceSentinelFootprintFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceSentinelFootprintFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceSentinelFootprintFeature: Feature dataset
        """
        distance_type = "distanceFromFootprint"
        return cls(
            _df=convert_from_wide_to_long(
                common_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )


class DistanceSentinelFootprintNeighbourhoodFeature(L2GFeature):
    """Distance between the sentinel variant and a gene footprint as a relation of the distnace with all the genes in the vicinity of a studyLocus. This is not weighted by the causal probability."""

    feature_dependency_type = [VariantIndex, TargetIndex]
    feature_name = "distanceSentinelFootprintNeighbourhood"

    @classmethod
    def compute(
        cls: type[DistanceSentinelFootprintNeighbourhoodFeature],
        study_loci_to_annotate: StudyLocus | L2GGoldStandard,
        feature_dependency: dict[str, Any],
    ) -> DistanceSentinelFootprintNeighbourhoodFeature:
        """Computes the feature.

        Args:
            study_loci_to_annotate (StudyLocus | L2GGoldStandard): The dataset containing study loci that will be used for annotation
            feature_dependency (dict[str, Any]): Dataset that contains the distance information

        Returns:
            DistanceSentinelFootprintNeighbourhoodFeature: Feature dataset
        """
        distance_type = "distanceFromFootprint"
        return cls(
            _df=convert_from_wide_to_long(
                common_neighbourhood_distance_feature_logic(
                    study_loci_to_annotate,
                    feature_name=cls.feature_name,
                    distance_type=distance_type,
                    **feature_dependency,
                ),
                id_vars=("studyLocusId", "geneId"),
                var_name="featureName",
                value_name="featureValue",
            ),
            _schema=cls.get_schema(),
        )
