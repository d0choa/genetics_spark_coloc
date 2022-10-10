"""Helper functions to generate the variant index."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as f

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from etl.common.ETLSession import ETLSession

from etl.common.spark_helpers import get_record_with_minimum_value, nullify_empty_array


def join_variants_w_credset(
    etl: ETLSession,
    variant_annotation_path: str,
    credible_sets_path: str,
    partition_count: int,
) -> tuple[DataFrame, DataFrame]:
    """Returns two dataframes: one is the variant index, filtered to contain only the variants present in the credible sets; the second is the variants of the credible set that are not present in the variant annotation.

    Args:
        etl (ETLSession): ETLSession
        variant_annotation_path (str): The path to the variant annotation file
        credible_sets_path (str): the path to the credible sets file
        partition_count (int): The number of partitions to use for the output

    Returns:
        variant_idx (DataFrame): A dataframe with all the variants of interest and their annotation
        fallen_variants (DataFrame): A dataframe with the variants of the credible filtered out of the variant index
    """
    _df = (
        get_variants_from_credset(etl, credible_sets_path)
        .join(
            read_variant_annotation(etl, variant_annotation_path), on="id", how="left"
        )
        .withColumn(
            "variantInGnomad", f.coalesce(f.col("variantInGnomad"), f.lit(False))
        )
    )

    variant_idx = (
        _df.filter(f.col("variantInGnomad"))
        .drop("variantInGnomad")
        .repartition(partition_count)
    )
    invalid_variants = _df.filter(~f.col("variantInGnomad"))

    return variant_idx, invalid_variants


def get_variants_from_credset(etl: ETLSession, credible_sets_path: str) -> DataFrame:
    """It reads the credible sets from the given path, extracts the lead and tag variants.

    Args:
        etl (ETLSession): ETLSession
        credible_sets_path (str): the path to the credible sets

    Returns:
        DataFrame: A dataframe with all variants contained in the credible sets
    """
    credset = etl.spark.read.parquet(credible_sets_path).select(
        "leadVariantId", "tagVariantId"
    )
    return (
        credset.selectExpr("leadVariantId as id")
        .union(credset.selectExpr("tagVariantId as id"))
        .dropDuplicates(["id"])
    )


def read_variant_annotation(etl: ETLSession, variant_annotation_path: str) -> DataFrame:
    """It reads the variant annotation parquet file and formats it to follow the OTG variant model.

    Args:
        etl (ETLSession): ETLSession
        variant_annotation_path (str): path to the variant annotation parquet file

    Returns:
        DataFrame: A dataframe of variants and their annotation
    """
    unchanged_cols = [
        "id",
        "chromosome",
        "position",
        "referenceAllele",
        "alternateAllele",
        "chromosomeB37",
        "positionB37",
        "alleleType",
        "rsIds",
        "alleleFrequencies",
        "cadd",
        "filters",
    ]
    return (
        # TODO: read input with read_parquet providing schema
        etl.spark.read.parquet(variant_annotation_path).select(
            *unchanged_cols,
            # schema of the variant index is the same as the variant annotation
            # except for `vep` which is slimmed and reshaped
            # TODO: convert vep annotation from arr to struct of arrays
            f.struct(
                f.col("vep.most_severe_consequence").alias("mostSevereConsequence"),
                f.col("vep.regulatory_feature_consequences").alias(
                    "regulatoryFeatureConsequences"
                ),
                f.col("vep.motif_feature_consequences").alias(
                    "motifFeatureConsequences"
                ),
                "vep.transcript_consequences.lof",
                f.col("vep.transcript_consequences.lof_flags").alias("lofFlags"),
                f.col("vep.transcript_consequences.lof_filter").alias("lofFilter"),
                f.col("vep.transcript_consequences.lof_info").alias("lofInfo"),
                f.col("vep.transcript_consequences.polyphen_score").alias(
                    "polyphenScore"
                ),
                f.col("vep.transcript_consequences.sift_score").alias("siftScore"),
            ).alias("vep"),
            # filters/rsid are arrays that can be empty, in this case we convert them to null
            nullify_empty_array("filters").alias("filters"),
            nullify_empty_array("rsIds").alias("rsIds"),
            f.lit(True).alias("variantInGnomad"),
        )
    )


def calculate_dist_to_gene(
    gene_idx: DataFrame,
    variant_idx: DataFrame,
    tss_distance_threshold: int,
) -> DataFrame:
    """For each variant, find the closest gene within a certain distance of the variant's position.

    Args:
        gene_idx (DataFrame): filtered genes dataset
        variant_idx (DataFrame): variants dataset
        tss_distance_threshold (int): The maximum distance from the TSS to consider a variant to be near to a gene

    Returns:
        DataFrame: A dataframe with the variantId, geneId, and distance
    """
    return (
        variant_idx.select(f.col("id").alias("variantId"), "chromosome", "position")
        .join(f.broadcast(gene_idx), on="chromosome", how="inner")
        .withColumn("distance", f.abs(f.col("position") - f.col("tss")))
        .filter(f.col("distance") <= tss_distance_threshold)
        .transform(
            lambda df: get_record_with_minimum_value(df, "variantId", "distance")
        )
        .select("variantId", "geneId", "distance")
    )
