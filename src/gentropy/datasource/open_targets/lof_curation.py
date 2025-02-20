"""Parser for Loss-of-Function variant data from Open Targets Project OTAR2075."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from gentropy.common.spark_helpers import enforce_schema
from gentropy.dataset.variant_index import VariantEffectNormaliser, VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


class OpenTargetsLOF:
    """Class to parse Loss-of-Function variant data from Open Targets Project OTAR2075."""

    VARIANT_EFFECT_SCHEMA = VariantIndex.get_schema()[
        "variantEffect"
    ].dataType.elementType

    @staticmethod
    @enforce_schema(VARIANT_EFFECT_SCHEMA)
    def _get_lof_assessment(verdict: Column) -> Column:
        """Get curated Loss-of-Function assessment from verdict column.

        Args:
            verdict (Column): verdict column from the input dataset.

        Returns:
            Column: struct following the variant effect schema.
        """
        return f.struct(
            f.lit("LossOfFunctionCuration").alias("method"),
            verdict.alias("assessment"),
        )

    @staticmethod
    def _compose_lof_description(verdict: Column) -> Column:
        """Compose variant description based on loss-of-function assessment.

        Args:
            verdict (Column): verdict column from the input dataset.

        Returns:
            Column: variant description.
        """
        lof_description = (
            f.when(verdict == "lof", f.lit("Assessed to cause LoF"))
            .when(verdict == "likely_lof", f.lit("Suspected to cause LoF"))
            .when(verdict == "uncertain", f.lit("Uncertain LoF assessment"))
            .when(verdict == "likely_not_lof", f.lit("Suspected not to cause LoF"))
            .when(verdict == "not_lof", f.lit("Assessed not to cause LoF"))
        )

        return f.concat(lof_description, f.lit(" by OTAR2075 variant curation effort."))

    @classmethod
    def as_variant_index(
        cls: type[OpenTargetsLOF],
        lof_dataset: DataFrame
    ) -> VariantIndex:
        """Ingest Loss-of-Function information as a VariantIndex object.

        Args:
            lof_dataset (DataFrame): curated input dataset from OTAR2075.

        Returns:
            VariantIndex: variant annotations with loss-of-function assessments.
        """
        return VariantIndex(
            _df=(
                lof_dataset
                .select(
                    f.from_csv(f.col("Variant ID GRCh37"), "chr string, pos string, ref string, alt string", {"sep": "-"}).alias("h37"),
                    f.from_csv(f.col("Variant ID GRCh38"), "chr string, pos string, ref string, alt string", {"sep": "-"}).alias("h38"),
                    "Verdict"
                )
                .select(
                    # As some GRCh37 variants do not correctly lift over to the correct GRCh38 variant,
                    # chr_pos is taken from the GRCh38 variant id, and ref_alt from the GRCh37 variant id
                    f.concat_ws("_", f.col("h38.chr"), f.col("h38.pos"), f.col("h37.ref"), f.col("h37.alt")).alias("variantId"),
                    # Mandatory fields for VariantIndex:
                    f.col("h38.chr").alias("chromosome"),
                    f.col("h38.pos").cast(t.IntegerType()).alias("position"),
                    f.col("h37.ref").alias("referenceAllele"),
                    f.col("h37.alt").alias("alternateAllele"),
                    # Populate variantEffect and variantDescription fields:
                    f.array(cls._get_lof_assessment(f.col("Verdict"))).alias("variantEffect"),
                    cls._compose_lof_description(f.col("Verdict")).alias("variantDescription"),
                )
                # Convert assessments to normalised scores:
                .withColumn("variantEffect", VariantEffectNormaliser.normalise_variant_effect(f.col("variantEffect")))
            ),
            _schema=VariantIndex.get_schema(),
        )
