"""Dataset definition for variant index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window

from gentropy.common.schemas import parse_spark_schema
from gentropy.common.spark_helpers import (
    get_nested_struct_schema,
    rename_all_columns,
    safe_array_union,
)
from gentropy.dataset.amino_acid_variants import AminoAcidVariants
from gentropy.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.types import StructType


@dataclass
class VariantIndex(Dataset):
    """Dataset for representing variants and methods applied on them."""

    id_threshold: int = field(default=300)

    def __post_init__(self: VariantIndex) -> None:
        """Forcing the presence of empty arrays even if the schema allows missing values.

        To bring in annotations from other sources, we use the `array_union()` function. However it assumes
        both columns have arrays (not just the array schema!). If one of the array is null, the union
        is nullified. This needs to be avoided.
        """
        # Calling dataset's post init to validate schema:
        super().__post_init__()

        # Composing a list of expressions to replace nulls with empty arrays if the schema assumes:
        array_columns = {
            column.name: f.when(f.col(column.name).isNull(), f.array()).otherwise(
                f.col(column.name)
            )
            for column in self.df.schema
            if "ArrayType" in column.dataType.__str__()
        }

        # Not returning, but changing the data:
        self.df = self.df.withColumns(array_columns).withColumn(
            # Hashing long variant identifiers:
            "variantId",
            self.hash_long_variant_ids(
                f.col("variantId"),
                f.col("chromosome"),
                f.col("position"),
                self.id_threshold,
            ),
        )

    @classmethod
    def get_schema(cls: type[VariantIndex]) -> StructType:
        """Provides the schema for the variant index dataset.

        Returns:
            StructType: Schema for the VariantIndex dataset
        """
        return parse_spark_schema("variant_index.json")

    @staticmethod
    def hash_long_variant_ids(
        variant_id: Column, chromosome: Column, position: Column, threshold: int
    ) -> Column:
        """Hash long variant identifiers.

        Args:
            variant_id (Column): Column containing variant identifiers.
            chromosome (Column): Chromosome column.
            position (Column): position column.
            threshold (int): Above this limit, a hash will be generated.

        Returns:
            Column: Hashed variant identifiers for long variants.

        Examples:
            >>> (
            ...    spark.createDataFrame([('v_short', 'x', 23),('v_looooooong', '23', 23), ('no_chrom', None, None), (None, None, None)], ['variantId', 'chromosome', 'position'])
            ...    .select('variantId', VariantIndex.hash_long_variant_ids(f.col('variantId'), f.col('chromosome'), f.col('position'), 10).alias('hashedVariantId'))
            ...    .show(truncate=False)
            ... )
            +------------+--------------------------------------------+
            |variantId   |hashedVariantId                             |
            +------------+--------------------------------------------+
            |v_short     |v_short                                     |
            |v_looooooong|OTVAR_23_23_3749d019d645894770c364992ae70a05|
            |no_chrom    |OTVAR_41acfcd7d4fd523b33600b504914ef25      |
            |NULL        |NULL                                        |
            +------------+--------------------------------------------+
            <BLANKLINE>
        """
        return (
            # If either the position or the chromosome is missing, we hash the identifier:
            f.when(
                chromosome.isNull() | position.isNull(),
                f.concat(
                    f.lit("OTVAR_"),
                    f.md5(variant_id).cast("string"),
                ),
            )
            # If chromosome and position are given, but alleles are too long, create hash:
            .when(
                f.length(variant_id) >= threshold,
                f.concat_ws(
                    "_",
                    f.lit("OTVAR"),
                    chromosome,
                    position,
                    f.md5(variant_id).cast("string"),
                ),
            )
            # Missing and regular variant identifiers are left unchanged:
            .otherwise(variant_id)
        )

    def add_annotation(
        self: VariantIndex, annotation_source: VariantIndex
    ) -> VariantIndex:
        """Import annotation from an other variant index dataset.

        At this point the annotation can be extended with extra cross-references,
        variant effects, allele frequencies, and variant descriptions.

        Args:
            annotation_source (VariantIndex): Annotation to add to the dataset

        Returns:
            VariantIndex: VariantIndex dataset with the annotation added
        """
        # Prefix for renaming columns:
        prefix = "annotation_"

        # Generate select expressions that to merge and import columns from annotation:
        select_expressions = []

        # Collect columns by iterating over the variant index schema:
        for schema_field in VariantIndex.get_schema():
            column = schema_field.name

            # If an annotation column can be found in both datasets:
            if (column in self.df.columns) and (column in annotation_source.df.columns):
                # Arrays are merged:
                if isinstance(schema_field.dataType, t.ArrayType):
                    fields_order = None
                    if isinstance(schema_field.dataType.elementType, t.StructType):
                        # Extract the schema of the array to get the order of the fields:
                        array_schema = [
                            schema_field
                            for schema_field in VariantIndex.get_schema().fields
                            if schema_field.name == column
                        ][0].dataType
                        fields_order = get_nested_struct_schema(
                            array_schema
                        ).fieldNames()
                    select_expressions.append(
                        safe_array_union(
                            f.col(column), f.col(f"{prefix}{column}"), fields_order
                        ).alias(column)
                    )
                # variantDescription columns are concatenated:
                elif column == "variantDescription":
                    select_expressions.append(
                        f.concat_ws(" ", f.col(column), f.col(f"{prefix}{column}")).alias(column)
                    )
                # All other non-array columns are coalesced:
                else:
                    select_expressions.append(
                        f.coalesce(f.col(column), f.col(f"{prefix}{column}")).alias(
                            column
                        )
                    )
            # If the column is only found in the annotation dataset rename it:
            elif column in annotation_source.df.columns:
                select_expressions.append(f.col(f"{prefix}{column}").alias(column))
            # If the column is only found in the main dataset:
            elif column in self.df.columns:
                select_expressions.append(f.col(column))
            # VariantIndex columns not found in either dataset are ignored.

        # Join the annotation to the dataset:
        return VariantIndex(
            _df=(
                f.broadcast(self.df)
                .join(
                    rename_all_columns(annotation_source.df, prefix),
                    on=[f.col("variantId") == f.col(f"{prefix}variantId")],
                    how="left",
                )
                .select(*select_expressions)
            ),
            _schema=self.schema,
        )

    def max_maf(self: VariantIndex) -> Column:
        """Maximum minor allele frequency accross all populations assuming all variants biallelic.

        Returns:
            Column: Maximum minor allele frequency accross all populations.

        Raises:
            ValueError: Allele frequencies are not present in the dataset.
        """
        if "alleleFrequencies" not in self.df.columns:
            raise ValueError("Allele frequencies are not present in the dataset.")

        return f.array_max(
            f.transform(
                self.df.alleleFrequencies,
                lambda af: f.when(
                    af.alleleFrequency > 0.5, 1 - af.alleleFrequency
                ).otherwise(af.alleleFrequency),
            )
        )

    def filter_by_variant(self: VariantIndex, df: DataFrame) -> VariantIndex:
        """Filter variant annotation dataset by a variant dataframe.

        Args:
            df (DataFrame): A dataframe of variants.

        Returns:
            VariantIndex: A filtered variant annotation dataset.

        Raises:
            AssertionError: When the variant dataframe does not contain eiter `variantId` or `chromosome` column.
        """
        join_columns = ["variantId", "chromosome"]

        assert all(
            col in df.columns for col in join_columns
        ), "The variant dataframe must contain the columns 'variantId' and 'chromosome'."

        return VariantIndex(
            _df=self._df.join(
                f.broadcast(df.select(*join_columns).distinct()),
                on=join_columns,
                how="inner",
            ),
            _schema=self.schema,
        )

    def get_distance_to_gene(
        self: VariantIndex,
        *,
        distance_type: str = "distanceFromTss",
        max_distance: int = 500_000,
    ) -> DataFrame:
        """Extracts variant to gene assignments for variants falling within a window of a gene's TSS or footprint.

        Args:
            distance_type (str): The type of distance to use. Can be "distanceFromTss" or "distanceFromFootprint". Defaults to "distanceFromTss".
            max_distance (int): The maximum distance to consider. Defaults to 500_000, the default window size for VEP.

        Returns:
            DataFrame: A dataframe with the distance between a variant and a gene's TSS or footprint.

        Raises:
            ValueError: Invalid distance type.
        """
        if distance_type not in {"distanceFromTss", "distanceFromFootprint"}:
            raise ValueError(
                f"Invalid distance_type: {distance_type}. Must be 'distanceFromTss' or 'distanceFromFootprint'."
            )
        df = self.df.select(
            "variantId", f.explode("transcriptConsequences").alias("tc")
        ).select("variantId", "tc.targetId", f"tc.{distance_type}")
        if max_distance == 500_000:
            return df
        elif max_distance < 500_000:
            return df.filter(f"{distance_type} <= {max_distance}")
        else:
            raise ValueError(
                f"max_distance must be less than 500_000. Got {max_distance}."
            )

    def annotate_with_amino_acid_consequences(
        self: VariantIndex, annotation: AminoAcidVariants
    ) -> VariantIndex:
        """Enriching variant effect assessments with amino-acid derived predicted consequences.

        Args:
            annotation (AminoAcidVariants): amio-acid level variant consequences.

        Returns:
            VariantIndex: where amino-acid causing variants are enriched with extra annotation
        """
        w = Window.partitionBy("variantId").orderBy(f.size("variantEffect").desc())

        return VariantIndex(
            _df=self.df
            # Extracting variant consequence on Uniprot and amino-acid changes from the transcripts:
            .withColumns(
                {
                    "aminoAcidChange": f.filter(
                        "transcriptConsequences",
                        lambda vep: vep.aminoAcidChange.isNotNull(),
                    )[0].aminoAcidChange,
                    "uniprotAccession": f.explode_outer(
                        f.filter(
                            "transcriptConsequences",
                            lambda vep: vep.aminoAcidChange.isNotNull(),
                        )[0].uniprotAccessions
                    ),
                }
            )
            # Joining with amino-acid predictions:
            .join(
                annotation.df.withColumnRenamed("variantEffect", "annotations"),
                on=["uniprotAccession", "aminoAcidChange"],
                how="left",
            )
            # Merge predictors:
            .withColumn(
                "variantEffect",
                f.when(
                    f.col("annotations").isNotNull(),
                    f.array_union("variantEffect", "annotations"),
                ).otherwise(f.col("variantEffect")),
            )
            # Dropping unused columns:
            .drop("uniprotAccession", "aminoAcidChange", "annotations")
            # Dropping potentially exploded variant rows:
            .distinct()
            .withColumn("rank", f.row_number().over(w))
            .filter(f.col("rank") == 1)
            .drop("rank"),
            _schema=self.get_schema(),
        )

    def get_loftee(self: VariantIndex) -> DataFrame:
        """Returns a dataframe with a flag indicating whether a variant is predicted to cause loss of function in a gene. The source of this information is the LOFTEE algorithm (https://github.com/konradjk/loftee).

        !!! note, "This will return a filtered dataframe with only variants that have been annotated by LOFTEE."

        Returns:
            DataFrame: variant to gene assignments from the LOFTEE algorithm
        """
        return (
            self.df.select("variantId", f.explode("transcriptConsequences").alias("tc"))
            .filter(f.col("tc.lofteePrediction").isNotNull())
            .withColumn(
                "isHighQualityPlof",
                f.when(f.col("tc.lofteePrediction") == "HC", True).when(
                    f.col("tc.lofteePrediction") == "LC", False
                ),
            )
            .select(
                "variantId",
                f.col("tc.targetId"),
                f.col("tc.lofteePrediction"),
                "isHighQualityPlof",
            )
        )


class VariantEffectNormaliser:
    """Class to normalise variant effect assessments.

    Essentially based on the raw scores, it normalises the scores to a range between -1 and 1, and appends the normalised
    value to the variant effect struct.

    The higher negative values indicate increasingly confident prediction to be a benign variant,
    while the higher positive values indicate increasingly deleterious predicted effect.

    The point of these operations to make the scores comparable across different variant effect assessments.
    """

    @classmethod
    def normalise_variant_effect(
        cls: type[VariantEffectNormaliser],
        variant_effect: Column,
    ) -> Column:
        """Normalise variant effect assessments. Appends a normalised score to the variant effect struct.

        Args:
            variant_effect (Column): Column containing variant effect assessments (list of structs).

        Returns:
            Column: Normalised variant effect assessments.
        """
        return f.transform(
            variant_effect,
            lambda predictor: f.struct(
                # Extracing all existing columns:
                predictor.method.alias("method"),
                predictor.assessment.alias("assessment"),
                predictor.score.alias("score"),
                predictor.assessmentFlag.alias("assessmentFlag"),
                predictor.targetId.alias("targetId"),
                # Normalising the score
                cls.resolve_predictor_methods(
                    predictor.score, predictor.method, predictor.assessment
                ).alias("normalisedScore"),
            ),
        )

    @classmethod
    def resolve_predictor_methods(
        cls: type[VariantEffectNormaliser],
        score: Column,
        method: Column,
        assessment: Column,
    ) -> Column:
        """It takes a score, a method, and an assessment, and returns a normalized score for the variant effect.

        Args:
            score (Column): The raw score from the variant effect.
            method (Column): The method used to generate the score.
            assessment (Column): The assessment of the score.

        Returns:
            Column: Normalised score for the variant effect.
        """
        return (
            f.when(method == "LOFTEE", cls._normalise_loftee(assessment))
            .when(method == "SIFT", cls._normalise_sift(score, assessment))
            .when(method == "PolyPhen", cls._normalise_polyphen(assessment, score))
            .when(method == "AlphaMissense", cls._normalise_alpha_missense(score))
            .when(method == "CADD", cls._normalise_cadd(score))
            .when(method == "Pangolin", cls._normalise_pangolin(score))
            .when(method == "LossOfFunctionCuration", cls._normalise_lof(assessment))
            # The following predictors are not normalised:
            .when(method == "SpliceAI", score)
            .when(method == "VEP", score)
            .when(method == "GERP", cls._normalise_gerp(score))
            .when(method == "FoldX", cls._normalise_foldx(score))
        )

    @staticmethod
    def _rescaleColumnValue(
        column: Column,
        min_value: float,
        max_value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> Column:
        """Rescale a column to a new range. Similar to MinMaxScaler in pyspark ML.

        Args:
            column (Column): Column to rescale.
            min_value (float): Minimum value of the column.
            max_value (float): Maximum value of the column.
            minimum (float, optional): Minimum value of the new range. Defaults to 0.0.
            maximum (float, optional): Maximum value of the new range. Defaults to 1.0.

        Returns:
            Column: Rescaled column.
        """
        return (column - min_value) / (max_value - min_value) * (
            maximum - minimum
        ) + minimum

    @classmethod
    def _normalise_foldx(
        cls: type[VariantEffectNormaliser], score: Column
    ) -> Column:
        """Normalise FoldX ddG energies.

        ΔΔG Range:
        - 0 to ±0.5 kcal/mol: Usually considered negligible or within the noise of predictions. The mutation has minimal or no effect on stability.
        - ±0.5 to ±1.5 kcal/mol: Moderate effect on stability. Such mutations might cause noticeable changes, depending on the context of the protein's function.
        - > ±1.5 kcal/mol: Significant impact on stability. Positive values indicate structural disruption or destabilization, while negative values suggest substantial stabilization.

        ΔΔG > +2.0 kcal/mol: Likely to cause a significant structural disruption, such as unfolding, local instability, or loss of functional conformation.

        Args:
            score (Column): column with ddG values

        Returns:
            Column: Normalised energies
        """
        return f.when(f.abs(score) >= 2, f.lit(1.0)).otherwise(
            cls._rescaleColumnValue(f.abs(score), 0.0, 2.0, 0.0, 1.00)
        )

    @classmethod
    def _normalise_cadd(
        cls: type[VariantEffectNormaliser],
        score: Column,
    ) -> Column:
        """Normalise CADD scores.

        Logic: CADD scores are divided into four ranges and scaled accordingly:
         - 0-10 -> -1-0 (likely benign ~2M)
         - 10-20 -> 0-0.5 (potentially deleterious ~300k)
         - 20-30 -> 0.5-0.75 (likely deleterious ~350k)
         - 30-81 -> 0.75-1 (highly likely deleterious ~86k)

        Args:
            score (Column): CADD score.

        Returns:
            Column: Normalised CADD score.
        """
        return (
            f.when(score <= 10, cls._rescaleColumnValue(score, 0, 10, -1.0, 0.0))
            .when(score <= 20, cls._rescaleColumnValue(score, 10, 20, 0.0, 0.5))
            .when(score <= 30, cls._rescaleColumnValue(score, 20, 30, 0.5, 0.75))
            .when(score > 30, cls._rescaleColumnValue(score, 30, 81, 0.75, 1))
        )

    @classmethod
    def _normalise_gerp(
        cls: type[VariantEffectNormaliser],
        score: Column,
    ) -> Column:
        """Normalise GERP scores.

        # Score interpretation from here:
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC7286533/
        # https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg19&g=allHg19RS_BW

        Logic: GERP scores are divided into three categories:
         - >6 : 1.0 - GERP scores are not bounded, so any value above 6 is considered as 1.0
         - 2-6: 0.5-1 - Highly conserved regions are scaled between 0.5 and 1
         - 0-2: 0-0.5 - Moderately conserved regions are scaled between 0 and 0.5
         - -3-0: -1-0.0 - Negative conservation indicates benign sequence alteration, so scaled between -1 and 0
         - < -3: -1.0 - As the score goes below -3, it is considered as -1.0

        Args:
            score (Column): GERP score.

        Returns:
            Column: Normalised GERP score.
        """
        return (
            f.when(score > 6, f.lit(1.0))
            .when(score >= 2, cls._rescaleColumnValue(score, 2, 6, 0.5, 1))
            .when(score >= 0, cls._rescaleColumnValue(score, 0, 2, 0, 0.5))
            .when(score >= -3, cls._rescaleColumnValue(score, -3, 0, -1, 0))
            .when(score < -3, f.lit(-1.0))
        )

    @classmethod
    def _normalise_lof(
        cls: type[VariantEffectNormaliser],
        assessment: Column,
    ) -> Column:
        """Normalise loss-of-function verdicts.

        There are five ordinal verdicts.
        The normalised score is determined by the verdict:
         - lof: 1
         - likely_lof: 0.5
         - uncertain: 0
         - likely_not_lof: -0.5
         - not_lof: -1

        Args:
            assessment (Column): Loss-of-function assessment.

        Returns:
            Column: Normalised loss-of-function score.
        """
        return (
            f.when(assessment == "lof", f.lit(1))
            .when(assessment == "likely_lof", f.lit(0.5))
            .when(assessment == "uncertain", f.lit(0))
            .when(assessment == "likely_not_lof", f.lit(-0.5))
            .when(assessment == "not_lof", f.lit(-1))
        )

    @classmethod
    def _normalise_loftee(
        cls: type[VariantEffectNormaliser],
        assessment: Column,
    ) -> Column:
        """Normalise LOFTEE scores.

        Logic: LOFTEE scores are divided into two categories:
         - HC (high confidence): 1.0 (~120k)
         - LC (low confidence): 0.85 (~18k)
        The normalised score is calculated based on the category the score falls into.

        Args:
            assessment (Column): LOFTEE assessment.

        Returns:
            Column: Normalised LOFTEE score.
        """
        return f.when(assessment == "HC", f.lit(1)).when(
            assessment == "LC", f.lit(0.85)
        )

    @classmethod
    def _normalise_sift(
        cls: type[VariantEffectNormaliser],
        score: Column,
        assessment: Column,
    ) -> Column:
        """Normalise SIFT scores.

        Logic: SIFT scores are divided into four categories:
         - deleterious and score >= 0.95: 0.75-1
         - deleterious_low_confidence and score >= 0.95: 0.5-0.75
         - tolerated_low_confidence and score <= 0.95: 0.25-0.5
         - tolerated and score <= 0.95: 0-0.25

        Args:
            score (Column): SIFT score.
            assessment (Column): SIFT assessment.

        Returns:
            Column: Normalised SIFT score.
        """
        return (
            f.when(
                (1 - f.round(score.cast(t.DoubleType()), 2) >= 0.95)
                & (assessment == "deleterious"),
                cls._rescaleColumnValue(1 - score, 0.95, 1, 0.5, 1),
            )
            .when(
                (1 - f.round(score.cast(t.DoubleType()), 2) >= 0.95)
                & (assessment == "deleterious_low_confidence"),
                cls._rescaleColumnValue(1 - score, 0.95, 1, 0, 0.5),
            )
            .when(
                (1 - f.round(score.cast(t.DoubleType()), 2) <= 0.95)
                & (assessment == "tolerated_low_confidence"),
                cls._rescaleColumnValue(1 - score, 0, 0.95, -0.5, 0.0),
            )
            .when(
                (1 - f.round(score.cast(t.DoubleType()), 2) <= 0.95)
                & (assessment == "tolerated"),
                cls._rescaleColumnValue(1 - score, 0, 0.95, -1, -0.5),
            )
        )

    @classmethod
    def _normalise_polyphen(
        cls: type[VariantEffectNormaliser],
        assessment: Column,
        score: Column,
    ) -> Column:
        """Normalise PolyPhen scores.

        Logic: PolyPhen scores are divided into three categories:
         - benign: 0-0.446: -1--0.25
         - possibly_damaging: 0.446-0.908: -0.25-0.25
         - probably_damaging: 0.908-1: 0.25-1
         - if assessment is unknown: None

        Args:
            assessment (Column): PolyPhen assessment.
            score (Column): PolyPhen score.

        Returns:
            Column: Normalised PolyPhen score.
        """
        return (
            f.when(assessment == "unknown", f.lit(None).cast(t.DoubleType()))
            .when(score <= 0.446, cls._rescaleColumnValue(score, 0, 0.446, -1.0, -0.25))
            .when(
                score <= 0.908,
                cls._rescaleColumnValue(score, 0.446, 0.908, -0.25, 0.25),
            )
            .when(score > 0.908, cls._rescaleColumnValue(score, 0.908, 1.0, 0.25, 1.0))
        )

    @classmethod
    def _normalise_alpha_missense(
        cls: type[VariantEffectNormaliser],
        score: Column,
    ) -> Column:
        """Normalise AlphaMissense scores.

        Logic: AlphaMissense scores are divided into three categories:
         - 0-0.06: -1.0--0.25
         - 0.06-0.77: -0.25-0.25
         - 0.77-1: 0.25-1

        Args:
            score (Column): AlphaMissense score.

        Returns:
            Column: Normalised AlphaMissense score.
        """
        return (
            f.when(score < 0.06, cls._rescaleColumnValue(score, 0, 0.06, -1.0, -0.25))
            .when(score < 0.77, cls._rescaleColumnValue(score, 0.06, 0.77, -0.25, 0.25))
            .when(score >= 0.77, cls._rescaleColumnValue(score, 0.77, 1, 0.25, 1))
        )

    @classmethod
    def _normalise_pangolin(
        cls: type[VariantEffectNormaliser],
        score: Column,
    ) -> Column:
        """Normalise Pangolin scores.

        Logic: Pangolin scores are divided into two categories:
            - 0-0.14: 0-0.25
            - 0.14-1: 0.75-1

        Args:
            score (Column): Pangolin score.

        Returns:
            Column: Normalised Pangolin score.
        """
        return f.when(
            f.abs(score) > 0.14, cls._rescaleColumnValue(f.abs(score), 0.14, 1, 0.5, 1)
        ).when(
            f.abs(score) <= 0.14,
            cls._rescaleColumnValue(f.abs(score), 0, 0.14, 0.0, 0.5),
        )
