"""Study locus dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, FloatType, LongType, StringType

from gentropy.common.genomic_region import GenomicRegion, KnownGenomicRegions
from gentropy.common.schemas import parse_spark_schema
from gentropy.common.spark_helpers import (
    calculate_neglog_pvalue,
    create_empty_column_if_not_exists,
    get_struct_field_schema,
    order_array_of_structs_by_field,
)
from gentropy.common.utils import get_logsum
from gentropy.config import WindowBasedClumpingStepConfig
from gentropy.dataset.dataset import Dataset
from gentropy.dataset.study_index import StudyQualityCheck
from gentropy.dataset.study_locus_overlap import StudyLocusOverlap
from gentropy.dataset.variant_index import VariantIndex
from gentropy.method.clump import LDclumping

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.types import StructType

    from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
    from gentropy.dataset.ld_index import LDIndex
    from gentropy.dataset.study_index import StudyIndex
    from gentropy.dataset.summary_statistics import SummaryStatistics
    from gentropy.dataset.target_index import TargetIndex
    from gentropy.method.l2g.feature_factory import L2GFeatureInputLoader


class CredibleSetConfidenceClasses(Enum):
    """Confidence assignments for credible sets, based on finemapping method and quality checks.

    List of confidence classes, from the highest to the lowest confidence level.

    Attributes:
        FINEMAPPED_IN_SAMPLE_LD (str): SuSiE fine-mapped credible set with in-sample LD
        FINEMAPPED_OUT_OF_SAMPLE_LD (str): SuSiE fine-mapped credible set with out-of-sample LD
        PICSED_SUMMARY_STATS (str): PICS fine-mapped credible set extracted from summary statistics
        PICSED_TOP_HIT (str): PICS fine-mapped credible set based on reported top hit
        UNKNOWN (str): Unknown confidence, for credible sets which did not fit any of the above categories
    """

    FINEMAPPED_IN_SAMPLE_LD = "SuSiE fine-mapped credible set with in-sample LD"
    FINEMAPPED_OUT_OF_SAMPLE_LD = "SuSiE fine-mapped credible set with out-of-sample LD"
    PICSED_SUMMARY_STATS = (
        "PICS fine-mapped credible set extracted from summary statistics"
    )
    PICSED_TOP_HIT = "PICS fine-mapped credible set based on reported top hit"
    UNKNOWN = "Unknown confidence"


class StudyLocusQualityCheck(Enum):
    """Study-Locus quality control options listing concerns on the quality of the association.

    Attributes:
        SUBSIGNIFICANT_FLAG (str): p-value below significance threshold
        NO_GENOMIC_LOCATION_FLAG (str): Incomplete genomic mapping
        COMPOSITE_FLAG (str): Composite association due to variant x variant interactions
        INCONSISTENCY_FLAG (str): Inconsistencies in the reported variants
        NON_MAPPED_VARIANT_FLAG (str): Variant not mapped to GnomAd
        PALINDROMIC_ALLELE_FLAG (str): Alleles are palindromic - cannot harmonize
        AMBIGUOUS_STUDY (str): Association with ambiguous study
        UNRESOLVED_LD (str): Variant not found in LD reference
        LD_CLUMPED (str): Explained by a more significant variant in high LD
        WINDOW_CLUMPED (str): Explained by a more significant variant in the same window
        NO_POPULATION (str): Study does not have population annotation to resolve LD
        FLAGGED_STUDY (str): Study has quality control flag(s)
        MISSING_STUDY (str): Flagging study loci if the study is not found in the study index as a reference
        DUPLICATED_STUDYLOCUS_ID (str): Study-locus identifier is not unique
        INVALID_VARIANT_IDENTIFIER (str): Flagging study loci where identifier of any tagging variant was not found in the variant index
        TOP_HIT (str): Study locus from curated top hit
        IN_MHC (str): Flagging study loci in the MHC region
        REDUNDANT_PICS_TOP_HIT (str): Flagging study loci in studies with PICS results from summary statistics
        EXPLAINED_BY_SUSIE (str): Study locus in region explained by a SuSiE credible set
        ABNORMAL_PIPS (str): Flagging study loci with a sum of PIPs that are not in [0.99,1]
        OUT_OF_SAMPLE_LD (str): Study locus finemapped without in-sample LD reference
        INVALID_CHROMOSOME (str): Chromosome not in 1:22, X, Y, XY or MT
        TOP_HIT_AND_SUMMARY_STATS (str): Curated top hit is flagged because summary statistics are available for study
    """

    SUBSIGNIFICANT_FLAG = "Subsignificant p-value"
    NO_GENOMIC_LOCATION_FLAG = "Incomplete genomic mapping"
    COMPOSITE_FLAG = "Composite association"
    INCONSISTENCY_FLAG = "Variant inconsistency"
    NON_MAPPED_VARIANT_FLAG = "No mapping in GnomAd"
    PALINDROMIC_ALLELE_FLAG = "Palindrome alleles - cannot harmonize"
    AMBIGUOUS_STUDY = "Association with ambiguous study"
    UNRESOLVED_LD = "Variant not found in LD reference"
    LD_CLUMPED = "Explained by a more significant variant in high LD"
    WINDOW_CLUMPED = "Explained by a more significant variant in the same window"
    NO_POPULATION = "Study does not have population annotation to resolve LD"
    FLAGGED_STUDY = "Study has quality control flag(s)"
    MISSING_STUDY = "Study not found in the study index"
    DUPLICATED_STUDYLOCUS_ID = "Non-unique study locus identifier"
    INVALID_VARIANT_IDENTIFIER = (
        "Some variant identifiers of this locus were not found in variant index"
    )
    IN_MHC = "MHC region"
    REDUNDANT_PICS_TOP_HIT = (
        "PICS results from summary statistics available for this same study"
    )
    TOP_HIT = "Study locus from curated top hit"
    EXPLAINED_BY_SUSIE = "Study locus in region explained by a SuSiE credible set"
    OUT_OF_SAMPLE_LD = "Study locus finemapped without in-sample LD reference"
    ABNORMAL_PIPS = (
        "Study locus with a sum of PIPs that not in the expected range [0.95,1]"
    )
    INVALID_CHROMOSOME = "Chromosome not in 1:22, X, Y, XY or MT"
    TOP_HIT_AND_SUMMARY_STATS = (
        "Curated top hit is flagged because summary statistics are available for study"
    )


class CredibleInterval(Enum):
    """Credible interval enum.

    Interval within which an unobserved parameter value falls with a particular probability.

    Attributes:
        IS95 (str): 95% credible interval
        IS99 (str): 99% credible interval
    """

    IS95 = "is95CredibleSet"
    IS99 = "is99CredibleSet"


class FinemappingMethod(Enum):
    """Finemapping method enum.

    Attributes:
        PICS (str): PICS
        SUSIE (str): SuSiE method
        SUSIE_INF (str): SuSiE-inf method implemented in `gentropy`
    """

    PICS = "PICS"
    SUSIE = "SuSie"
    SUSIE_INF = "SuSiE-inf"


@dataclass
class StudyLocus(Dataset):
    """Study-Locus dataset.

    This dataset captures associations between study/traits and a genetic loci as provided by finemapping methods.
    """

    def validate_study(self: StudyLocus, study_index: StudyIndex) -> StudyLocus:
        """Flagging study loci if the corresponding study has issues.

        There are two different potential flags:
        - flagged study: flagging locus if the study has quality control flags.
        - study with summary statistics for top hit: flagging locus if the study has available summary statistics.
        - missing study: flagging locus if the study was not found in the reference study index.

        Args:
            study_index (StudyIndex): Study index to resolve study types.

        Returns:
            StudyLocus: Updated study locus with quality control flags.
        """
        # Quality controls is not a mandatory field in the study index schema, so we have to be ready to handle it:
        qc_select_expression = (
            f.col("qualityControls")
            if "qualityControls" in study_index.df.columns
            else f.lit(None).cast(StringType())
        )

        # The study Id of the study index needs to be kept, because we would not know which study was in the index after the left join:
        study_flags = study_index.df.select(
            f.col("studyId").alias("study_studyId"),
            qc_select_expression.alias("study_qualityControls"),
        )

        return StudyLocus(
            _df=(
                self.df.join(
                    study_flags, f.col("studyId") == f.col("study_studyId"), "left"
                )
                # Flagging loci with flagged studies - without propagating the actual flags:
                .withColumn(
                    "qualityControls",
                    StudyLocus.update_quality_flag(
                        f.col("qualityControls"),
                        f.size(f.col("study_qualityControls")) > 0,
                        StudyLocusQualityCheck.FLAGGED_STUDY,
                    ),
                )
                # Flagging top-hits, where the study has available summary statistics:
                .withColumn(
                    "qualityControls",
                    StudyLocus.update_quality_flag(
                        f.col("qualityControls"),
                        # Condition is true, if the study has summary statistics available and the locus is a top hit:
                        f.array_contains(
                            f.col("qualityControls"),
                            StudyLocusQualityCheck.TOP_HIT.value,
                        )
                        & ~f.array_contains(
                            f.col("study_qualityControls"),
                            StudyQualityCheck.SUMSTATS_NOT_AVAILABLE.value,
                        ),
                        StudyLocusQualityCheck.TOP_HIT_AND_SUMMARY_STATS,
                    ),
                )
                # Flagging loci where no studies were found:
                .withColumn(
                    "qualityControls",
                    StudyLocus.update_quality_flag(
                        f.col("qualityControls"),
                        f.col("study_studyId").isNull(),
                        StudyLocusQualityCheck.MISSING_STUDY,
                    ),
                )
                .drop("study_studyId", "study_qualityControls")
            ),
            _schema=self.get_schema(),
        )

    def annotate_study_type(self: StudyLocus, study_index: StudyIndex) -> StudyLocus:
        """Gets study type from study index and adds it to study locus.

        Args:
            study_index (StudyIndex): Study index to get study type.

        Returns:
            StudyLocus: Updated study locus with study type.
        """
        return StudyLocus(
            _df=(
                self.df.drop("studyType").join(
                    study_index.study_type_lut(), on="studyId", how="left"
                )
            ),
            _schema=self.get_schema(),
        )

    def validate_chromosome_label(self: StudyLocus) -> StudyLocus:
        """Flagging study loci, where chromosome is coded not as 1:22, X, Y, Xy and MT.

        Returns:
            StudyLocus: Updated study locus with quality control flags.
        """
        # QC column might not be present in the variant index schema, so we have to be ready to handle it:
        qc_select_expression = (
            f.col("qualityControls")
            if "qualityControls" in self.df.columns
            else f.lit(None).cast(ArrayType(StringType()))
        )
        valid_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y", "XY", "MT"]

        return StudyLocus(
            _df=(
                self.df.withColumn(
                    "qualityControls",
                    self.update_quality_flag(
                        qc_select_expression,
                        ~f.col("chromosome").isin(valid_chromosomes),
                        StudyLocusQualityCheck.INVALID_CHROMOSOME,
                    ),
                )
            ),
            _schema=self.get_schema(),
        )

    def validate_variant_identifiers(
        self: StudyLocus, variant_index: VariantIndex
    ) -> StudyLocus:
        """Flagging study loci, where tagging variant identifiers are not found in variant index.

        Args:
            variant_index (VariantIndex): Variant index to resolve variant identifiers.

        Returns:
            StudyLocus: Updated study locus with quality control flags.
        """
        # QC column might not be present in the variant index schema, so we have to be ready to handle it:
        qc_select_expression = (
            f.col("qualityControls")
            if "qualityControls" in self.df.columns
            else f.lit(None).cast(ArrayType(StringType()))
        )

        # Find out which study loci have variants not in the variant index:
        flag = (
            self.df
            # Exploding locus:
            .select("studyLocusId", f.explode("locus").alias("locus"))
            .select("studyLocusId", "locus.variantId")
            # Join with variant index variants:
            .join(
                variant_index.df.select(
                    "variantId", f.lit(True).alias("inVariantIndex")
                ),
                on="variantId",
                how="left",
            )
            # Flagging variants not in the variant index:
            .withColumn("inVariantIndex", f.col("inVariantIndex").isNotNull())
            # Flagging study loci with ANY variants not in the variant index:
            .groupBy("studyLocusId")
            .agg(f.collect_set("inVariantIndex").alias("inVariantIndex"))
            .select(
                "studyLocusId",
                f.array_contains("inVariantIndex", False).alias("toFlag"),
            )
        )

        return StudyLocus(
            _df=(
                self.df.join(flag, on="studyLocusId", how="left")
                .withColumn(
                    "qualityControls",
                    self.update_quality_flag(
                        qc_select_expression,
                        f.col("toFlag"),
                        StudyLocusQualityCheck.INVALID_VARIANT_IDENTIFIER,
                    ),
                )
                .drop("toFlag")
            ),
            _schema=self.get_schema(),
        )

    def validate_lead_pvalue(self: StudyLocus, pvalue_cutoff: float) -> StudyLocus:
        """Flag associations below significant threshold.

        Args:
            pvalue_cutoff (float): association p-value cut-off

        Returns:
            StudyLocus: Updated study locus with quality control flags.
        """
        df = self.df
        qc_colname = StudyLocus.get_QC_column_name()
        if qc_colname not in self.df.columns:
            df = self.df.withColumn(
                qc_colname,
                create_empty_column_if_not_exists(
                    qc_colname,
                    get_struct_field_schema(StudyLocus.get_schema(), qc_colname),
                ),
            )
        return StudyLocus(
            _df=(
                df.withColumn(
                    qc_colname,
                    # Because this QC might already run on the dataset, the unique set of flags is generated:
                    f.array_distinct(
                        self._qc_subsignificant_associations(
                            f.col("qualityControls"),
                            f.col("pValueMantissa"),
                            f.col("pValueExponent"),
                            pvalue_cutoff,
                        )
                    ),
                )
            ),
            _schema=self.get_schema(),
        )

    def validate_unique_study_locus_id(self: StudyLocus) -> StudyLocus:
        """Validating the uniqueness of study-locus identifiers and flagging duplicated studyloci.

        Returns:
            StudyLocus: with flagged duplicated studies.
        """
        return StudyLocus(
            _df=self.df.withColumn(
                "qualityControls",
                self.update_quality_flag(
                    f.col("qualityControls"),
                    self.flag_duplicates(f.col("studyLocusId")),
                    StudyLocusQualityCheck.DUPLICATED_STUDYLOCUS_ID,
                ),
            ),
            _schema=StudyLocus.get_schema(),
        )

    @staticmethod
    def _qc_subsignificant_associations(
        quality_controls_column: Column,
        p_value_mantissa: Column,
        p_value_exponent: Column,
        pvalue_cutoff: float,
    ) -> Column:
        """Flag associations below significant threshold.

        Args:
            quality_controls_column (Column): QC column
            p_value_mantissa (Column): P-value mantissa column
            p_value_exponent (Column): P-value exponent column
            pvalue_cutoff (float): association p-value cut-off

        Returns:
            Column: Updated QC column with flag.

        Examples:
            >>> import pyspark.sql.types as t
            >>> d = [{'qc': None, 'p_value_mantissa': 1, 'p_value_exponent': -7}, {'qc': None, 'p_value_mantissa': 1, 'p_value_exponent': -8}, {'qc': None, 'p_value_mantissa': 5, 'p_value_exponent': -8}, {'qc': None, 'p_value_mantissa': 1, 'p_value_exponent': -9}]
            >>> df = spark.createDataFrame(d, t.StructType([t.StructField('qc', t.ArrayType(t.StringType()), True), t.StructField('p_value_mantissa', t.IntegerType()), t.StructField('p_value_exponent', t.IntegerType())]))
            >>> df.withColumn('qc', StudyLocus._qc_subsignificant_associations(f.col("qc"), f.col("p_value_mantissa"), f.col("p_value_exponent"), 5e-8)).show(truncate = False)
            +------------------------+----------------+----------------+
            |qc                      |p_value_mantissa|p_value_exponent|
            +------------------------+----------------+----------------+
            |[Subsignificant p-value]|1               |-7              |
            |[]                      |1               |-8              |
            |[]                      |5               |-8              |
            |[]                      |1               |-9              |
            +------------------------+----------------+----------------+
            <BLANKLINE>

        """
        return StudyLocus.update_quality_flag(
            quality_controls_column,
            calculate_neglog_pvalue(p_value_mantissa, p_value_exponent)
            < f.lit(-np.log10(pvalue_cutoff)),
            StudyLocusQualityCheck.SUBSIGNIFICANT_FLAG,
        )

    def qc_abnormal_pips(
        self: StudyLocus,
        sum_pips_lower_threshold: float = 0.99,
        # Set slightly above 1 to account for floating point errors
        sum_pips_upper_threshold: float = 1.0001,
    ) -> StudyLocus:
        """Filter study-locus by sum of posterior inclusion probabilities to ensure that the sum of PIPs is within a given range.

        Args:
            sum_pips_lower_threshold (float): Lower threshold for the sum of PIPs.
            sum_pips_upper_threshold (float): Upper threshold for the sum of PIPs.

        Returns:
            StudyLocus: Filtered study-locus dataset.
        """
        # QC column might not be present so we have to be ready to handle it:
        qc_select_expression = (
            f.col("qualityControls")
            if "qualityControls" in self.df.columns
            else f.lit(None).cast(ArrayType(StringType()))
        )

        flag = self.df.withColumn(
            "sumPosteriorProbability",
            f.aggregate(
                f.col("locus"),
                f.lit(0.0),
                lambda acc, x: acc + x["posteriorProbability"],
            ),
        ).withColumn(
            "pipOutOfRange",
            f.when(
                (f.col("sumPosteriorProbability") < sum_pips_lower_threshold)
                | (f.col("sumPosteriorProbability") > sum_pips_upper_threshold),
                True,
            ).otherwise(False),
        )

        return StudyLocus(
            _df=(
                flag
                # Flagging loci with failed studies:
                .withColumn(
                    "qualityControls",
                    self.update_quality_flag(
                        qc_select_expression,
                        f.col("pipOutOfRange"),
                        StudyLocusQualityCheck.ABNORMAL_PIPS,
                    ),
                ).drop("sumPosteriorProbability", "pipOutOfRange")
            ),
            _schema=self.get_schema(),
        )

    @staticmethod
    def _overlapping_peaks(
        credset_to_overlap: DataFrame, intra_study_overlap: bool = False
    ) -> DataFrame:
        """Calculate overlapping signals (study-locus) between GWAS-GWAS and GWAS-Molecular trait.

        Args:
            credset_to_overlap (DataFrame): DataFrame containing at least `studyLocusId`, `studyType`, `chromosome` and `tagVariantId` columns.
            intra_study_overlap (bool): When True, finds intra-study overlaps for credible set deduplication. Default is False.

        Returns:
            DataFrame: containing `leftStudyLocusId`, `rightStudyLocusId` and `chromosome` columns.
        """
        # Reduce columns to the minimum to reduce the size of the dataframe
        credset_to_overlap = credset_to_overlap.select(
            "studyLocusId",
            "studyId",
            "studyType",
            "chromosome",
            "region",
            "tagVariantId",
        )
        # Define join condition - if intra_study_overlap is True, finds overlaps within the same study. Otherwise finds gwas vs everything overlaps for coloc.
        join_condition = (
            [
                f.col("left.studyId") == f.col("right.studyId"),
                f.col("left.chromosome") == f.col("right.chromosome"),
                f.col("left.tagVariantId") == f.col("right.tagVariantId"),
                f.col("left.studyLocusId") > f.col("right.studyLocusId"),
                f.col("left.region") != f.col("right.region"),
            ]
            if intra_study_overlap
            else [
                f.col("left.chromosome") == f.col("right.chromosome"),
                f.col("left.tagVariantId") == f.col("right.tagVariantId"),
                (f.col("right.studyType") != "gwas")
                | (f.col("left.studyLocusId") > f.col("right.studyLocusId")),
                f.col("left.studyType") == f.lit("gwas"),
            ]
        )

        return (
            credset_to_overlap.alias("left")
            # Self join with complex condition.
            .join(
                credset_to_overlap.alias("right"),
                on=join_condition,
                how="inner",
            )
            .select(
                f.col("left.studyLocusId").alias("leftStudyLocusId"),
                f.col("right.studyLocusId").alias("rightStudyLocusId"),
                f.col("right.studyType").alias("rightStudyType"),
                f.col("left.chromosome").alias("chromosome"),
            )
            .distinct()
            .repartition("chromosome")
            .persist()
        )

    @staticmethod
    def _align_overlapping_tags(
        loci_to_overlap: DataFrame, peak_overlaps: DataFrame
    ) -> StudyLocusOverlap:
        """Align overlapping tags in pairs of overlapping study-locus, keeping all tags in both loci.

        Args:
            loci_to_overlap (DataFrame): containing `studyLocusId`, `studyType`, `chromosome`, `tagVariantId`, `logBF` and `posteriorProbability` columns.
            peak_overlaps (DataFrame): containing `leftStudyLocusId`, `rightStudyLocusId` and `chromosome` columns.

        Returns:
            StudyLocusOverlap: Pairs of overlapping study-locus with aligned tags.
        """
        # Complete information about all tags in the left study-locus of the overlap
        stats_cols = [
            "logBF",
            "posteriorProbability",
            "beta",
            "pValueMantissa",
            "pValueExponent",
        ]
        overlapping_left = loci_to_overlap.select(
            f.col("chromosome"),
            f.col("tagVariantId"),
            f.col("studyLocusId").alias("leftStudyLocusId"),
            *[f.col(col).alias(f"left_{col}") for col in stats_cols],
        ).join(peak_overlaps, on=["chromosome", "leftStudyLocusId"], how="inner")

        # Complete information about all tags in the right study-locus of the overlap
        overlapping_right = loci_to_overlap.select(
            f.col("chromosome"),
            f.col("tagVariantId"),
            f.col("studyLocusId").alias("rightStudyLocusId"),
            *[f.col(col).alias(f"right_{col}") for col in stats_cols],
        ).join(peak_overlaps, on=["chromosome", "rightStudyLocusId"], how="inner")

        # Include information about all tag variants in both study-locus aligned by tag variant id
        overlaps = overlapping_left.join(
            overlapping_right,
            on=[
                "chromosome",
                "rightStudyLocusId",
                "leftStudyLocusId",
                "tagVariantId",
                "rightStudyType",
            ],
            how="outer",
        ).select(
            "leftStudyLocusId",
            "rightStudyLocusId",
            "rightStudyType",
            "chromosome",
            "tagVariantId",
            f.struct(
                *[f"left_{e}" for e in stats_cols] + [f"right_{e}" for e in stats_cols]
            ).alias("statistics"),
        )
        return StudyLocusOverlap(
            _df=overlaps,
            _schema=StudyLocusOverlap.get_schema(),
        )

    @staticmethod
    def assign_study_locus_id(uniqueness_defining_columns: list[str]) -> Column:
        """Hashes the provided columns to extract a consistent studyLocusId.

        Args:
            uniqueness_defining_columns (list[str]): list of columns defining uniqueness

        Returns:
            Column: column with a study locus ID

        Examples:
            >>> df = spark.createDataFrame([("GCST000001", "1_1000_A_C", "SuSiE-inf"), ("GCST000002", "1_1000_A_C", "pics")]).toDF("studyId", "variantId", "finemappingMethod")
            >>> df.withColumn("study_locus_id", StudyLocus.assign_study_locus_id(["studyId", "variantId", "finemappingMethod"])).show(truncate=False)
            +----------+----------+-----------------+--------------------------------+
            |studyId   |variantId |finemappingMethod|study_locus_id                  |
            +----------+----------+-----------------+--------------------------------+
            |GCST000001|1_1000_A_C|SuSiE-inf        |109804fe1e20c94231a31bafd71b566e|
            |GCST000002|1_1000_A_C|pics             |de310be4558e0482c9cc359c97d37773|
            +----------+----------+-----------------+--------------------------------+
            <BLANKLINE>
        """
        return Dataset.generate_identifier(uniqueness_defining_columns).alias(
            "studyLocusId"
        )

    @classmethod
    def calculate_credible_set_log10bf(cls: type[StudyLocus], logbfs: Column) -> Column:
        """Calculate Bayes factor for the entire credible set. The Bayes factor is calculated as the logsumexp of the logBF values of the variants in the locus.

        Args:
            logbfs (Column): Array column with the logBF values of the variants in the locus.

        Returns:
            Column: log10 Bayes factor for the entire credible set.

        Examples:
            >>> spark.createDataFrame([([0.2, 0.1, 0.05, 0.0],)]).toDF("logBF").select(f.round(StudyLocus.calculate_credible_set_log10bf(f.col("logBF")), 7).alias("credibleSetlog10BF")).show()
            +------------------+
            |credibleSetlog10BF|
            +------------------+
            |         0.6412604|
            +------------------+
            <BLANKLINE>
        """
        # log10=log/log(10)=log*0.43429448190325176
        logsumexp_udf = f.udf(
            lambda x: (get_logsum(x) * 0.43429448190325176), FloatType()
        )
        return logsumexp_udf(logbfs).cast("double").alias("credibleSetlog10BF")

    @classmethod
    def get_schema(cls: type[StudyLocus]) -> StructType:
        """Provides the schema for the StudyLocus dataset.

        Returns:
            StructType: schema for the StudyLocus dataset.
        """
        return parse_spark_schema("study_locus.json")

    @classmethod
    def get_QC_column_name(cls: type[StudyLocus]) -> str:
        """Quality control column.

        Returns:
            str: Name of the quality control column.
        """
        return "qualityControls"

    @classmethod
    def get_QC_mappings(cls: type[StudyLocus]) -> dict[str, str]:
        """Quality control flag to QC column category mappings.

        Returns:
            dict[str, str]: Mapping between flag name and QC column category value.
        """
        return {member.name: member.value for member in StudyLocusQualityCheck}

    def flag_trans_qtls(
        self: StudyLocus,
        study_index: StudyIndex,
        target_index: TargetIndex,
        trans_threshold: int = 5_000_000,
    ) -> StudyLocus:
        """Flagging transQTL credible sets based on genomic location of the measured gene.

        Process:
        0. Make sure that the `isTransQtl` column does not exist (remove if exists)
        1. Enrich study-locus dataset with geneId based on study metadata. (only QTL studies are considered)
        2. Enrich with transcription start site and chromosome of the studied gegne.
        3. Flagging any tagging variant of QTL credible sets, if chromosome is different from the gene or distance is above the threshold.
        4. Propagate flags to credible sets where any tags are considered as trans.
        5. Return study locus object with annotation stored in 'isTransQtl` boolean column, where gwas credible sets will be `null`

        Args:
            study_index (StudyIndex): study index to extract identifier of the measured gene
            target_index (TargetIndex): target index bringing TSS and chromosome of the measured gene
            trans_threshold (int): Distance above which the QTL is considered trans. Default: 5_000_000bp

        Returns:
            StudyLocus: new column added indicating if the QTL credibles sets are trans.
        """
        # As the `geneId` column in the study index is optional, we have to test for that:
        if "geneId" not in study_index.df.columns:
            return self

        # We have to remove the column `isTransQtl` to ensure the column is not duplicated
        # The duplication can happen when one reads the StudyLocus from parquet with
        # predefined schema that already contains the `isTransQtl` column.
        if "isTransQtl" in self.df.columns:
            self.df = self.df.drop("isTransQtl")

        # Process study index:
        processed_studies = (
            study_index.df
            # Dropping gwas studies. This ensures that only QTLs will have "isTrans" annotation:
            .filter(f.col("studyType") != "gwas").select(
                "studyId", "geneId", "projectId"
            )
        )

        # Process study locus:
        processed_credible_set = (
            self.df
            # Exploding locus to test all tag variants:
            .withColumn("locus", f.explode("locus")).select(
                "studyLocusId",
                "studyId",
                f.split("locus.variantId", "_")[0].alias("chromosome"),
                f.split("locus.variantId", "_")[1].cast(LongType()).alias("position"),
            )
        )

        # Process target index:
        processed_targets = target_index.df.select(
            f.col("id").alias("geneId"),
            f.col("tss"),
            f.col("genomicLocation.chromosome").alias("geneChromosome"),
        )

        # Pool datasets:
        joined_data = (
            processed_credible_set
            # Join processed studies:
            .join(processed_studies, on="studyId", how="inner")
            # Join processed targets:
            .join(processed_targets, on="geneId", how="left")
            # Assign True/False for QTL studies:
            .withColumn(
                "isTagTrans",
                # The QTL signal is considered trans if the locus is on a different chromosome than the measured gene.
                # OR the distance from the gene's transcription start site is > threshold.
                f.when(
                    (f.col("chromosome") != f.col("geneChromosome"))
                    | (f.abs(f.col("tss") - f.col("position")) > trans_threshold),
                    f.lit(True),
                ).otherwise(f.lit(False)),
            )
            .groupby("studyLocusId")
            .agg(
                # If all tagging variants of the locus is in trans position, the QTL is considered trans:
                f.when(
                    f.array_contains(f.collect_set("isTagTrans"), f.lit(False)), False
                )
                .otherwise(f.lit(True))
                .alias("isTransQtl")
            )
        )
        # Adding new column, where the value is null for gwas loci:
        return StudyLocus(self.df.join(joined_data, on="studyLocusId", how="left"))

    def filter_credible_set(
        self: StudyLocus,
        credible_interval: CredibleInterval,
    ) -> StudyLocus:
        """Annotate and filter study-locus tag variants based on given credible interval.

        Args:
            credible_interval (CredibleInterval): Credible interval to filter for.

        Returns:
            StudyLocus: Filtered study-locus dataset.
        """
        return StudyLocus(
            _df=self.annotate_credible_sets().df.withColumn(
                "locus",
                f.filter(
                    f.col("locus"),
                    lambda tag: (tag[credible_interval.value]),
                ),
            ),
            _schema=self._schema,
        )

    @staticmethod
    def filter_ld_set(ld_set: Column, r2_threshold: float) -> Column:
        """Filter the LD set by a given R2 threshold.

        Args:
            ld_set (Column): LD set
            r2_threshold (float): R2 threshold to filter the LD set on

        Returns:
            Column: Filtered LD index
        """
        return f.when(
            ld_set.isNotNull(),
            f.filter(
                ld_set,
                lambda tag: tag["r2Overall"] >= r2_threshold,
            ),
        )

    def find_overlaps(
        self: StudyLocus, intra_study_overlap: bool = False
    ) -> StudyLocusOverlap:
        """Calculate overlapping study-locus.

        Find overlapping study-locus that share at least one tagging variant. All GWAS-GWAS and all GWAS-Molecular traits are computed with the Molecular traits always
        appearing on the right side.

        Args:
            intra_study_overlap (bool): If True, finds intra-study overlaps for credible set deduplication. Default is False.

        Returns:
            StudyLocusOverlap: Pairs of overlapping study-locus with aligned tags.
        """
        loci_to_overlap = (
            self.df.filter(f.col("studyType").isNotNull())
            .withColumn("locus", f.explode("locus"))
            .select(
                "studyLocusId",
                "studyId",
                "studyType",
                "chromosome",
                "region",
                f.col("locus.variantId").alias("tagVariantId"),
                f.col("locus.logBF").alias("logBF"),
                f.col("locus.posteriorProbability").alias("posteriorProbability"),
                f.col("locus.pValueMantissa").alias("pValueMantissa"),
                f.col("locus.pValueExponent").alias("pValueExponent"),
                f.col("locus.beta").alias("beta"),
            )
            .persist()
        )

        # overlapping study-locus
        peak_overlaps = self._overlapping_peaks(loci_to_overlap, intra_study_overlap)

        # study-locus overlap by aligning overlapping variants
        return self._align_overlapping_tags(loci_to_overlap, peak_overlaps)

    def unique_variants_in_locus(self: StudyLocus) -> DataFrame:
        """All unique variants collected in a `StudyLocus` dataframe.

        Returns:
            DataFrame: A dataframe containing `variantId` and `chromosome` columns.
        """
        return (
            self.df.withColumn(
                "variantId",
                # Joint array of variants in that studylocus. Locus can be null
                f.explode(
                    f.array_union(
                        f.array(f.col("variantId")),
                        f.coalesce(f.col("locus.variantId"), f.array()),
                    )
                ),
            )
            .select(
                "variantId", f.split(f.col("variantId"), "_")[0].alias("chromosome")
            )
            .distinct()
        )

    def neglog_pvalue(self: StudyLocus) -> Column:
        """Returns the negative log p-value.

        Returns:
            Column: Negative log p-value
        """
        return calculate_neglog_pvalue(
            self.df.pValueMantissa,
            self.df.pValueExponent,
        )

    def build_feature_matrix(
        self: StudyLocus,
        features_list: list[str],
        features_input_loader: L2GFeatureInputLoader,
    ) -> L2GFeatureMatrix:
        """Returns the feature matrix for a StudyLocus.

        Args:
            features_list (list[str]): List of features to include in the feature matrix.
            features_input_loader (L2GFeatureInputLoader): Feature input loader to use.

        Returns:
            L2GFeatureMatrix: Feature matrix for this study-locus.
        """
        from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix

        return L2GFeatureMatrix.from_features_list(
            self,
            features_list,
            features_input_loader,
        ).fill_na()

    def annotate_credible_sets(self: StudyLocus) -> StudyLocus:
        """Annotate study-locus dataset with credible set flags.

        Sorts the array in the `locus` column elements by their `posteriorProbability` values in descending order and adds
        `is95CredibleSet` and `is99CredibleSet` fields to the elements, indicating which are the tagging variants whose cumulative sum
        of their `posteriorProbability` values is below 0.95 and 0.99, respectively.

        Returns:
            StudyLocus: including annotation on `is95CredibleSet` and `is99CredibleSet`.

        Raises:
            ValueError: If `locus` column is not available.
        """
        if "locus" not in self.df.columns:
            raise ValueError("Locus column not available.")

        self.df = self.df.withColumn(
            # Sort credible set by posterior probability in descending order
            "locus",
            f.when(
                f.col("locus").isNotNull() & (f.size(f.col("locus")) > 0),
                order_array_of_structs_by_field("locus", "posteriorProbability"),
            ),
        ).withColumn(
            # Calculate array of cumulative sums of posterior probabilities to determine which variants are in the 95% and 99% credible sets
            # and zip the cumulative sums array with the credible set array to add the flags
            "locus",
            f.when(
                f.col("locus").isNotNull() & (f.size(f.col("locus")) > 0),
                f.zip_with(
                    f.col("locus"),
                    f.transform(
                        f.sequence(f.lit(1), f.size(f.col("locus"))),
                        lambda index: f.aggregate(
                            f.slice(
                                # By using `index - 1` we introduce a value of `0.0` in the cumulative sums array. to ensure that the last variant
                                # that exceeds the 0.95 threshold is included in the cumulative sum, as its probability is necessary to satisfy the threshold.
                                f.col("locus.posteriorProbability"),
                                1,
                                index - 1,
                            ),
                            f.lit(0.0),
                            lambda acc, el: acc + el,
                        ),
                    ),
                    lambda struct_e, acc: struct_e.withField(
                        CredibleInterval.IS95.value, (acc < 0.95) & acc.isNotNull()
                    ).withField(
                        CredibleInterval.IS99.value, (acc < 0.99) & acc.isNotNull()
                    ),
                ),
            ),
        )
        return self

    def annotate_locus_statistics(
        self: StudyLocus,
        summary_statistics: SummaryStatistics,
        collect_locus_distance: int,
    ) -> StudyLocus:
        """Annotates study locus with summary statistics in the specified distance around the position.

        Args:
            summary_statistics (SummaryStatistics): Summary statistics to be used for annotation.
            collect_locus_distance (int): distance from variant defining window for inclusion of variants in locus.

        Returns:
            StudyLocus: Study locus annotated with summary statistics in `locus` column. If no statistics are found, the `locus` column will be empty.
        """
        # The clumps will be used several times (persisting)
        self.df.persist()
        # Renaming columns:
        sumstats_renamed = summary_statistics.df.selectExpr(
            *[f"{col} as tag_{col}" for col in summary_statistics.df.columns]
        ).alias("sumstat")

        locus_df = (
            sumstats_renamed
            # Joining the two datasets together:
            .join(
                f.broadcast(
                    self.df.alias("clumped").select(
                        "position", "chromosome", "studyId", "studyLocusId"
                    )
                ),
                on=[
                    (f.col("sumstat.tag_studyId") == f.col("clumped.studyId"))
                    & (f.col("sumstat.tag_chromosome") == f.col("clumped.chromosome"))
                    & (
                        f.col("sumstat.tag_position")
                        >= (f.col("clumped.position") - collect_locus_distance)
                    )
                    & (
                        f.col("sumstat.tag_position")
                        <= (f.col("clumped.position") + collect_locus_distance)
                    )
                ],
                how="inner",
            )
            .withColumn(
                "locus",
                f.struct(
                    f.col("tag_variantId").alias("variantId"),
                    f.col("tag_beta").alias("beta"),
                    f.col("tag_pValueMantissa").alias("pValueMantissa"),
                    f.col("tag_pValueExponent").alias("pValueExponent"),
                    f.col("tag_standardError").alias("standardError"),
                ),
            )
            .groupBy("studyLocusId")
            .agg(
                f.collect_list(f.col("locus")).alias("locus"),
            )
        )

        self.df = self.df.drop("locus").join(
            locus_df,
            on="studyLocusId",
            how="left",
        )

        return self

    def annotate_ld(
        self: StudyLocus,
        study_index: StudyIndex,
        ld_index: LDIndex,
        r2_threshold: float = 0.0,
    ) -> StudyLocus:
        """Annotate LD information to study-locus.

        Args:
            study_index (StudyIndex): Study index to resolve ancestries.
            ld_index (LDIndex): LD index to resolve LD information.
            r2_threshold (float): R2 threshold to filter the LD index. Default is 0.0.

        Returns:
            StudyLocus: Study locus annotated with ld information from LD index.
        """
        from gentropy.method.ld import LDAnnotator

        return LDAnnotator.ld_annotate(self, study_index, ld_index, r2_threshold)

    def clump(self: StudyLocus) -> StudyLocus:
        """Perform LD clumping of the studyLocus.

        Evaluates whether a lead variant is linked to a tag (with lowest p-value) in the same studyLocus dataset.

        Returns:
            StudyLocus: with empty credible sets for linked variants and QC flag.
        """
        clumped_df = (
            self.df.withColumn(
                "is_lead_linked",
                LDclumping._is_lead_linked(
                    self.df.studyId,
                    self.df.chromosome,
                    self.df.variantId,
                    self.df.pValueExponent,
                    self.df.pValueMantissa,
                    self.df.ldSet,
                ),
            )
            .withColumn(
                "ldSet",
                f.when(f.col("is_lead_linked"), f.array()).otherwise(f.col("ldSet")),
            )
            .withColumn(
                "qualityControls",
                StudyLocus.update_quality_flag(
                    f.col("qualityControls"),
                    f.col("is_lead_linked"),
                    StudyLocusQualityCheck.LD_CLUMPED,
                ),
            )
            .drop("is_lead_linked")
        )
        return StudyLocus(
            _df=clumped_df,
            _schema=self.get_schema(),
        )

    def exclude_region(
        self: StudyLocus, region: GenomicRegion, exclude_overlap: bool = False
    ) -> StudyLocus:
        """Exclude a region from the StudyLocus dataset.

        Args:
            region (GenomicRegion): genomic region object.
            exclude_overlap (bool): If True, excludes StudyLocus windows with any overlap with the region.

        Returns:
            StudyLocus: filtered StudyLocus object.
        """
        if exclude_overlap:
            filter_condition = ~(
                (f.col("chromosome") == region.chromosome)
                & (
                    (f.col("locusStart") <= region.end)
                    & (f.col("locusEnd") >= region.start)
                )
            )
        else:
            filter_condition = ~(
                (f.col("chromosome") == region.chromosome)
                & (
                    (f.col("position") >= region.start)
                    & (f.col("position") <= region.end)
                )
            )

        return StudyLocus(
            _df=self.df.filter(filter_condition),
            _schema=StudyLocus.get_schema(),
        )

    def qc_MHC_region(self: StudyLocus) -> StudyLocus:
        """Adds qualityControl flag when lead overlaps with MHC region.

        Returns:
            StudyLocus: including qualityControl flag if in MHC region.
        """
        region = GenomicRegion.from_known_genomic_region(KnownGenomicRegions.MHC)
        self.df = self.df.withColumn(
            "qualityControls",
            self.update_quality_flag(
                f.col("qualityControls"),
                (
                    (f.col("chromosome") == region.chromosome)
                    & (
                        (f.col("position") <= region.end)
                        & (f.col("position") >= region.start)
                    )
                ),
                StudyLocusQualityCheck.IN_MHC,
            ),
        )
        return self

    def qc_redundant_top_hits_from_PICS(self: StudyLocus) -> StudyLocus:
        """Flag associations from top hits when the study contains other PICS associations from summary statistics.

        This flag can be useful to identify top hits that should be explained by other associations in the study derived from the summary statistics.

        Returns:
            StudyLocus: Updated study locus with redundant top hits flagged.
        """
        studies_with_pics_sumstats = (
            self.df.filter(f.col("finemappingMethod") == FinemappingMethod.PICS.value)
            # Returns True if the study contains any PICS associations from summary statistics
            .withColumn(
                "hasPicsSumstats",
                ~f.array_contains(
                    "qualityControls", StudyLocusQualityCheck.TOP_HIT.value
                ),
            )
            .groupBy("studyId")
            .agg(f.max(f.col("hasPicsSumstats")).alias("studiesWithPicsSumstats"))
        )

        return StudyLocus(
            _df=self.df.join(studies_with_pics_sumstats, on="studyId", how="left")
            .withColumn(
                "qualityControls",
                self.update_quality_flag(
                    f.col("qualityControls"),
                    f.array_contains(
                        "qualityControls", StudyLocusQualityCheck.TOP_HIT.value
                    )
                    & f.col("studiesWithPicsSumstats"),
                    StudyLocusQualityCheck.REDUNDANT_PICS_TOP_HIT,
                ),
            )
            .drop("studiesWithPicsSumstats"),
            _schema=StudyLocus.get_schema(),
        )

    def qc_explained_by_SuSiE(self: StudyLocus) -> StudyLocus:
        """Flag associations that are explained by SuSiE associations.

        Credible sets overlapping in the same region as a SuSiE credible set are flagged as explained by SuSiE.

        Returns:
            StudyLocus: Updated study locus with SuSiE explained flags.
        """
        # unique study-regions covered by SuSie credible sets
        susie_study_regions = (
            self.filter(
                f.col("finemappingMethod").isin(
                    FinemappingMethod.SUSIE.value, FinemappingMethod.SUSIE_INF.value
                )
            )
            .df.select(
                "studyId",
                "chromosome",
                "locusStart",
                "locusEnd",
                f.lit(True).alias("inSuSiE"),
            )
            .distinct()
        )

        # non SuSiE credible sets (studyLocusId) overlapping in any variant with SuSiE locus
        redundant_study_locus = (
            self.filter(
                ~f.col("finemappingMethod").isin(
                    FinemappingMethod.SUSIE.value, FinemappingMethod.SUSIE_INF.value
                )
            )
            .df.withColumn("l", f.explode("locus"))
            .select(
                "studyLocusId",
                "studyId",
                "chromosome",
                f.split(f.col("l.variantId"), "_")[1].alias("tag_position"),
            )
            .alias("study_locus")
            .join(
                susie_study_regions.alias("regions"),
                how="inner",
                on=[
                    (f.col("study_locus.chromosome") == f.col("regions.chromosome"))
                    & (f.col("study_locus.studyId") == f.col("regions.studyId"))
                    & (f.col("study_locus.tag_position") >= f.col("regions.locusStart"))
                    & (f.col("study_locus.tag_position") <= f.col("regions.locusEnd"))
                ],
            )
            .select("studyLocusId", "inSuSiE")
            .distinct()
        )

        return StudyLocus(
            _df=(
                self.df.join(redundant_study_locus, on="studyLocusId", how="left")
                .withColumn(
                    "qualityControls",
                    self.update_quality_flag(
                        f.col("qualityControls"),
                        # credible set in SuSiE overlapping region
                        f.col("inSuSiE")
                        # credible set not based on SuSiE
                        & (
                            ~f.col("finemappingMethod").isin(
                                FinemappingMethod.SUSIE.value,
                                FinemappingMethod.SUSIE_INF.value,
                            )
                        ),
                        StudyLocusQualityCheck.EXPLAINED_BY_SUSIE,
                    ),
                )
                .drop("inSuSiE")
            ),
            _schema=StudyLocus.get_schema(),
        )

    def _qc_no_population(self: StudyLocus) -> StudyLocus:
        """Flag associations where the study doesn't have population information to resolve LD.

        Returns:
            StudyLocus: Updated study locus.
        """
        # If the tested column is not present, return self unchanged:
        if "ldPopulationStructure" not in self.df.columns:
            return self

        self.df = self.df.withColumn(
            "qualityControls",
            self.update_quality_flag(
                f.col("qualityControls"),
                f.col("ldPopulationStructure").isNull(),
                StudyLocusQualityCheck.NO_POPULATION,
            ),
        )
        return self

    def annotate_locus_statistics_boundaries(
        self: StudyLocus,
        summary_statistics: SummaryStatistics,
    ) -> StudyLocus:
        """Annotates study locus with summary statistics in the specified boundaries - locusStart and locusEnd.

        Args:
            summary_statistics (SummaryStatistics): Summary statistics to be used for annotation.

        Returns:
            StudyLocus: Study locus annotated with summary statistics in `locus` column. If no statistics are found, the `locus` column will be empty.
        """
        # The clumps will be used several times (persisting)
        self.df.persist()
        # Renaming columns:
        sumstats_renamed = summary_statistics.df.selectExpr(
            *[f"{col} as tag_{col}" for col in summary_statistics.df.columns]
        ).alias("sumstat")

        locus_df = (
            sumstats_renamed
            # Joining the two datasets together:
            .join(
                f.broadcast(
                    self.df.alias("clumped").select(
                        "position",
                        "chromosome",
                        "studyId",
                        "studyLocusId",
                        "locusStart",
                        "locusEnd",
                    )
                ),
                on=[
                    (f.col("sumstat.tag_studyId") == f.col("clumped.studyId"))
                    & (f.col("sumstat.tag_chromosome") == f.col("clumped.chromosome"))
                    & (f.col("sumstat.tag_position") >= (f.col("clumped.locusStart")))
                    & (f.col("sumstat.tag_position") <= (f.col("clumped.locusEnd")))
                ],
                how="inner",
            )
            .withColumn(
                "locus",
                f.struct(
                    f.col("tag_variantId").alias("variantId"),
                    f.col("tag_beta").alias("beta"),
                    f.col("tag_pValueMantissa").alias("pValueMantissa"),
                    f.col("tag_pValueExponent").alias("pValueExponent"),
                    f.col("tag_standardError").alias("standardError"),
                ),
            )
            .groupBy("studyLocusId")
            .agg(
                f.collect_list(f.col("locus")).alias("locus"),
            )
        )

        self.df = self.df.drop("locus").join(
            locus_df,
            on="studyLocusId",
            how="left",
        )

        return self

    def window_based_clumping(
        self: StudyLocus,
        window_size: int = WindowBasedClumpingStepConfig().distance,
    ) -> StudyLocus:
        """Clump study locus by window size.

        Args:
            window_size (int): Window size for clumping.

        Returns:
            StudyLocus: Clumped study locus, where clumped associations are flagged.
        """
        from gentropy.method.window_based_clumping import WindowBasedClumping

        return WindowBasedClumping.clump(self, window_size)

    def assign_confidence(self: StudyLocus) -> StudyLocus:
        """Assign confidence to study locus.

        Returns:
            StudyLocus: Study locus with confidence assigned.
        """
        # Return self if the required columns are not in the dataframe:
        if (
            "qualityControls" not in self.df.columns
            or "finemappingMethod" not in self.df.columns
        ):
            return self

        # Assign confidence based on the presence of quality controls
        df = self.df.withColumn(
            "confidence",
            f.when(
                (
                    f.col("finemappingMethod").isin(
                        FinemappingMethod.SUSIE.value,
                        FinemappingMethod.SUSIE_INF.value,
                    )
                )
                & (
                    ~f.array_contains(
                        f.col("qualityControls"),
                        StudyLocusQualityCheck.OUT_OF_SAMPLE_LD.value,
                    )
                ),
                CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD.value,
            )
            .when(
                (
                    f.col("finemappingMethod").isin(
                        FinemappingMethod.SUSIE.value,
                        FinemappingMethod.SUSIE_INF.value,
                    )
                )
                & (
                    f.array_contains(
                        f.col("qualityControls"),
                        StudyLocusQualityCheck.OUT_OF_SAMPLE_LD.value,
                    )
                ),
                CredibleSetConfidenceClasses.FINEMAPPED_OUT_OF_SAMPLE_LD.value,
            )
            .when(
                (f.col("finemappingMethod") == FinemappingMethod.PICS.value)
                & (
                    ~f.array_contains(
                        f.col("qualityControls"), StudyLocusQualityCheck.TOP_HIT.value
                    )
                ),
                CredibleSetConfidenceClasses.PICSED_SUMMARY_STATS.value,
            )
            .when(
                (f.col("finemappingMethod") == FinemappingMethod.PICS.value)
                & (
                    f.array_contains(
                        f.col("qualityControls"), StudyLocusQualityCheck.TOP_HIT.value
                    )
                ),
                CredibleSetConfidenceClasses.PICSED_TOP_HIT.value,
            )
            .otherwise(CredibleSetConfidenceClasses.UNKNOWN.value),
        )

        return StudyLocus(
            _df=df,
            _schema=self.get_schema(),
        )
