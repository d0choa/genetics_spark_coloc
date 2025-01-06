"""Study index dataset."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING

from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, StringType, StructType
from pyspark.sql.window import Window

from gentropy.assets import data
from gentropy.common.schemas import parse_spark_schema
from gentropy.common.spark_helpers import convert_from_wide_to_long
from gentropy.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.types import StructType

    from gentropy.dataset.biosample_index import BiosampleIndex
    from gentropy.dataset.target_index import TargetIndex


class StudyQualityCheck(Enum):
    """Study quality control options listing concerns on the quality of the study.

    Attributes:
        UNRESOLVED_TARGET (str): Target/gene identifier could not match to reference - Labelling failing target.
        UNRESOLVED_DISEASE (str): Disease identifier could not match to referece or retired identifier - labelling failing disease
        UNKNOWN_STUDY_TYPE (str): Indicating the provided type of study is not supported.
        UNKNOWN_BIOSAMPLE (str): Flagging if a biosample identifier is not found in the reference.
        DUPLICATED_STUDY (str): Flagging if a study identifier is not unique.
        SUMSTATS_NOT_AVAILABLE (str): Flagging if harmonized summary statistics are not available or empty.
        NO_OT_CURATION (str): Flagging if a study has not been curated by Open Targets.
        FAILED_MEAN_BETA_CHECK (str): Flagging if the mean beta QC check value is not within the expected range.
        FAILED_PZ_CHECK (str): Flagging if the PZ QC check values are not within the expected range.
        FAILED_GC_LAMBDA_CHECK (str): Flagging if the GC lambda value is not within the expected range.
        SMALL_NUMBER_OF_SNPS (str): Flagging if the number of SNPs in the study is below the expected threshold.
    """

    UNRESOLVED_TARGET = "Target/gene identifier could not match to reference"
    UNRESOLVED_DISEASE = "No valid disease identifier found"
    UNKNOWN_STUDY_TYPE = "This type of study is not supported"
    UNKNOWN_BIOSAMPLE = "Biosample identifier was not found in the reference"
    DUPLICATED_STUDY = "The identifier of this study is not unique"
    SUMSTATS_NOT_AVAILABLE = "Harmonized summary statistics are not available or empty"
    NO_OT_CURATION = "GWAS Catalog study has not been curated by Open Targets"
    FAILED_MEAN_BETA_CHECK = (
        "The mean beta QC check value is not within the expected range"
    )
    FAILED_PZ_CHECK = "The PZ QC check values are not within the expected range"
    FAILED_GC_LAMBDA_CHECK = "The GC lambda value is not within the expected range"
    SMALL_NUMBER_OF_SNPS = (
        "The number of SNPs in the study is below the expected threshold"
    )


@dataclass
class StudyIndex(Dataset):
    """Study index dataset.

    A study index dataset captures all the metadata for all studies including GWAS and Molecular QTL.
    """

    VALID_TYPES = [
        "gwas",
        "eqtl",
        "pqtl",
        "sqtl",
        "tuqtl",
        "sceqtl",
        "scpqtl",
        "scsqtl",
        "sctuqtl",
    ]

    @staticmethod
    def _aggregate_samples_by_ancestry(merged: Column, ancestry: Column) -> Column:
        """Aggregate sample counts by ancestry in a list of struct colmns.

        Args:
            merged (Column): A column representing merged data (list of structs).
            ancestry (Column): The `ancestry` parameter is a column that represents the ancestry of each
                sample. (a struct)

        Returns:
            Column: the modified "merged" column after aggregating the samples by ancestry.
        """
        # Iterating over the list of ancestries and adding the sample size if label matches:
        return f.transform(
            merged,
            lambda a: f.when(
                a.ancestry == ancestry.ancestry,
                f.struct(
                    a.ancestry.alias("ancestry"),
                    (a.sampleSize + ancestry.sampleSize).alias("sampleSize"),
                ),
            ).otherwise(a),
        )

    @staticmethod
    def _map_ancestries_to_ld_population(gwas_ancestry_label: Column) -> Column:
        """Normalise ancestry column from GWAS studies into reference LD panel based on a pre-defined map.

        This function assumes all possible ancestry categories have a corresponding
        LD panel in the LD index. It is very important to have the ancestry labels
        moved to the LD panel map.

        Args:
            gwas_ancestry_label (Column): A struct column with ancestry label like Finnish,
                European, African etc. and the corresponding sample size.

        Returns:
            Column: Struct column with the mapped LD population label and the sample size.
        """
        # Loading ancestry label to LD population label:
        json_dict = json.loads(
            pkg_resources.read_text(
                data, "gwas_population_2_LD_panel_map.json", encoding="utf-8"
            )
        )
        map_expr = f.create_map(*[f.lit(x) for x in chain(*json_dict.items())])

        return f.struct(
            map_expr[gwas_ancestry_label.ancestry].alias("ancestry"),
            gwas_ancestry_label.sampleSize.alias("sampleSize"),
        )

    @classmethod
    def get_schema(cls: type[StudyIndex]) -> StructType:
        """Provide the schema for the StudyIndex dataset.

        Returns:
            StructType: The schema of the StudyIndex dataset.
        """
        return parse_spark_schema("study_index.json")

    @classmethod
    def get_QC_column_name(cls: type[StudyIndex]) -> str:
        """Return the name of the quality control column.

        Returns:
            str: The name of the quality control column.
        """
        return "qualityControls"

    @classmethod
    def get_QC_mappings(cls: type[StudyIndex]) -> dict[str, str]:
        """Quality control flag to QC column category mappings.

        Returns:
            dict[str, str]: Mapping between flag name and QC column category value.
        """
        return {member.name: member.value for member in StudyQualityCheck}

    @classmethod
    def aggregate_and_map_ancestries(
        cls: type[StudyIndex], discovery_samples: Column
    ) -> Column:
        """Map ancestries to populations in the LD reference and calculate relative sample size.

        Args:
            discovery_samples (Column): A list of struct column. Has an `ancestry` column and a `sampleSize` columns

        Returns:
            Column: A list of struct with mapped LD population and their relative sample size.
        """
        # Map ancestry categories to population labels of the LD index:
        mapped_ancestries = f.transform(
            discovery_samples, cls._map_ancestries_to_ld_population
        )

        # Aggregate sample sizes belonging to the same LD population:
        aggregated_counts = f.aggregate(
            mapped_ancestries,
            f.array_distinct(
                f.transform(
                    mapped_ancestries,
                    lambda x: f.struct(
                        x.ancestry.alias("ancestry"), f.lit(0.0).alias("sampleSize")
                    ),
                )
            ),
            cls._aggregate_samples_by_ancestry,
        )
        # Getting total sample count:
        total_sample_count = f.aggregate(
            aggregated_counts, f.lit(0.0), lambda total, pop: total + pop.sampleSize
        ).alias("sampleSize")

        # Calculating relative sample size for each LD population:
        return f.transform(
            aggregated_counts,
            lambda ld_population: f.struct(
                ld_population.ancestry.alias("ldPopulation"),
                (ld_population.sampleSize / total_sample_count).alias(
                    "relativeSampleSize"
                ),
            ),
        )

    def study_type_lut(self: StudyIndex) -> DataFrame:
        """Return a lookup table of study type.

        Returns:
            DataFrame: A dataframe containing `studyId` and `studyType` columns.
        """
        return self.df.select("studyId", "studyType")

    def is_qtl(self: StudyIndex) -> Column:
        """Return a boolean column with true values for QTL studies.

        Returns:
            Column: True if the study is a QTL study.
        """
        return self.df.studyType.endswith("qtl")

    def is_gwas(self: StudyIndex) -> Column:
        """Return a boolean column with true values for GWAS studies.

        Returns:
            Column: True if the study is a GWAS study.
        """
        return self.df.studyType == "gwas"

    def has_mapped_trait(self: StudyIndex) -> Column:
        """Return a boolean column indicating if a study has mapped disease.

        Returns:
            Column: True if the study has mapped disease.
        """
        return f.size(self.df.traitFromSourceMappedIds) > 0

    def is_quality_flagged(self: StudyIndex) -> Column:
        """Return a boolean column indicating if a study is flagged due to quality issues.

        Returns:
            Column: True if the study is flagged.
        """
        # Testing for the presence of the qualityControls column:
        if "qualityControls" not in self.df.columns:
            return f.lit(False)
        else:
            return f.size(self.df["qualityControls"]) != 0

    def has_summarystats(self: StudyIndex) -> Column:
        """Return a boolean column indicating if a study has harmonized summary statistics.

        Returns:
            Column: True if the study has harmonized summary statistics.
        """
        return self.df.hasSumstats

    def validate_unique_study_id(self: StudyIndex) -> StudyIndex:
        """Validating the uniqueness of study identifiers and flagging duplicated studies.

        Returns:
            StudyIndex: with flagged duplicated studies.
        """
        return StudyIndex(
            _df=self.df.withColumn(
                "qualityControls",
                self.update_quality_flag(
                    f.col("qualityControls"),
                    self.flag_duplicates(f.col("studyId")),
                    StudyQualityCheck.DUPLICATED_STUDY,
                ),
            ),
            _schema=StudyIndex.get_schema(),
        )

    def _normalise_disease(
        self: StudyIndex,
        source_disease_column_name: str,
        disease_column_name: str,
        disease_map: DataFrame,
    ) -> DataFrame:
        """Normalising diseases in the study index.

        Given a reference disease map (containing all potential EFO ids with the corresponding reference disease ids),
        this function maps all EFO ids in the study index to the reference disease ids.

        Args:
            source_disease_column_name (str): The column name of the disease column to validate.
            disease_column_name (str): The resulting disease column name that contains the validated ids.
            disease_map (DataFrame): Reference dataframe with diseases

        Returns:
            DataFrame: where the newly added diseaseIds column will contain the validated EFO identifiers.
        """
        return (
            self.df
            # Only validating studies with diseases:
            .filter(f.size(f.col(source_disease_column_name)) > 0)
            # Explode disease column:
            .select(
                "studyId",
                "studyType",
                f.explode_outer(source_disease_column_name).alias("efo"),
            )
            # Join disease map:
            .join(disease_map, on="efo", how="left")
            .groupBy("studyId")
            .agg(
                f.collect_set(f.col("diseaseId")).alias(disease_column_name),
            )
        )

    def validate_disease(self: StudyIndex, disease_map: DataFrame) -> StudyIndex:
        """Validate diseases in the study index dataset.

        Args:
            disease_map (DataFrame): a dataframe with two columns (efo, diseaseId).

        Returns:
            StudyIndex: where gwas studies are flagged where no valid disease id could be found.
        """
        # Because the disease ids are not mandatory fields of the schema, we skip vaildation if these columns are not present:
        if ("traitFromSourceMappedIds" not in self.df.columns) or (
            "backgroundTraitFromSourceMappedIds" not in self.df.columns
        ):
            return self

        # Disease Column names:
        foreground_disease_column = "diseaseIds"
        background_disease_column = "backgroundDiseaseIds"

        # If diseaseId in schema, we need to drop it:
        drop_columns = [
            column
            for column in self.df.columns
            if column in [foreground_disease_column, background_disease_column]
        ]

        if len(drop_columns) > 0:
            self.df = self.df.drop(*drop_columns)

        # Normalise disease:
        normalised_disease = self._normalise_disease(
            "traitFromSourceMappedIds", foreground_disease_column, disease_map
        )
        normalised_background_disease = self._normalise_disease(
            "backgroundTraitFromSourceMappedIds", background_disease_column, disease_map
        )

        return StudyIndex(
            _df=(
                self.df.join(normalised_disease, on="studyId", how="left")
                .join(normalised_background_disease, on="studyId", how="left")
                # Updating disease columns:
                .withColumn(
                    foreground_disease_column,
                    f.when(
                        f.col(foreground_disease_column).isNull(), f.array()
                    ).otherwise(f.col(foreground_disease_column)),
                )
                .withColumn(
                    background_disease_column,
                    f.when(
                        f.col(background_disease_column).isNull(), f.array()
                    ).otherwise(f.col(background_disease_column)),
                )
                # Flagging gwas studies where no valid disease is avilable:
                .withColumn(
                    "qualityControls",
                    StudyIndex.update_quality_flag(
                        f.col("qualityControls"),
                        # Flagging all gwas studies with no normalised disease:
                        (f.size(f.col(foreground_disease_column)) == 0)
                        & (f.col("studyType") == "gwas"),
                        StudyQualityCheck.UNRESOLVED_DISEASE,
                    ),
                )
            ),
            _schema=StudyIndex.get_schema(),
        )

    def validate_study_type(self: StudyIndex) -> StudyIndex:
        """Validating study type and flag unsupported types.

        Returns:
            StudyIndex: with flagged studies with unsupported type.
        """
        validated_df = (
            self.df
            # Flagging unsupported study types:
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    f.when(
                        (f.col("studyType") == "gwas")
                        | f.col("studyType").endswith("qtl"),
                        False,
                    ).otherwise(True),
                    StudyQualityCheck.UNKNOWN_STUDY_TYPE,
                ),
            )
        )
        return StudyIndex(_df=validated_df, _schema=StudyIndex.get_schema())

    def validate_target(self: StudyIndex, target_index: TargetIndex) -> StudyIndex:
        """Validating gene identifiers in the study index against the provided target index.

        Args:
            target_index (TargetIndex): target index containing the reference gene identifiers (Ensembl gene identifiers).

        Returns:
            StudyIndex: with flagged studies if geneId could not be validated.
        """
        gene_set = target_index.df.select("geneId", f.lit(True).alias("isIdFound"))

        # As the geneId is not a mandatory field of study index, we return if the column is not there:
        if "geneId" not in self.df.columns:
            return self

        validated_df = (
            self.df.join(gene_set, on="geneId", how="left")
            .withColumn(
                "isIdFound",
                f.when(
                    (f.col("studyType") != "gwas") & f.col("isIdFound").isNull(),
                    f.lit(False),
                ).otherwise(f.lit(True)),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~f.col("isIdFound"),
                    StudyQualityCheck.UNRESOLVED_TARGET,
                ),
            )
            .drop("isIdFound")
        )

        return StudyIndex(_df=validated_df, _schema=StudyIndex.get_schema())

    def validate_biosample(
        self: StudyIndex, biosample_index: BiosampleIndex
    ) -> StudyIndex:
        """Validating biosample identifiers in the study index against the provided biosample index.

        Args:
            biosample_index (BiosampleIndex): Biosample index containing a reference of biosample identifiers e.g. cell types, tissues, cell lines, etc.

        Returns:
            StudyIndex: where non-gwas studies are flagged if biosampleIndex could not be validated.
        """
        biosample_set = biosample_index.df.select(
            "biosampleId", f.lit(True).alias("isIdFound")
        )

        # If biosampleId in df, we need to drop it:
        if "biosampleId" in self.df.columns:
            self.df = self.df.drop("biosampleId")

        # As the biosampleFromSourceId is not a mandatory field of study index, we return if the column is not there:
        if "biosampleFromSourceId" not in self.df.columns:
            return self

        validated_df = (
            self.df.join(
                biosample_set,
                self.df.biosampleFromSourceId == biosample_set.biosampleId,
                how="left",
            )
            .withColumn(
                "isIdFound",
                f.when(
                    (f.col("studyType") != "gwas") & (f.col("isIdFound").isNull()),
                    f.lit(False),
                ).otherwise(f.lit(True)),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~f.col("isIdFound"),
                    StudyQualityCheck.UNKNOWN_BIOSAMPLE,
                ),
            )
            .drop("isIdFound")
        )

        return StudyIndex(_df=validated_df, _schema=StudyIndex.get_schema())

    def annotate_sumstats_qc(
        self: StudyIndex,
        sumstats_qc: DataFrame,
        threshold_mean_beta: float = 0.05,
        threshold_mean_diff_pz: float = 0.05,
        threshold_se_diff_pz: float = 0.05,
        threshold_min_gc_lambda: float = 0.7,
        threshold_max_gc_lambda: float = 2.5,
        threshold_min_n_variants: int = 2_000_000,
    ) -> StudyIndex:
        """Annotate summary stats QC information.

        Args:
            sumstats_qc (DataFrame): containing summary statistics-based quality controls.
            threshold_mean_beta (float): Threshold for mean beta check. Defaults to 0.05.
            threshold_mean_diff_pz (float): Threshold for mean diff PZ check. Defaults to 0.05.
            threshold_se_diff_pz (float): Threshold for SE diff PZ check. Defaults to 0.05.
            threshold_min_gc_lambda (float): Minimum threshold for GC lambda check. Defaults to 0.7.
            threshold_max_gc_lambda (float): Maximum threshold for GC lambda check. Defaults to 2.5.
            threshold_min_n_variants (int): Minimum number of variants for SuSiE check. Defaults to 2_000_000.

        Returns:
            StudyIndex: Updated study index with QC information
        """
        # convert all columns in sumstats_qc dataframe in array of structs grouped by studyId
        cols = [c for c in sumstats_qc.columns if c != "studyId"]

        studies = self.df

        melted_df = convert_from_wide_to_long(
            sumstats_qc,
            id_vars=["studyId"],
            value_vars=cols,
            var_name="QCCheckName",
            value_name="QCCheckValue",
        )

        qc_df = (
            melted_df.groupBy("studyId")
            .agg(
                f.map_from_entries(
                    f.collect_list(
                        f.struct(f.col("QCCheckName"), f.col("QCCheckValue"))
                    )
                ).alias("sumStatQCValues")
            )
            .select("studyId", "sumstatQCValues")
        )

        df = (
            studies.drop("sumStatQCValues", "hasSumstats")
            .join(
                qc_df.withColumn("hasSumstats", f.lit(True)), how="left", on="studyId"
            )
            .withColumn("hasSumstats", f.coalesce(f.col("hasSumstats"), f.lit(False)))
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~f.col("hasSumstats"),
                    StudyQualityCheck.SUMSTATS_NOT_AVAILABLE,
                ),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~(f.abs(f.col("sumstatQCValues.mean_beta")) <= threshold_mean_beta),
                    StudyQualityCheck.FAILED_MEAN_BETA_CHECK,
                ),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~(
                        (
                            f.abs(f.col("sumstatQCValues.mean_diff_pz"))
                            <= threshold_mean_diff_pz
                        )
                        & (f.col("sumstatQCValues.se_diff_pz") <= threshold_se_diff_pz)
                    ),
                    StudyQualityCheck.FAILED_PZ_CHECK,
                ),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    ~(
                        (f.col("sumstatQCValues.gc_lambda") <= threshold_max_gc_lambda)
                        & (
                            f.col("sumstatQCValues.gc_lambda")
                            >= threshold_min_gc_lambda
                        )
                    ),
                    StudyQualityCheck.FAILED_GC_LAMBDA_CHECK,
                ),
            )
            .withColumn(
                "qualityControls",
                StudyIndex.update_quality_flag(
                    f.col("qualityControls"),
                    (f.col("sumstatQCValues.n_variants") < threshold_min_n_variants),
                    StudyQualityCheck.SMALL_NUMBER_OF_SNPS,
                ),
            )
        )

        # Annotate study index with QC information:
        return StudyIndex(
            _df=df,
            _schema=StudyIndex.get_schema(),
        )

    def deconvolute_studies(self: StudyIndex) -> StudyIndex:
        """Deconvolute the study index dataset.

        When ingesting the study index dataset, the same studyId might be ingested from more than one source.
        In such cases, the data needs to be merged and the quality control flags need to be combined.

        Returns:
            StudyIndex: Deconvoluted study index dataset.
        """
        # Windowing by study ID assume random order, but this is OK, because we are not selecting rows by a specific order.
        study_id_window = Window.partitionBy("studyId").orderBy(f.rand())

        # For certain aggregation, the full window is needed to be considered:
        full_study_id_window = study_id_window.orderBy("studyId").rangeBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

        # Temporary columns to drop at the end:
        columns_to_drop = ["keepTopHit", "mostGranular", "rank"]

        return StudyIndex(
            _df=(
                self.df
                # Initialising quality controls column, if not present:
                .withColumn(
                    "qualityControls",
                    f.when(
                        f.col("qualityControls").isNull(),
                        f.array().cast(ArrayType(StringType())),
                    ).otherwise(f.col("qualityControls")),
                )
                # Keeping top hit studies unless the same study is available from a summmary statistics source:
                # This value will be set for all rows for the same `studyId`:
                .withColumn(
                    "keepTopHit",
                    f.when(
                        f.array_contains(
                            f.collect_set(f.col("hasSumstats")).over(
                                full_study_id_window
                            ),
                            True,
                        ),
                        f.lit(False),
                    ).otherwise(True),
                )
                # For studies without summary statistics, we remove the "Not curated by Open Targets" flag:
                .withColumn(
                    "qualityControls",
                    f.when(
                        ~f.col("hasSumstats"),
                        f.array_remove(
                            f.col("qualityControls"),
                            StudyQualityCheck.NO_OT_CURATION.value,
                        ),
                    ).otherwise(f.col("qualityControls")),
                )
                # If top hits are not kept, we remove the "sumstats not available" flag from all QC lists:
                .withColumn(
                    "qualityControls",
                    f.when(
                        ~f.col("keepTopHit"),
                        f.array_remove(
                            f.col("qualityControls"),
                            StudyQualityCheck.SUMSTATS_NOT_AVAILABLE.value,
                        ),
                    ).otherwise(f.col("qualityControls")),
                )
                # Then propagate quality checks for all sources of the same study:
                .withColumn(
                    "qualityControls",
                    f.array_distinct(
                        f.flatten(
                            f.collect_set("qualityControls").over(full_study_id_window)
                        )
                    ),
                )
                # Propagating sumstatQCValues -> map, cannot be flatten:
                .withColumn(
                    "sumstatQCValues",
                    f.first("sumstatQCValues", ignorenulls=True).over(
                        full_study_id_window
                    ),
                )
                # Propagating analysisFlags:
                .withColumn(
                    "analysisFlags",
                    f.flatten(
                        f.collect_list("analysisFlags").over(full_study_id_window)
                    ),
                )
                # Propagating hasSumstatsFlag - if no flag, leave null:
                .withColumn(
                    "hasSumstats",
                    f.when(
                        # There's a true:
                        f.array_contains(
                            f.collect_set("hasSumstats").over(full_study_id_window),
                            True,
                        ),
                        f.lit(True),
                    ).when(
                        # There's a false:
                        f.array_contains(
                            f.collect_set("hasSumstats").over(full_study_id_window),
                            False,
                        ),
                        f.lit(False),
                    ),
                )
                # Propagating disease: when different sets of diseases available for the same study,
                # we pick the shortest list, becasuse we assume, that is the most accurate disease assignment:
                .withColumn(
                    "mostGranular",
                    f.size(f.col("traitFromSourceMappedIds"))
                    == f.min(f.size(f.col("traitFromSourceMappedIds"))).over(
                        full_study_id_window
                    ),
                )
                # Remove less granular disease mappings:
                .withColumn(
                    "traitFromSourceMappedIds",
                    f.when(f.col("mostGranular"), f.col("traitFromSourceMappedIds")),
                )
                # Propagate mapped disease:
                .withColumn(
                    "traitFromSourceMappedIds",
                    f.last(f.col("traitFromSourceMappedIds"), True).over(
                        full_study_id_window
                    ),
                )
                # Repeating these steps for the `traitFromSource` column:
                .withColumn(
                    "traitFromSource",
                    f.when(f.col("mostGranular"), f.col("traitFromSource")),
                )
                # Propagate disease:
                .withColumn(
                    "traitFromSource",
                    f.last(f.col("traitFromSource"), True).over(full_study_id_window),
                )
                # Distinct study types are joined together into a string. So, if there's ambiguite, the study will be flagged when the study type is validated:
                .withColumn(
                    "studyType",
                    f.concat_ws(
                        ",", f.collect_set("studyType").over(full_study_id_window)
                    ),
                )
                # At this point, all studies in one window is expected to be identical. Let's just pick one:
                .withColumn("rank", f.row_number().over(study_id_window))
                .filter(f.col("rank") == 1)
                .drop(*columns_to_drop)
            ),
            _schema=StudyIndex.get_schema(),
        )
