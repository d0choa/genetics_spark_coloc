"""Study index dataset."""
from __future__ import annotations

import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

from pyspark.sql import functions as f

from otg.assets import data
from otg.common.schemas import parse_spark_schema
from otg.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.types import StructType


@dataclass
class StudyIndex(Dataset):
    """Study index dataset.

    A study index dataset captures all the metadata for all studies including GWAS and Molecular QTL.
    """

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

    def get_eligible_gwas_study_ids(self: StudyIndex) -> list[str]:
        """Get a list of study identifiers eligible for ingestion.

        Currently the eligibility criteria is to have `gwas` as study type and have no quality controls flag.

        Returns:
            list[str]: list of study identifiers.
        """
        return [
            row["studyId"]
            for row in self.df.filter(
                (f.col("studyType") == "gwas") & (f.size(f.col("qualityControls")) == 0)
            )
            .distinct()
            .collect()
        ]

    def is_qtl(self: StudyIndex) -> Column:
        """Return booleans indicating if a study is a qtl.

        Returns:
            Column: Boolean columns with true values for QTL studies.
        """
        return f.when(self.df.studyType.endswith("qtl"), f.lit(True)).otherwise(
            f.lit(False)
        )

    def is_gwas(self: StudyIndex) -> Column:
        """Return booleans indicating if a study is a gwas.

        Returns:
            Column: Boolean columns with true values for GWAS studies.
        """
        return f.when(self.df.studyType == "gwas", f.lit(True)).otherwise(f.lit(False))

    def has_mapped_trait(self: StudyIndex) -> Column:
        """Return booleans indicating if a study has mapped disease.

        Returns:
            Column: Boolean columns with true values studies with mapped disease (EFO).
        """
        return f.when(
            f.size(self.df.traitFromSourceMappedIds) > 0, f.lit(True)
        ).otherwise(f.lit(False))

    def is_quality_flagged(self: StudyIndex) -> Column:
        """Return booleans indicating if a study flagged due to quality issues.

        Returns:
            Column: Boolean columns with true values with no quality issues.
        """
        return f.when(f.size(self.df.qualityControls) == 0, f.lit(True)).otherwise(
            f.lit(False)
        )

    def has_summarystats(self: StudyIndex) -> Column:
        """Return booleans indicating if a study has summary harmonized summary statistics.

        Returns:
            Column: Boolean columns with true values for studies with summary statistics.
        """
        return f.when(self.df.hasSumstats, f.lit(True)).otherwise(f.lit(False))
