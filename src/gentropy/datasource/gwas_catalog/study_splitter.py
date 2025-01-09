"""Splitting multi-trait GWAS Catalog studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql.window import Window

if TYPE_CHECKING:
    from pyspark.sql import Column

    from gentropy.datasource.gwas_catalog.associations import StudyLocusGWASCatalog
    from gentropy.datasource.gwas_catalog.study_index import StudyIndexGWASCatalog


class GWASCatalogStudySplitter:
    """Splitting multi-trait GWAS Catalog studies."""

    @staticmethod
    def _resolve_trait(
        study_trait: Column, association_trait: Column, p_value_text: Column
    ) -> Column:
        """Resolve trait names by consolidating association-level and study-level trait names.

        Args:
            study_trait (Column): Study-level trait name.
            association_trait (Column): Association-level trait name.
            p_value_text (Column): P-value text.

        Returns:
            Column: Resolved trait name.
        """
        return (
            f.when(
                (p_value_text.isNotNull()) & (p_value_text != ("no_pvalue_text")),
                f.concat(
                    association_trait,
                    f.lit(" ["),
                    p_value_text,
                    f.lit("]"),
                ),
            )
            .when(
                association_trait.isNotNull(),
                association_trait,
            )
            .otherwise(study_trait)
        )

    @staticmethod
    def _resolve_efo(association_efo: Column, study_efo: Column) -> Column:
        """Resolve EFOs by consolidating association-level and study-level EFOs.

        Args:
            association_efo (Column): EFO column from the association table.
            study_efo (Column): EFO column from the study table.

        Returns:
            Column: Consolidated EFO column.
        """
        return f.coalesce(f.split(association_efo, r"\/"), study_efo)

    @staticmethod
    def _resolve_study_id(study_id: Column, sub_study_description: Column) -> Column:
        """Resolve study IDs by exploding association-level information (e.g. pvalue_text, EFO).

        Args:
            study_id (Column): Study ID column.
            sub_study_description (Column): Sub-study description column from the association table.

        Returns:
            Column: Resolved study ID column.
        """
        split_w = Window.partitionBy(study_id).orderBy(sub_study_description)
        row_number = f.dense_rank().over(split_w)
        substudy_count = f.approx_count_distinct(row_number).over(split_w)
        return f.when(substudy_count == 1, study_id).otherwise(
            f.concat_ws("_", study_id, row_number)
        )

    @classmethod
    def split(
        cls: type[GWASCatalogStudySplitter],
        studies: StudyIndexGWASCatalog,
        associations: StudyLocusGWASCatalog,
    ) -> tuple[StudyIndexGWASCatalog, StudyLocusGWASCatalog]:
        """Splitting multi-trait GWAS Catalog studies.

        If assigned disease of the study and the association don't agree, we assume the study needs to be split.
        Then disease EFOs, trait names and study ID are consolidated

        Args:
            studies (StudyIndexGWASCatalog): GWAS Catalog studies.
            associations (StudyLocusGWASCatalog): GWAS Catalog associations.

        Returns:
            tuple[StudyIndexGWASCatalog, StudyLocusGWASCatalog]: Split studies and associations.
        """
        # Composite of studies and associations to resolve scattered information
        st_ass = (
            associations.df.join(f.broadcast(studies.df), on="studyId", how="inner")
            .select(
                "studyId",
                "subStudyDescription",
                cls._resolve_study_id(
                    f.col("studyId"), f.col("subStudyDescription")
                ).alias("updatedStudyId"),
                cls._resolve_trait(
                    f.col("traitFromSource"),
                    f.split("subStudyDescription", r"\|").getItem(0),
                    f.split("subStudyDescription", r"\|").getItem(1),
                ).alias("traitFromSource"),
                cls._resolve_efo(
                    f.split("subStudyDescription", r"\|").getItem(2),
                    f.col("traitFromSourceMappedIds"),
                ).alias("traitFromSourceMappedIds"),
            )
            .persist()
        )

        return (
            studies.update_study_id(
                st_ass.select(
                    "studyId",
                    "updatedStudyId",
                    "traitFromSource",
                    "traitFromSourceMappedIds",
                ).distinct()
            ),
            associations.update_study_id(
                st_ass.select(
                    "updatedStudyId", "studyId", "subStudyDescription"
                ).distinct()
            )
            .qc_ambiguous_study()
            .qc_flag_all_tophits(),
        )
