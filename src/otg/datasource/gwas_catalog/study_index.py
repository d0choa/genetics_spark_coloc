"""Study Index for GWAS Catalog data source."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from otg.common.spark_helpers import column2camel_case
from otg.common.utils import parse_efos
from otg.dataset.study_index import StudyIndex

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


@dataclass
class GWASCatalogStudyIndex(StudyIndex):
    """Study index from GWAS Catalog.

    The following information is harmonised from the GWAS Catalog:

    - All publication related information retained.
    - Mapped measured and background traits parsed.
    - Flagged if harmonized summary statistics datasets available.
    - If available, the ftp path to these files presented.
    - Ancestries from the discovery and replication stages are structured with sample counts.
    - Case/control counts extracted.
    - The number of samples with European ancestry extracted.

    """

    @staticmethod
    def _parse_discovery_samples(discovery_samples: Column) -> Column:
        """Parse discovery sample sizes from GWAS Catalog.

        This is a curated field. From publication sometimes it is not clear how the samples were split
        across the reported ancestries. In such cases we are assuming the ancestries were evenly presented
        and the total sample size is split:

        ["European, African", 100] -> ["European, 50], ["African", 50]

        Args:
            discovery_samples (Column): Raw discovery sample sizes

        Returns:
            Column: Parsed and de-duplicated list of discovery ancestries with sample size.

        Examples:
            >>> data = [('s1', "European", 10), ('s1', "African", 10), ('s2', "European, African, Asian", 100), ('s2', "European", 50)]
            >>> df = (
            ...    spark.createDataFrame(data, ['studyId', 'ancestry', 'sampleSize'])
            ...    .groupBy('studyId')
            ...    .agg(
            ...        f.collect_set(
            ...            f.struct('ancestry', 'sampleSize')
            ...        ).alias('discoverySampleSize')
            ...    )
            ...    .orderBy('studyId')
            ...    .withColumn('discoverySampleSize', GWASCatalogStudyIndex._parse_discovery_samples(f.col('discoverySampleSize')))
            ...    .select('discoverySampleSize')
            ...    .show(truncate=False)
            ... )
            +--------------------------------------------+
            |discoverySampleSize                         |
            +--------------------------------------------+
            |[{African, 10}, {European, 10}]             |
            |[{European, 83}, {African, 33}, {Asian, 33}]|
            +--------------------------------------------+
            <BLANKLINE>
        """
        # To initialize return objects for aggregate functions, schema has to be definied:
        schema = t.ArrayType(
            t.StructType(
                [
                    t.StructField("ancestry", t.StringType(), True),
                    t.StructField("sampleSize", t.IntegerType(), True),
                ]
            )
        )

        # Splitting comma separated ancestries:
        exploded_ancestries = f.transform(
            discovery_samples,
            lambda sample: f.split(sample.ancestry, r",\s(?![^()]*\))"),
        )

        # Initialize discoverySample object from unique list of ancestries:
        unique_ancestries = f.transform(
            f.aggregate(
                exploded_ancestries,
                f.array().cast(t.ArrayType(t.StringType())),
                lambda x, y: f.array_union(x, y),
                f.array_distinct,
            ),
            lambda ancestry: f.struct(
                ancestry.alias("ancestry"),
                f.lit(0).cast(t.LongType()).alias("sampleSize"),
            ),
        )

        # Computing sample sizes for ancestries when splitting is needed:
        resolved_sample_count = f.transform(
            f.arrays_zip(
                f.transform(exploded_ancestries, lambda pop: f.size(pop)).alias(
                    "pop_size"
                ),
                f.transform(discovery_samples, lambda pop: pop.sampleSize).alias(
                    "pop_count"
                ),
            ),
            lambda pop: (pop.pop_count / pop.pop_size).cast(t.IntegerType()),
        )

        # Flattening out ancestries with sample sizes:
        parsed_sample_size = f.aggregate(
            f.transform(
                f.arrays_zip(
                    exploded_ancestries.alias("ancestries"),
                    resolved_sample_count.alias("sample_count"),
                ),
                GWASCatalogStudyIndex._merge_ancestries_and_counts,
            ),
            f.array().cast(schema),
            lambda x, y: f.array_union(x, y),
        )

        # Normalize ancestries:
        return f.aggregate(
            parsed_sample_size,
            unique_ancestries,
            GWASCatalogStudyIndex._normalize_ancestries,
        )

    @staticmethod
    def _normalize_ancestries(merged: Column, ancestry: Column) -> Column:
        """Normalize ancestries from a list of structs.

        As some ancestry label might be repeated with different sample counts,
        these counts need to be collected.

        Args:
            merged (Column): Resulting list of struct with unique ancestries.
            ancestry (Column): One ancestry object coming from raw.

        Returns:
            Column: Unique list of ancestries with the sample counts.
        """
        # Iterating over the list of unique ancestries and adding the sample size if label matches:
        return f.transform(
            merged,
            lambda a: f.when(
                a.ancestry == ancestry.ancestry,
                f.struct(
                    a.ancestry.alias("ancestry"),
                    (a.sampleSize + ancestry.sampleSize)
                    .cast(t.LongType())
                    .alias("sampleSize"),
                ),
            ).otherwise(a),
        )

    @staticmethod
    def _merge_ancestries_and_counts(ancestry_group: Column) -> Column:
        """Merge ancestries with sample sizes.

        After splitting ancestry annotations, all resulting ancestries needs to be assigned
        with the proper sample size.

        Args:
            ancestry_group (Column): Each element is a struct with `sample_count` (int) and `ancestries` (list)

        Returns:
            Column: a list of structs with `ancestry` and `sampleSize` fields.

        Examples:
            >>> data = [(12, ['African', 'European']),(12, ['African'])]
            >>> (
            ...     spark.createDataFrame(data, ['sample_count', 'ancestries'])
            ...     .select(GWASCatalogStudyIndex._merge_ancestries_and_counts(f.struct('sample_count', 'ancestries')).alias('test'))
            ...     .show(truncate=False)
            ... )
            +-------------------------------+
            |test                           |
            +-------------------------------+
            |[{African, 12}, {European, 12}]|
            |[{African, 12}]                |
            +-------------------------------+
            <BLANKLINE>
        """
        # Extract sample size for the ancestry group:
        count = ancestry_group.sample_count

        # We need to loop through the ancestries:
        return f.transform(
            ancestry_group.ancestries,
            lambda ancestry: f.struct(
                ancestry.alias("ancestry"),
                count.alias("sampleSize"),
            ),
        )

    @classmethod
    def _parse_study_table(
        cls: type[GWASCatalogStudyIndex], catalog_studies: DataFrame
    ) -> GWASCatalogStudyIndex:
        """Harmonise GWASCatalog study table with `StudyIndex` schema.

        Args:
            catalog_studies (DataFrame): GWAS Catalog study table

        Returns:
            GWASCatalogStudyIndex: Parsed and annotated GWAS Catalog study table.
        """
        return GWASCatalogStudyIndex(
            _df=catalog_studies.select(
                f.coalesce(
                    f.col("STUDY ACCESSION"), f.monotonically_increasing_id()
                ).alias("studyId"),
                f.lit("GCST").alias("projectId"),
                f.lit("gwas").alias("studyType"),
                f.col("PUBMED ID").alias("pubmedId"),
                f.col("FIRST AUTHOR").alias("publicationFirstAuthor"),
                f.col("DATE").alias("publicationDate"),
                f.col("JOURNAL").alias("publicationJournal"),
                f.col("STUDY").alias("publicationTitle"),
                f.coalesce(f.col("DISEASE/TRAIT"), f.lit("Unreported")).alias(
                    "traitFromSource"
                ),
                f.col("INITIAL SAMPLE SIZE").alias("initialSampleSize"),
                parse_efos(f.col("MAPPED_TRAIT_URI")).alias("traitFromSourceMappedIds"),
                parse_efos(f.col("MAPPED BACKGROUND TRAIT URI")).alias(
                    "backgroundTraitFromSourceMappedIds"
                ),
            ),
            _schema=GWASCatalogStudyIndex.get_schema(),
        )

    @classmethod
    def from_source(
        cls: type[GWASCatalogStudyIndex],
        catalog_studies: DataFrame,
        ancestry_file: DataFrame,
        sumstats_lut: DataFrame,
    ) -> StudyIndex:
        """Ingests study level metadata from the GWAS Catalog.

        Args:
            catalog_studies (DataFrame): GWAS Catalog raw study table
            ancestry_file (DataFrame): GWAS Catalog ancestry table.
            sumstats_lut (DataFrame): GWAS Catalog summary statistics list.

        Returns:
            StudyIndex: Parsed and annotated GWAS Catalog study table.
        """
        # Read GWAS Catalogue raw data
        return (
            cls._parse_study_table(catalog_studies)
            ._annotate_ancestries(ancestry_file)
            ._annotate_sumstats_info(sumstats_lut)
            ._annotate_discovery_sample_sizes()
        )

    def update_study_id(
        self: GWASCatalogStudyIndex, study_annotation: DataFrame
    ) -> GWASCatalogStudyIndex:
        """Update studyId with a dataframe containing study.

        Args:
            study_annotation (DataFrame): Dataframe containing `updatedStudyId`, `traitFromSource`, `traitFromSourceMappedIds` and key column `studyId`.

        Returns:
            GWASCatalogStudyIndex: Updated study table.
        """
        self.df = (
            self._df.join(
                study_annotation.select(
                    *[
                        f.col(c).alias(f"updated{c}")
                        if c not in ["studyId", "updatedStudyId"]
                        else f.col(c)
                        for c in study_annotation.columns
                    ]
                ),
                on="studyId",
                how="left",
            )
            .withColumn(
                "studyId",
                f.coalesce(f.col("updatedStudyId"), f.col("studyId")),
            )
            .withColumn(
                "traitFromSource",
                f.coalesce(f.col("updatedtraitFromSource"), f.col("traitFromSource")),
            )
            .withColumn(
                "traitFromSourceMappedIds",
                f.coalesce(
                    f.col("updatedtraitFromSourceMappedIds"),
                    f.col("traitFromSourceMappedIds"),
                ),
            )
            .select(self._df.columns)
        )

        return self

    def _annotate_ancestries(
        self: GWASCatalogStudyIndex, ancestry_lut: DataFrame
    ) -> GWASCatalogStudyIndex:
        """Extracting sample sizes and ancestry information.

        This function parses the ancestry data. Also get counts for the europeans in the same
        discovery stage.

        Args:
            ancestry_lut (DataFrame): Ancestry table as downloaded from the GWAS Catalog

        Returns:
            GWASCatalogStudyIndex: Slimmed and cleaned version of the ancestry annotation.
        """
        ancestry = (
            ancestry_lut
            # Convert column headers to camelcase:
            .transform(
                lambda df: df.select(
                    *[f.expr(column2camel_case(x)) for x in df.columns]
                )
            ).withColumnRenamed(
                "studyAccession", "studyId"
            )  # studyId has not been split yet
        )

        # Get a high resolution dataset on experimental stage:
        ancestry_stages = (
            ancestry.groupBy("studyId")
            .pivot("stage")
            .agg(
                f.collect_set(
                    f.struct(
                        f.col("broadAncestralCategory").alias("ancestry"),
                        f.col("numberOfIndividuals")
                        .cast(t.LongType())
                        .alias("sampleSize"),
                    )
                )
            )
            .withColumn(
                "discoverySamples", self._parse_discovery_samples(f.col("initial"))
            )
            .withColumnRenamed("replication", "replicationSamples")
            # Mapping discovery stage ancestries to LD reference:
            .withColumn(
                "ldPopulationStructure",
                self.aggregate_and_map_ancestries(f.col("discoverySamples")),
            )
            .drop("initial")
            .persist()
        )

        # Generate information on the ancestry composition of the discovery stage, and calculate
        # the proportion of the Europeans:
        europeans_deconvoluted = (
            ancestry
            # Focus on discovery stage:
            .filter(f.col("stage") == "initial")
            # Sorting ancestries if European:
            .withColumn(
                "ancestryFlag",
                # Excluding finnish:
                f.when(
                    f.col("initialSampleDescription").contains("Finnish"),
                    f.lit("other"),
                )
                # Excluding Icelandic population:
                .when(
                    f.col("initialSampleDescription").contains("Icelandic"),
                    f.lit("other"),
                )
                # Including European ancestry:
                .when(f.col("broadAncestralCategory") == "European", f.lit("european"))
                # Exclude all other population:
                .otherwise("other"),
            )
            # Grouping by study accession and initial sample description:
            .groupBy("studyId")
            .pivot("ancestryFlag")
            .agg(
                # Summarizing sample sizes for all ancestries:
                f.sum(f.col("numberOfIndividuals"))
            )
            # Do arithmetics to make sure we have the right proportion of european in the set:
            .withColumn(
                "initialSampleCountEuropean",
                f.when(f.col("european").isNull(), f.lit(0)).otherwise(
                    f.col("european")
                ),
            )
            .withColumn(
                "initialSampleCountOther",
                f.when(f.col("other").isNull(), f.lit(0)).otherwise(f.col("other")),
            )
            .withColumn(
                "initialSampleCount",
                f.col("initialSampleCountEuropean") + f.col("other"),
            )
            .drop(
                "european",
                "other",
                "initialSampleCount",
                "initialSampleCountEuropean",
                "initialSampleCountOther",
            )
        )

        parsed_ancestry_lut = ancestry_stages.join(
            europeans_deconvoluted, on="studyId", how="outer"
        )

        self.df = self.df.join(parsed_ancestry_lut, on="studyId", how="left")
        return self

    def _annotate_sumstats_info(
        self: GWASCatalogStudyIndex, sumstats_lut: DataFrame
    ) -> GWASCatalogStudyIndex:
        """Annotate summary stat locations.

        Args:
            sumstats_lut (DataFrame): listing GWAS Catalog summary stats paths

        Returns:
            GWASCatalogStudyIndex: including `summarystatsLocation` and `hasSumstats` columns
        """
        gwas_sumstats_base_uri = (
            "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        )

        parsed_sumstats_lut = sumstats_lut.withColumn(
            "summarystatsLocation",
            f.concat(
                f.lit(gwas_sumstats_base_uri),
                f.regexp_replace(f.col("_c0"), r"^\.\/", ""),
            ),
        ).select(
            f.regexp_extract(f.col("summarystatsLocation"), r"\/(GCST\d+)\/", 1).alias(
                "studyId"
            ),
            "summarystatsLocation",
            f.lit(True).alias("hasSumstats"),
        )

        self.df = (
            self.df.drop("hasSumstats")
            .join(parsed_sumstats_lut, on="studyId", how="left")
            .withColumn("hasSumstats", f.coalesce(f.col("hasSumstats"), f.lit(False)))
        )
        return self

    def _annotate_discovery_sample_sizes(
        self: GWASCatalogStudyIndex,
    ) -> GWASCatalogStudyIndex:
        """Extract the sample size of the discovery stage of the study as annotated in the GWAS Catalog.

        For some studies that measure quantitative traits, nCases and nControls can't be extracted. Therefore, we assume these are 0.

        Returns:
            GWASCatalogStudyIndex: object with columns `nCases`, `nControls`, and `nSamples` per `studyId` correctly extracted.
        """
        sample_size_lut = (
            self.df.select(
                "studyId",
                f.explode_outer(f.split(f.col("initialSampleSize"), r",\s+")).alias(
                    "samples"
                ),
            )
            # Extracting the sample size from the string:
            .withColumn(
                "sampleSize",
                f.regexp_extract(
                    f.regexp_replace(f.col("samples"), ",", ""), r"[0-9,]+", 0
                ).cast(t.IntegerType()),
            )
            .select(
                "studyId",
                "sampleSize",
                f.when(f.col("samples").contains("cases"), f.col("sampleSize"))
                .otherwise(f.lit(0))
                .alias("nCases"),
                f.when(f.col("samples").contains("controls"), f.col("sampleSize"))
                .otherwise(f.lit(0))
                .alias("nControls"),
            )
            # Aggregating sample sizes for all ancestries:
            .groupBy("studyId")  # studyId has not been split yet
            .agg(
                f.sum("nCases").alias("nCases"),
                f.sum("nControls").alias("nControls"),
                f.sum("sampleSize").alias("nSamples"),
            )
        )
        self.df = self.df.join(sample_size_lut, on="studyId", how="left")
        return self
