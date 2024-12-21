"""Step to import filtered version of a LD matrix (block matrix)."""

from __future__ import annotations

import sys
from functools import reduce
from typing import TYPE_CHECKING

import hail as hl
import numpy as np
import pyspark.sql.functions as f
from hail.linalg import BlockMatrix
from pyspark.sql import Window

from gentropy.common.spark_helpers import get_top_ranked_in_window, get_value_from_row
from gentropy.common.types import LD_Population
from gentropy.common.utils import _liftover_loci
from gentropy.config import LDIndexConfig
from gentropy.dataset.ld_index import LDIndex

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, Row


class GnomADLDMatrix:
    """Toolset ot interact with GnomAD LD dataset (version: r2.1.1)."""

    def __init__(
        self,
        ld_matrix_template: str = LDIndexConfig().ld_matrix_template,
        ld_index_raw_template: str = LDIndexConfig().ld_index_raw_template,
        grch37_to_grch38_chain_path: str = LDIndexConfig().grch37_to_grch38_chain_path,
        ld_populations: list[LD_Population | str] = LDIndexConfig().ld_populations,
        liftover_ht_path: str = LDIndexConfig().liftover_ht_path,
    ):
        """Initialize.

        Datasets are accessed in Hail's native format, as provided by the [GnomAD consortium](https://gnomad.broadinstitute.org/downloads/#v2-linkage-disequilibrium).

        Args:
            ld_matrix_template (str): Template for the LD matrix path.
            ld_index_raw_template (str): Template for the LD index path.
            grch37_to_grch38_chain_path (str): Path to the chain file used to lift over the coordinates.
            ld_populations (list[LD_Population | str]): List of populations to use to build the LDIndex.
            liftover_ht_path (str): Path to the liftover ht file.

        Default values are set in LDIndexConfig.
        """
        self.ld_matrix_template = ld_matrix_template
        self.ld_index_raw_template = ld_index_raw_template
        self.grch37_to_grch38_chain_path = grch37_to_grch38_chain_path
        self.ld_populations = ld_populations
        self.liftover_ht_path = liftover_ht_path

    @staticmethod
    def _aggregate_ld_index_across_populations(
        unaggregated_ld_index: DataFrame,
    ) -> DataFrame:
        """Aggregate LDIndex across populations.

        Args:
            unaggregated_ld_index (DataFrame): Unaggregate LDIndex index dataframe  each row is a variant pair in a population

        Returns:
            DataFrame: Aggregated LDIndex index dataframe  each row is a variant with the LD set across populations

        Examples:
            >>> data = [("1.0", "var1", "X", "var1", "pop1"), ("1.0", "X", "var2", "var2", "pop1"),
            ...         ("0.5", "var1", "X", "var2", "pop1"), ("0.5", "var1", "X", "var2", "pop2"),
            ...         ("0.5", "var2", "X", "var1", "pop1"), ("0.5", "X", "var2", "var1", "pop2")]
            >>> df = spark.createDataFrame(data, ["r", "variantId", "chromosome", "tagvariantId", "population"])
            >>> GnomADLDMatrix._aggregate_ld_index_across_populations(df).printSchema()
            root
             |-- variantId: string (nullable = true)
             |-- chromosome: string (nullable = true)
             |-- ldSet: array (nullable = false)
             |    |-- element: struct (containsNull = false)
             |    |    |-- tagVariantId: string (nullable = true)
             |    |    |-- rValues: array (nullable = false)
             |    |    |    |-- element: struct (containsNull = false)
             |    |    |    |    |-- population: string (nullable = true)
             |    |    |    |    |-- r: string (nullable = true)
            <BLANKLINE>
        """
        return (
            unaggregated_ld_index
            # First level of aggregation: get r/population for each variant/tagVariant pair
            .withColumn("r_pop_struct", f.struct("population", "r"))
            .groupBy("chromosome", "variantId", "tagVariantId")
            .agg(
                f.collect_set("r_pop_struct").alias("rValues"),
            )
            # Second level of aggregation: get r/population for each variant
            .withColumn("r_pop_tag_struct", f.struct("tagVariantId", "rValues"))
            .groupBy("variantId", "chromosome")
            .agg(
                f.collect_set("r_pop_tag_struct").alias("ldSet"),
            )
        )

    @staticmethod
    def _convert_ld_matrix_to_table(
        block_matrix: BlockMatrix, min_r2: float
    ) -> DataFrame:
        """Convert LD matrix to table.

        Args:
            block_matrix (BlockMatrix): LD matrix
            min_r2 (float): Minimum r2 value to keep in the table

        Returns:
            DataFrame: LD matrix as a Spark DataFrame
        """
        table = block_matrix.entries(keyed=False)
        return (
            table.filter(hl.abs(table.entry) >= min_r2**0.5)
            .to_spark()
            .withColumnRenamed("entry", "r")
        )

    @staticmethod
    def _create_ldindex_for_population(
        population_id: str,
        ld_matrix_path: str,
        ld_index_raw_path: str,
        grch37_to_grch38_chain_path: str,
        min_r2: float,
    ) -> DataFrame:
        """Create LDIndex for a specific population.

        Args:
            population_id (str): Population ID
            ld_matrix_path (str): Path to the LD matrix
            ld_index_raw_path (str): Path to the LD index
            grch37_to_grch38_chain_path (str): Path to the chain file used to lift over the coordinates
            min_r2 (float): Minimum r2 value to keep in the table

        Returns:
            DataFrame: LDIndex for a specific population
        """
        # Prepare LD Block matrix
        ld_matrix = GnomADLDMatrix._convert_ld_matrix_to_table(
            BlockMatrix.read(ld_matrix_path), min_r2
        )

        # Prepare table with variant indices
        ld_index = GnomADLDMatrix._process_variant_indices(
            hl.read_table(ld_index_raw_path),
            grch37_to_grch38_chain_path,
        )

        return GnomADLDMatrix._resolve_variant_indices(ld_index, ld_matrix).select(
            "*",
            f.lit(population_id).alias("population"),
        )

    @staticmethod
    def _process_variant_indices(
        ld_index_raw: hl.Table, grch37_to_grch38_chain_path: str
    ) -> DataFrame:
        """Creates a look up table between variants and their coordinates in the LD Matrix.

        !!! info "Gnomad's LD Matrix and Index are based on GRCh37 coordinates. This function will lift over the coordinates to GRCh38 to build the lookup table."

        Args:
            ld_index_raw (hl.Table): LD index table from GnomAD
            grch37_to_grch38_chain_path (str): Path to the chain file used to lift over the coordinates

        Returns:
            DataFrame: Look up table between variants in build hg38 and their coordinates in the LD Matrix
        """
        ld_index_38 = _liftover_loci(
            ld_index_raw, grch37_to_grch38_chain_path, "GRCh38"
        )

        return (
            ld_index_38.to_spark()
            # Filter out variants where the liftover failed
            .filter(f.col("`locus_GRCh38.position`").isNotNull())
            .select(
                f.regexp_replace("`locus_GRCh38.contig`", "chr", "").alias(
                    "chromosome"
                ),
                f.col("`locus_GRCh38.position`").alias("position"),
                f.concat_ws(
                    "_",
                    f.regexp_replace("`locus_GRCh38.contig`", "chr", ""),
                    f.col("`locus_GRCh38.position`"),
                    f.col("`alleles`").getItem(0),
                    f.col("`alleles`").getItem(1),
                ).alias("variantId"),
                f.col("idx"),
            )
            # Filter out ambiguous liftover results: multiple indices for the same variant
            .withColumn("count", f.count("*").over(Window.partitionBy(["variantId"])))
            .filter(f.col("count") == 1)
            .drop("count")
        )

    @staticmethod
    def _resolve_variant_indices(
        ld_index: DataFrame, ld_matrix: DataFrame
    ) -> DataFrame:
        """Resolve the `i` and `j` indices of the block matrix to variant IDs (build 38).

        Args:
            ld_index (DataFrame): Dataframe with resolved variant indices
            ld_matrix (DataFrame): Dataframe with the filtered LD matrix

        Returns:
            DataFrame: Dataframe with variant IDs instead of `i` and `j` indices
        """
        ld_index_i = ld_index.selectExpr(
            "idx as i", "variantId as variantIdI", "chromosome"
        )
        ld_index_j = ld_index.selectExpr("idx as j", "variantId as variantIdJ")
        return (
            ld_matrix.join(ld_index_i, on="i", how="inner")
            .join(ld_index_j, on="j", how="inner")
            .drop("i", "j")
        )

    @staticmethod
    def _transpose_ld_matrix(ld_matrix: DataFrame) -> DataFrame:
        """Transpose LD matrix to a square matrix format.

        Args:
            ld_matrix (DataFrame): Triangular LD matrix converted to a Spark DataFrame

        Returns:
            DataFrame: Square LD matrix without diagonal duplicates

        Examples:
        >>> df = spark.createDataFrame(
        ...     [
        ...         (1, 1, 1.0, "1", "AFR"),
        ...         (1, 2, 0.5, "1", "AFR"),
        ...         (2, 2, 1.0, "1", "AFR"),
        ...     ],
        ...     ["variantIdI", "variantIdJ", "r", "chromosome", "population"],
        ... )
        >>> GnomADLDMatrix._transpose_ld_matrix(df).show()
        +----------+----------+---+----------+----------+
        |variantIdI|variantIdJ|  r|chromosome|population|
        +----------+----------+---+----------+----------+
        |         1|         2|0.5|         1|       AFR|
        |         1|         1|1.0|         1|       AFR|
        |         2|         1|0.5|         1|       AFR|
        |         2|         2|1.0|         1|       AFR|
        +----------+----------+---+----------+----------+
        <BLANKLINE>
        """
        ld_matrix_transposed = ld_matrix.selectExpr(
            "variantIdI as variantIdJ",
            "variantIdJ as variantIdI",
            "r",
            "chromosome",
            "population",
        )
        return ld_matrix.filter(f.col("variantIdI") != f.col("variantIdJ")).unionByName(
            ld_matrix_transposed
        )

    def as_ld_index(
        self: GnomADLDMatrix,
        min_r2: float,
    ) -> LDIndex:
        """Create LDIndex dataset aggregating the LD information across a set of populations.

        **The basic steps to generate the LDIndex are:**

        1. Convert LD matrix to a Spark DataFrame.
        2. Resolve the matrix indices to variant IDs by lifting over the coordinates to GRCh38.
        3. Aggregate the LD information across populations.

        Args:
            min_r2 (float): Minimum r2 value to keep in the table

        Returns:
            LDIndex: LDIndex dataset
        """
        ld_indices_unaggregated = []
        for pop in self.ld_populations:
            try:
                ld_matrix_path = self.ld_matrix_template.format(POP=pop)
                ld_index_raw_path = self.ld_index_raw_template.format(POP=pop)
                pop_ld_index = self._create_ldindex_for_population(
                    pop,
                    ld_matrix_path,
                    ld_index_raw_path.format(pop),
                    self.grch37_to_grch38_chain_path,
                    min_r2,
                )
                ld_indices_unaggregated.append(pop_ld_index)
            except Exception as e:
                print(f"Failed to create LDIndex for population {pop}: {e}")  # noqa: T201
                sys.exit(1)

        ld_index_unaggregated = (
            GnomADLDMatrix._transpose_ld_matrix(
                reduce(lambda df1, df2: df1.unionByName(df2), ld_indices_unaggregated)
            )
            .withColumnRenamed("variantIdI", "variantId")
            .withColumnRenamed("variantIdJ", "tagVariantId")
        )
        return LDIndex(
            _df=self._aggregate_ld_index_across_populations(ld_index_unaggregated),
            _schema=LDIndex.get_schema(),
        )

    def get_ld_variants(
        self: GnomADLDMatrix,
        gnomad_ancestry: str,
        chromosome: str,
        start: int,
        end: int,
    ) -> DataFrame | None:
        """Return melted LD table with resolved variant id based on ancestry and genomic location.

        Args:
            gnomad_ancestry (str): GnomAD major ancestry label eg. `nfe`
            chromosome (str): chromosome label
            start (int): window upper bound
            end (int): window lower bound

        Returns:
            DataFrame | None: LD table with resolved variant id based on ancestry and genomic location
        """
        # Extracting locus:
        ld_index_df = (
            self._process_variant_indices(
                hl.read_table(self.ld_index_raw_template.format(POP=gnomad_ancestry)),
                self.grch37_to_grch38_chain_path,
            )
            .filter(
                (f.col("chromosome") == chromosome)
                & (f.col("position") >= start)
                & (f.col("position") <= end)
            )
            .select("chromosome", "position", "variantId", "idx")
        )

        if ld_index_df.limit(1).count() == 0:
            # If the returned slice from the ld index is empty, return None
            return None

        # Compute start and end indices
        start_index = get_value_from_row(
            get_top_ranked_in_window(
                ld_index_df, Window.partitionBy().orderBy(f.col("position").asc())
            ).collect()[0],
            "idx",
        )
        end_index = get_value_from_row(
            get_top_ranked_in_window(
                ld_index_df, Window.partitionBy().orderBy(f.col("position").desc())
            ).collect()[0],
            "idx",
        )

        return self._extract_square_matrix(
            ld_index_df, gnomad_ancestry, start_index, end_index
        )

    def _extract_square_matrix(
        self: GnomADLDMatrix,
        ld_index_df: DataFrame,
        gnomad_ancestry: str,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """Return LD square matrix for a region where coordinates are normalised.

        Args:
            ld_index_df (DataFrame): Look up table between a variantId and its index in the LD matrix
            gnomad_ancestry (str): GnomAD major ancestry label eg. `nfe`
            start_index (int): start index of the slice
            end_index (int): end index of the slice

        Returns:
            DataFrame: square LD matrix resolved to variants.
        """
        return (
            self.get_ld_matrix_slice(
                gnomad_ancestry, start_index=start_index, end_index=end_index
            )
            .join(
                ld_index_df.select(
                    f.col("idx").alias("idx_i"),
                    f.col("variantId").alias("variantIdI"),
                ),
                on="idx_i",
                how="inner",
            )
            .join(
                ld_index_df.select(
                    f.col("idx").alias("idx_j"),
                    f.col("variantId").alias("variantIdJ"),
                ),
                on="idx_j",
                how="inner",
            )
            .select("variantIdI", "variantIdJ", "r")
        )

    def get_ld_matrix_slice(
        self: GnomADLDMatrix,
        gnomad_ancestry: str,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """Extract a slice of the LD matrix based on the provided ancestry and stop and end indices.

        - The half matrix is completed into a full square.
        - The returned indices are adjusted based on the start index.

        Args:
            gnomad_ancestry (str): LD population label eg. `nfe`
            start_index (int): start index of the slice
            end_index (int): end index of the slice

        Returns:
            DataFrame: square slice of the LD matrix melted as dataframe with idx_i, idx_j and r columns
        """
        # Extracting block matrix slice:
        half_matrix = BlockMatrix.read(
            self.ld_matrix_template.format(POP=gnomad_ancestry)
        ).filter(range(start_index, end_index + 1), range(start_index, end_index + 1))

        # Return converted Dataframe:
        return (
            (half_matrix + half_matrix.T)
            .entries()
            .to_spark()
            .select(
                (f.col("i") + start_index).alias("idx_i"),
                (f.col("j") + start_index).alias("idx_j"),
                f.when(f.col("i") == f.col("j"), f.col("entry") / 2)
                .otherwise(f.col("entry"))
                .alias("r"),
            )
        )

    def get_locus_index(
        self: GnomADLDMatrix,
        study_locus_row: Row,
        radius: int = 500_000,
        major_population: str = "nfe",
    ) -> DataFrame:
        """Extract hail matrix index from StudyLocus rows.

        Args:
            study_locus_row (Row): Study-locus row
            radius (int): Locus radius to extract from gnomad matrix
            major_population (str): Major population to extract from gnomad matrix, default is "nfe"

        Returns:
            DataFrame: Returns the index of the gnomad matrix for the locus

        """
        chromosome = str("chr" + study_locus_row["chromosome"])
        start = study_locus_row["position"] - radius
        end = study_locus_row["position"] + radius

        liftover_ht = hl.read_table(self.liftover_ht_path)
        liftover_ht = (
            liftover_ht.filter(
                (liftover_ht.locus.contig == chromosome)
                & (liftover_ht.locus.position >= start)
                & (liftover_ht.locus.position <= end)
            )
            .key_by()
            .select("locus", "alleles", "original_locus")
            .key_by("original_locus", "alleles")
            .naive_coalesce(20)
        )

        hail_index = hl.read_table(
            self.ld_index_raw_template.format(POP=major_population)
        )

        joined_index = (
            liftover_ht.join(hail_index, how="inner").order_by("idx").to_spark()
        )

        return joined_index

    @staticmethod
    def get_numpy_matrix(
        locus_index: DataFrame,
        gnomad_ancestry: str = "nfe",
    ) -> np.ndarray:
        """Extract the LD block matrix for a locus.

        Args:
            locus_index (DataFrame): hail matrix variant index table
            gnomad_ancestry (str): GnomAD major ancestry label eg. `nfe`

        Returns:
            np.ndarray: LD block matrix for the locus
        """
        idx = [row["idx"] for row in locus_index.select("idx").collect()]

        half_matrix = (
            BlockMatrix.read(
                GnomADLDMatrix().ld_matrix_template.format(POP=gnomad_ancestry)
            )
            .filter(idx, idx)
            .to_numpy()
        )

        return (half_matrix + half_matrix.T) - np.diag(np.diag(half_matrix))

    def get_locus_index_boundaries(
        self: GnomADLDMatrix,
        study_locus_row: Row,
        major_population: str = "nfe",
    ) -> DataFrame:
        """Extract hail matrix index from StudyLocus rows.

        Args:
            study_locus_row (Row): Study-locus row
            major_population (str): Major population to extract from gnomad matrix, default is "nfe"

        Returns:
            DataFrame: Returns the index of the gnomad matrix for the locus

        """
        chromosome = str("chr" + study_locus_row["chromosome"])
        start = int(study_locus_row["locusStart"])
        end = int(study_locus_row["locusEnd"])

        liftover_ht = hl.read_table(self.liftover_ht_path)
        liftover_ht = (
            liftover_ht.filter(
                (liftover_ht.locus.contig == chromosome)
                & (liftover_ht.locus.position >= start)
                & (liftover_ht.locus.position <= end)
            )
            .key_by()
            .select("locus", "alleles", "original_locus")
            .key_by("original_locus", "alleles")
            .naive_coalesce(20)
        )

        hail_index = hl.read_table(
            self.ld_index_raw_template.format(POP=major_population)
        )

        joined_index = (
            liftover_ht.join(hail_index, how="inner").order_by("idx").to_spark()
        )

        return joined_index

    def get_locus_index_boundaries_no_studyLocus(
        self: GnomADLDMatrix,
        chromosomeInput: str,
        start: int,
        end: int,
        major_population: str = "nfe",
    ) -> DataFrame:
        """Extract hail matrix index from StudyLocus rows.

        Args:
            chromosomeInput (str): Chromosome label
            start (int): Locus start
            end (int): Locus end
            major_population (str): Major population to extract from gnomad matrix, default is "nfe"

        Returns:
            DataFrame: Returns the index of the gnomad matrix for the locus

        """
        chromosome = str("chr" + chromosomeInput)

        liftover_ht = hl.read_table(self.liftover_ht_path)
        liftover_ht = (
            liftover_ht.filter(
                (liftover_ht.locus.contig == chromosome)
                & (liftover_ht.locus.position >= start)
                & (liftover_ht.locus.position <= end)
            )
            .key_by()
            .select("locus", "alleles", "original_locus")
            .key_by("original_locus", "alleles")
            .naive_coalesce(20)
        )

        hail_index = hl.read_table(
            self.ld_index_raw_template.format(POP=major_population)
        )

        joined_index = (
            liftover_ht.join(hail_index, how="inner").order_by("idx").to_spark()
        )

        return joined_index
