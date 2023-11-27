"""GWAS Catalog Summary Statistics reader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import pyspark.sql.types as t

from otg.common.spark_helpers import neglog_pvalue_to_mantissa_and_exponent
from otg.common.utils import convert_odds_ratio_to_beta, parse_pvalue
from otg.dataset.summary_statistics import SummaryStatistics

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def filename_to_study_identifier(path: str) -> str:
    r"""Extract GWAS Catalog study identifier from path.

    There's an expectation that the filename has to have the GCST accession of the study.

    Args:
        path(str): filename of the harmonized summary statistics.

    Returns:
        str: GWAS Catalog stuy accession.

    Examples:
        >>> filename_to_study_identifier("http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006090/harmonised/29895819-GCST006090-HP_0000975.h.tsv.gz")
        'GCST006090'
        >>> filename_to_study_identifier("wrong/path")
        Traceback (most recent call last):
        ...
        AssertionError: Path ("wrong/path") does not contain GWAS Catalog study identifier.
        assert None is not None
    """
    file_name = path.split("/")[-1]
    study_id_matches = re.search(r"(GCST\d+)", file_name)

    assert (
        study_id_matches is not None
    ), f'Path ("{path}") does not contain GWAS Catalog study identifier.'

    return study_id_matches[0]


@dataclass
class GWASCatalogSummaryStatistics(SummaryStatistics):
    """GWAS Catalog Summary Statistics reader."""

    @classmethod
    def from_gwas_harmonized_summary_stats(
        cls: type[GWASCatalogSummaryStatistics],
        spark: SparkSession,
        sumstats_file: str,
    ) -> GWASCatalogSummaryStatistics:
        """Create summary statistics object from summary statistics flatfile, harmonized by the GWAS Catalog.

        Things got slightly complicated given the GWAS Catalog harmonization pipelines changed recently so we had to accomodate to
        both formats.

        Args:
            spark (SparkSession): spark session
            sumstats_file (str): list of GWAS Catalog summary stat files, with study ids in them.

        Returns:
            GWASCatalogSummaryStatistics: Summary statistics object.
        """
        sumstats_df = spark.read.csv(sumstats_file, sep="\t", header=True).withColumn(
            # Parsing GWAS Catalog study identifier from filename:
            "studyId",
            f.lit(filename_to_study_identifier(sumstats_file)),
        )

        # Parsing variant id fields:
        chromosome = (
            f.col("hm_chrom")
            if "hm_chrom" in sumstats_df.columns
            else f.col("chromosome")
        ).cast(t.StringType())
        position = (
            f.col("hm_pos")
            if "hm_pos" in sumstats_df.columns
            else f.col("base_pair_location")
        ).cast(t.IntegerType())
        ref_allele = (
            f.col("hm_other_allele")
            if "hm_other_allele" in sumstats_df.columns
            else f.col("other_allele")
        )
        alt_allele = (
            f.col("hm_effect_allele")
            if "hm_effect_allele" in sumstats_df.columns
            else f.col("effect_allele")
        )

        # Parsing p-value (get a tuple with mantissa and exponent):
        p_value_expression = (
            parse_pvalue(f.col("p_value"))
            if "p_value" in sumstats_df.columns
            else neglog_pvalue_to_mantissa_and_exponent(f.col("neg_log_10_p_value"))
        )

        # The effect allele frequency is an optional column, we have to test if it is there:
        allele_frequency = (
            f.col("effect_allele_frequency")
            if "effect_allele_frequency" in sumstats_df.columns
            else f.lit(None)
        ).cast(t.FloatType())

        # Do we have sample size? This expression captures 99.7% of sample size columns.
        sample_size = (f.col("n") if "n" in sumstats_df.columns else f.lit(None)).cast(
            t.IntegerType()
        )

        # Depending on the input, we might have beta, but the column might not be there at all also old format calls differently:
        beta_expression = (
            f.col("hm_beta")
            if "hm_beta" in sumstats_df.columns
            else f.col("beta")
            if "beta" in sumstats_df.columns
            # If no column, create one:
            else f.lit(None)
        ).cast(t.DoubleType())

        # We might have odds ratio or hazard ratio, wich are basically the same:
        odds_ratio_expression = (
            f.col("hm_odds_ratio")
            if "hm_odds_ratio" in sumstats_df.columns
            else f.col("odds_ratio")
            if "odds_ratio" in sumstats_df.columns
            else f.col("hazard_ratio")
            if "hazard_ratio" in sumstats_df.columns
            # If no column, create one:
            else f.lit(None)
        ).cast(t.DoubleType())

        # Standard error is mandatory but called differently in the new format:
        standard_error = f.col("standard_error").cast(t.DoubleType())

        # Processing columns of interest:
        processed_sumstats_df = (
            sumstats_df
            # Dropping rows which doesn't have proper position:
            .select(
                "studyId",
                # Adding variant identifier:
                f.concat_ws(
                    "_",
                    chromosome,
                    position,
                    ref_allele,
                    alt_allele,
                ).alias("variantId"),
                chromosome.alias("chromosome"),
                position.alias("position"),
                # Parsing p-value mantissa and exponent:
                *p_value_expression,
                # Converting/calculating effect and confidence interval:
                *convert_odds_ratio_to_beta(
                    beta_expression,
                    odds_ratio_expression,
                    standard_error,
                ),
                allele_frequency.alias("effectAlleleFrequencyFromSource"),
                sample_size.alias("sampleSize"),
            )
            .filter(
                # Dropping associations where no harmonized position is available:
                f.col("position").isNotNull()
                &
                # We are not interested in associations with zero effect:
                (f.col("beta") != 0)
            )
            .orderBy(f.col("chromosome"), f.col("position"))
            # median study size is 200Mb, max is 2.6Gb
            .repartition(20)
        )

        # Initializing summary statistics object:
        return cls(
            _df=processed_sumstats_df,
            _schema=cls.get_schema(),
        )
