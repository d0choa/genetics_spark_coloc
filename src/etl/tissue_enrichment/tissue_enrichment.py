"""Calculation of tissue enrichment!"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from scipy import special as sps

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _melt(
    df: DataFrame,
    id_vars: list,
    value_vars: list,
    var_name: str = "variable",
    value_name: str = "value",
) -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format.  Copied from: https://stackoverflow.com/a/41673644.

    Args:
        df (DataFrame): input wide dataframe.
        id_vars (list): columns names to melt to.
        value_vars (list): values as entries for melted dataframe.
        var_name (str): New column name for the output variables. Defaults to "variable".
        value_name (str): New column name for the output values. Defaults to "value".

    Returns:
        DataFrame: long format dataframe
    """
    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = f.array(
        *(
            f.struct(f.lit(c).alias(var_name), f.col(c).alias(value_name))
            for c in value_vars
        )
    )

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", f.explode(_vars_and_vals))

    cols = id_vars + [
        f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]
    ]
    return _tmp.select(*cols)


def _ndtr(x: float, mean: float, std: float) -> float:
    """This function calculates the area under the cumulative normal distribution.

    Args:
        x (float): Observed rank of the peak.
        mean (float): Expected rank under the null hypothesis.
        std (float): Standard deviation of the null.

    Returns:
        float: Area under the pdf.
    """
    return 0.5 + 0.5 * sps.erf((x - mean) / (std * 2**0.5))


@f.pandas_udf("long")
def vectorized_cdf(x: float, y: float, z: float) -> pd.Series:
    """Calculates the p-value.

    Args:
        x (float): observed peak rank.
        y (float): expected peak rank.
        z (float): standard deviation under the null.

    Returns:
        pd.Series: pd.series of p-values.
    """
    return pd.Series(1 - _ndtr(x, z, y))


def cheers(peaks_wide: DataFrame, snps: DataFrame) -> DataFrame:
    """Calculates tissue enrichment based on the CHEERS method (link here).

    Args:
        peaks_wide (DataFrame): A 4 column .bed file for consensus peaks, 4th column indicates the signal intensity within each peak.
        snps (DataFrame): LD-expanded/Credible SNPs for the trait of interest.

    Returns:
        DataFrame: The enrichment p-value.
    """
    sample_names = peaks_wide.columns[3:]
    peaks_long = _melt(
        df=peaks_wide,
        id_vars=["chr", "start", "end"],
        value_vars=sample_names,
        var_name="sample",
        value_name="score",
    )
    window_spec = Window.partitionBy("sample").orderBy("score")

    # Get ranks. The original code starts ranks from 0, so subtract 1.
    peak_ranks = peaks_long.withColumn("rank", f.rank().over(window_spec) - 1)

    peaks_overlapping = (
        # Only need the peak coords
        peaks_wide.select("chr", "start", "end")
        # Do a inner join
        .join(
            snps,
            (
                (f.col("chrom") == f.col("chr"))
                & (f.col("pos") >= f.col("start"))
                & (f.col("pos") <= f.col("end"))
            ),
            how="inner",
        )
        .drop("chrom")
        .select("study_id", "chr", "start", "end")
        .distinct()
        .persist()
    )

    # Store total number of peaks and total overlapping peaks
    n_total_peaks = float(peaks_wide.count())
    n_unique_peaks = (
        peaks_overlapping.groupBy("study_id")
        .count()
        .withColumnRenamed("count", "count_peaks")
    )

    # Get peaks that overlap
    unique_peaks = peaks_overlapping.join(
        peak_ranks, on=["chr", "start", "end"], how="inner"
    )

    # Get mean rank per sample
    sample_mean_rank = (
        unique_peaks.groupby("study_id", "sample")
        .agg(f.mean(f.col("rank")).alias("mean_rank"))
        .cache()
    )

    n_unique_peaks = n_unique_peaks.withColumn(
        "mean_sd",
        numpy.sqrt(
            ((float(n_total_peaks) ** 2) - 1) / (12 * n_unique_peaks.count_peaks)
        ),
    )

    mean_mean = (1 + n_total_peaks) / 2
    sample_mean_rank_unique_peaks = sample_mean_rank.join(
        n_unique_peaks, on="study_id", how="inner"
    ).withColumn("mean_mean", f.lit(mean_mean))

    sample_mean_rank_unique_peaks = (
        sample_mean_rank_unique_peaks.withColumn(
            "mean_mean", sample_mean_rank_unique_peaks.mean_mean.cast(FloatType())
        )
        .withColumn("mean_sd", sample_mean_rank_unique_peaks.mean_sd.cast(FloatType()))
        .withColumn(
            "mean_rank", sample_mean_rank_unique_peaks.mean_rank.cast(FloatType())
        )
    )

    sample_mean_rank_unique_peaks = sample_mean_rank_unique_peaks.withColumn(
        "pvalue",
        vectorized_cdf(
            sample_mean_rank_unique_peaks.mean_rank,
            sample_mean_rank_unique_peaks.mean_sd,
            sample_mean_rank_unique_peaks.mean_mean,
        ),
    )

    return sample_mean_rank_unique_peaks
