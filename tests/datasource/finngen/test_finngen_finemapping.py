"""Tests for finemapping SuSie results from FinnGen."""

from __future__ import annotations

from otg.dataset.study_locus import StudyLocus
from otg.datasource.finngen.finngen_finemapping import FinnGenFinemapping
from pyspark.sql import SparkSession


def test_finngen_finemapping_from_finngen_susie_finemapping(
    spark: SparkSession,
) -> None:
    """Test finemapping results (SuSie) from source."""
    assert isinstance(
        FinnGenFinemapping.from_finngen_susie_finemapping(
            spark=spark,
            finngen_finemapping_df="tests/data_samples/finngen_R9_AB1_EBV.SUSIE.snp.gz",
            finngen_finemapping_summaries="tests/data_samples/finngen_credset_summary_sample.csv",
            finngen_release_prefix="finngen_R9_",
        ),
        StudyLocus,
    )
