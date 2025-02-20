"""Test of main SuSiE-inf functions."""

from __future__ import annotations

import numpy as np
import pyspark.sql.functions as f

from gentropy.common.session import Session
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics
from gentropy.method.susie_inf import SUSIE_inf
from gentropy.susie_finemapper import SusieFineMapperStep


class TestSUSIE_inf:
    """Test of SuSiE-inf main functions."""

    def test_SUSIE_inf_lbf_moments(
        self: TestSUSIE_inf, sample_data_for_susie_inf: list[np.ndarray]
    ) -> None:
        """Test of SuSiE-inf LBF method of moments."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        lbf_moments = sample_data_for_susie_inf[2]
        susie_output = SUSIE_inf.susie_inf(z=z, LD=ld, est_tausq=True, method="moments")
        lbf_calc = susie_output["lbf_variable"][:, 0]
        assert np.allclose(lbf_calc, lbf_moments), (
            "LBFs for method of moments are not equal"
        )

    def test_SUSIE_inf_lbf_mle(
        self: TestSUSIE_inf, sample_data_for_susie_inf: list[np.ndarray]
    ) -> None:
        """Test of SuSiE-inf LBF maximum likelihood estimation."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        lbf_mle = sample_data_for_susie_inf[3]
        susie_output = SUSIE_inf.susie_inf(z=z, LD=ld, est_tausq=True, method="MLE")
        lbf_calc = susie_output["lbf_variable"][:, 0]
        assert np.allclose(lbf_calc, lbf_mle, atol=1e-1), (
            "LBFs for maximum likelihood estimation are not equal"
        )

    def test_SUSIE_inf_cred(
        self: TestSUSIE_inf, sample_data_for_susie_inf: list[np.ndarray]
    ) -> None:
        """Test of SuSiE-inf credible set generator."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        susie_output = SUSIE_inf.susie_inf(
            z=z,
            LD=ld,
            est_tausq=True,
        )
        cred = SUSIE_inf.cred_inf(susie_output["PIP"], LD=ld)
        assert cred[0] == [5]

    def test_SUSIE_inf_convert_to_study_locus(
        self: TestSUSIE_inf,
        sample_data_for_susie_inf: list[np.ndarray],
        sample_summary_statistics: SummaryStatistics,
        session: Session,
    ) -> None:
        """Test of SuSiE-inf credible set generator."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        susie_output = SUSIE_inf.susie_inf(
            z=z,
            LD=ld,
            est_tausq=False,
        )
        gwas_df = sample_summary_statistics._df.withColumn(
            "z", f.col("beta") / f.col("standardError")
        ).filter(f.col("z").isNotNull())
        gwas_df = gwas_df.limit(21)

        L1 = SusieFineMapperStep.susie_inf_to_studylocus(
            susie_output=susie_output,
            session=session,
            studyId="sample_id",
            region="sample_region",
            variant_index=gwas_df,
            cs_lbf_thr=2,
            ld_matrix=ld,
            lead_pval_threshold=1,
            purity_mean_r2_threshold=0,
            purity_min_r2_threshold=0,
            sum_pips=0.99,
            ld_min_r2=1,
            locusStart=1,
            locusEnd=2,
        )
        assert isinstance(L1, StudyLocus), "L1 is not an instance of StudyLocus"
