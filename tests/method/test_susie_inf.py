"""Test of main SuSiE-inf functions."""

from __future__ import annotations

import numpy as np
from otg.method.susie_inf import SUSIE_inf


class TestSUSIE_inf:
    """Test of SuSiE-inf main functions."""

    def test_SUSIE_inf_lbf_moments(
        self: TestSUSIE_inf, sample_data_for_susie_inf: list[np.ndarray]
    ) -> None:
        """Test of SuSiE-inf LBF method of moments."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        lbf_moments = sample_data_for_susie_inf[2]
        susie_output = SUSIE_inf.susie_inf(z=z, LD=ld, method="moments")
        lbf_calc = susie_output["lbf_variable"][:, 0]
        assert np.array_equal(
            lbf_calc, lbf_moments
        ), "LBFs for method of moments are not equal"

    def test_SUSIE_inf_lbf_mle(
        self: TestSUSIE_inf, sample_data_for_susie_inf: list[np.ndarray]
    ) -> None:
        """Test of SuSiE-inf LBF maximum likelihood estimation."""
        ld = sample_data_for_susie_inf[0]
        z = sample_data_for_susie_inf[1]
        lbf_mle = sample_data_for_susie_inf[3]
        susie_output = SUSIE_inf.susie_inf(z=z, LD=ld, method="MLE")
        lbf_calc = susie_output["lbf_variable"][:, 0]
        assert np.array_equal(
            lbf_calc, lbf_mle
        ), "LBFs for maximum likelihood estimation are not equal"
