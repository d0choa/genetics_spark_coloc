"""Test the command-line interface (CLI)."""

from unittest.mock import patch

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf.errors import MissingMandatoryValue

from gentropy.cli import main


def test_main_no_step() -> None:
    """Test the main function of the CLI without a valid step."""
    override_key = "step"
    expected = f"You must specify '{override_key}', e.g, {override_key}=<OPTION>\nAvailable options:"

    with (
        patch("sys.argv", ["cli.py"]),
        pytest.raises(ConfigCompositionException, match=expected),
    ):
        main()


def test_main_step() -> None:
    """Test the main function of the CLI complains about mandatory values."""
    with (
        patch("sys.argv", ["cli.py", "step=gene_index"]),
        pytest.raises(
            MissingMandatoryValue, match="Missing mandatory value: step.target_path"
        ),
    ):
        main()
