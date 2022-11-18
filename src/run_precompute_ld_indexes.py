"""Precompute LD indexes"""
from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from etl.gwas_ingest.ld import precompute_ld_index

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    for population in cfg.etl.gwas_ingest.inputs.gnomad_populations:
        parsed_index = precompute_ld_index(
            population.matrix,
            population.index,
            cfg.etl.gwas_ingest.parameters.ld_window,
        )

        parsed_index.write.mode(cfg.environment.sparkWriteMode).parquet(
            population.parsed_index
        )
    return None


if __name__ == "__main__":
    main()
