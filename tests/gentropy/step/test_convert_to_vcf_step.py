"""Test convert to vcf step."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from gentropy.common.session import Session
from gentropy.variant_index import ConvertToVcfStep

if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.step_test
class TestConvertToVcfStep:
    """Test ConvertToVcfStep.

    Test if the step correctly read multiple variant sources and extracts
    non duplicated variants and collects to sorted vcf partitions.
    """

    @pytest.mark.parametrize(
        ["sources", "partition_size", "expected_partition_number"],
        [
            pytest.param(
                [
                    {
                        "path": "tests/gentropy/data_samples/variant_sources/uniprot-test.jsonl",
                        "format": "json",
                        "n_variants": 50,  # 2 variants per chromosome
                    },
                    {
                        "path": "tests/gentropy/data_samples/variant_sources/eva-test.jsonl",
                        "format": "json",
                        "n_variants": 50,  # 2 variants per chromosome
                    },
                    {
                        "path": "tests/gentropy/data_samples/variant_sources/pharmacogenomics-test.jsonl",
                        "format": "json",
                        "n_variants": 44,  # missing Y and MT, input contains two duplicated variants 22_19963748_G_A and 12_21178615_T_C and
                    },
                    {
                        "path": "tests/gentropy/data_samples/variant_sources/credible-sets",
                        "format": "parquet",
                        "n_variants": 42,  # after loci explosion
                    },
                ],
                10,
                19,  # 186 variants / 10 size ~ 19 partitions
                id="Multiple OT datasets",
            ),
            pytest.param(
                [
                    {
                        "path": "tests/gentropy/data_samples/variant_sources/credible-sets-extended",
                        "format": "parquet",
                        "n_variants": 1187,  # after deduplication of locus object
                    }
                ],
                2000,
                1,  # 1199 variants / 2000 size ~ 1 partition
                id="More variants than spark default partition size",
            ),
        ],
    )
    def test_step(
        self,
        session: Session,
        tmp_path: Path,
        sources: list[dict[Literal["path", "format", "n_variants"], Any]],
        partition_size: int,
        expected_partition_number: int,
    ) -> None:
        """Test step.

        Expect that step outputs asserted number of partitions, where each partition
        contains expected number of variants.
        """
        source_paths = [s["path"] for s in sources]
        source_formats = [s["format"] for s in sources]
        output_path = str(tmp_path / "variants")
        ConvertToVcfStep(
            session, source_paths, source_formats, output_path, partition_size
        )

        variants_df = session.spark.read.csv(output_path, sep="\t", header=True)
        # 40 variants (10 variants from each source)
        expected_variant_count = sum(c["n_variants"] for c in sources)
        assert variants_df.count() == expected_variant_count, (
            "Found incorrect number of variants"
        )
        partitions = [
            str(p) for p in Path(output_path).iterdir() if str(p).endswith("csv")
        ]
        assert len(partitions) == expected_partition_number, (
            "Found incorrect number of partitions"
        )

    def test_sorting(
        self,
        session: Session,
        tmp_path: Path,
    ) -> None:
        """Test sorting in partitions.

        Test if variants within single partition are sorted correctly.
        The partition should be naturally sorted by #CHROM and POS fields.
        """
        source_path = (
            "tests/gentropy/data_samples/variant_sources/uniprot-test-sort.jsonl"
        )
        output_path = str(tmp_path / "variants")
        ConvertToVcfStep(session, [source_path], ["json"], output_path, 10)
        partitions = [
            str(p) for p in Path(output_path).iterdir() if str(p).endswith("csv")
        ]
        assert len(partitions) == 1, "Must be one partition to test variant sorting"
        df = pd.read_csv(
            partitions[0],
            usecols=[0, 1],  # just read #CHROM and POS
            sep="\t",
        )
        # values comes from input file tests/gentropy/data_samples/variant_sources/uniprot-test-sorting.jsonl
        # NOTE: Natural ordering in CHROM (str) and POS (int)
        with open(partitions[0]) as fp:
            assert fp.readline().startswith("CHROM\tPOS")

        expected_df = pd.DataFrame(
            [
                ("1", 1525242),
                ("1", 161306863),
                ("11", 108345818),
                ("2", 98396018),
                ("21", 44286656),
                ("3", 38585800),
                ("MT", 6277),
                ("X", 129562612),
                ("Y", 2787426),
            ],
            columns=["CHROM", "POS"],
        )

        assert list(df.columns) == list(expected_df.columns)
        assert df.equals(expected_df), "Variant sorting does not match expectations."

    def test_raises_assertion_imbalanced_arg_ratios(self, session: Session) -> None:
        """Test imbalanced argument ratio exception.

        Test if passing uneven number of sources to paths, not 1:1 ratio should result in assertion
        """
        with pytest.raises(AssertionError) as e:
            ConvertToVcfStep(session, ["dummy_path"], ["json", "json"], "output", 10)
            assert e.value[0] == "Must provide format for each source path."
