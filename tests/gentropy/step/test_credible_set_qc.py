"""Test credible set qc step."""

from pathlib import Path

import pytest
from pyspark.sql import functions as f
from pyspark.sql import types as t

from gentropy.common.session import Session
from gentropy.credible_set_qc import CredibleSetQCStep
from gentropy.dataset.study_locus import StudyLocus


@pytest.mark.step_test
class TestCredibleSetQCStep:
    """Test credible set qc."""

    @pytest.fixture(autouse=True)
    def _setup(self, session: Session, tmp_path: Path) -> None:
        """Setup StudyLocus for testing."""
        # NOTE: About the input dataset for tests
        # Entry dataset contains 6 loci (3 of them contains duplicated studyLocusId, 2 contains the same studyId)
        # The step is expected to remove the duplicates of the studyLocus (1 row)
        # THe step is expected to remove rows which pValue <= p_val_threshold (1 row)
        # The step is expected to remove rows which purityMinR2 >= to purity_min_r2 (1 row)
        # at the end we should end up with 2 non duplicated loci
        self.purity_min_r2 = 0.01
        self.p_value_threshold = 1e-5
        self.n_partitions = 1
        credible_set_data = [
            (
                "A",  # duplicated credibleSetId
                "1_100_G_GA",  # variantId
                "GCST1",  # duplicated studyId
                1.0,  # pValMantissa
                -6,  # pValExponent
                1.0,  # credibleSetlog10BF -> should be skipped due to the lowest Log10BF
                1.0,  # purityMinR2
            ),
            (
                "A",  # duplicated credibleSetId
                "1_100_G_GA",  # variantId
                "GCST1",  # duplicated studyId
                1.0,  # pValMantissa
                -6,  # pValExponent
                2.0,  # credibleSetlog10BF -> highest log10BF within duplicates considering single study
                1.0,  # purityMinR2
            ),
            (
                "A",  # duplicated credibleSetId
                "1_100_G_GA",  # variantId
                "GCST2",  # studyId
                1.0,  # pValMantissa
                -6,  # pValExponent
                3.0,  # credibleSetlog10BF -> highest log10BF within duplicates
                1.0,  # purityMinR2
            ),
            (
                "B",  # credibleSetId
                "1_200_G_GA",  # variantId
                "GCST3",  # studyId
                1.0,  # pValMantissa
                -4,  # too high pValExponent => pVal = 1.0e-4 < p_val_threshold
                1.0,  # credibleSetlog10BF
                1.0,  # purityMinR2
            ),
            (
                "C",  # credibleSetId
                "1_300_G_GA",  # variantId
                "GCST3",  # studyId
                1.0,  # pValMantissa
                -6,  # pValExponent
                1.0,  # credibleSetlog10BF
                0.001,  # purityMinR2 < purity_min_r2
            ),
            (
                # full row OK!
                "D",  # credibleSetId
                "1_400_G_GA",  # variantId
                "GCST3",  # studyId
                1.0,  # pValMantissa
                -6,  # pValExponent
                1.0,  # credibleSetlog10BF
                1.0,  # purityMinR2
            ),
        ]
        cs_schema = t.StructType(
            [
                t.StructField("studyLocusId", t.StringType(), True),
                t.StructField("variantId", t.StringType(), True),
                t.StructField("studyId", t.StringType(), True),
                t.StructField("pValueMantissa", t.FloatType(), True),
                t.StructField("pValueExponent", t.IntegerType(), True),
                t.StructField("credibleSetlog10BF", t.DoubleType(), True),
                t.StructField("purityMinR2", t.DoubleType(), True),
            ]
        )

        # NOTE: Use proper input!
        # Ensure the input dataset is saved per studyLocusId in recursive manner.
        # This mimics the dataset with multiple loci evaluated separately.
        self.credible_set_path = str(tmp_path / "credible_set_datasets")
        cs_df = session.spark.createDataFrame(credible_set_data, schema=cs_schema)
        cs_path = tmp_path / "credible_set_dataset"
        loci_ids = {row["studyLocusId"] for row in cs_df.collect()}
        for loci_id in loci_ids:
            loci_path = str(cs_path / loci_id)
            cs_df.filter(f.col("studyLocusId") == loci_id).write.parquet(loci_path)
        self.input_cs_df = cs_df
        self.cs_path = str(cs_path)
        self.output_path = str(tmp_path / "clean_credible_sets")

    def test_step(self, session: Session) -> None:
        """Invoke the step to check if it works correctly."""
        assert not Path(self.output_path).exists(), "Input for qc does not exists."
        assert Path(self.cs_path).exists(), "Output of qc is not emptied before test."
        assert self.input_cs_df.count() == 6, "Incorrect number of rows."
        CredibleSetQCStep(
            session=session,
            credible_sets_path=self.cs_path,
            output_path=self.output_path,
            p_value_threshold=self.p_value_threshold,
            purity_min_r2=self.purity_min_r2,
            clump=False,
            ld_index_path=None,
            study_index_path=None,
            ld_min_r2=None,
            n_partitions=self.n_partitions,
        )

        assert Path(self.output_path).exists(), "Output of qc does not exists."
        # check the number of partitions
        partitions = [
            str(p)
            for p in Path(self.output_path).iterdir()
            if str(p).endswith(".parquet")
        ]
        assert len(partitions) == self.n_partitions, (
            "Incorrect number of partitions in the output."
        )
        cs = StudyLocus.from_parquet(
            session, self.output_path, recursiveFileLookup=True
        )
        assert cs.df.count() == 2  # Row A where LogBF == 3.0 and row D
        assert cs.df.rdd.getNumPartitions() == self.n_partitions
        data = {
            row["studyLocusId"]: row["credibleSetlog10BF"] for row in cs.df.collect()
        }
        assert sorted(data.keys()) == ["A", "D"]
        # ensure the Locus A with highest credibleSetlog10BF was chosen
        assert data["A"] == 3.0
