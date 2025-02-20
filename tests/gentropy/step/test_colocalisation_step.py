"""Test colocalisation step."""

from pathlib import Path

import pytest

from gentropy.colocalisation import ColocalisationStep
from gentropy.common.session import Session
from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.study_locus import StudyLocus
from gentropy.method.colocalisation import Coloc, ColocalisationMethodInterface, ECaviar


@pytest.mark.step_test
class TestColocalisationStep:
    """Test colocalisation steps."""

    @pytest.fixture(autouse=True)
    def _setup(self, session: Session, tmp_path: Path) -> None:
        """Setup StudyLocus for testing."""
        credible_set_data = [
            (
                "-1299941111165481046",
                "gwas",
                "1_62634374_G_GA",
                "1",
                62634374,
                "1:62116600-63176657",
                "GCST90269661",
                None,
                -18.026562155233105,
                8.294741,
                -72,
                None,
                None,
                None,
                [
                    "Variant not found in LD reference, Study locus finemapped without in-sample LD reference"
                ],
                "SuSiE-inf",
                2,
                128.08235878972883,
                1.0,
                1.0,
                62116600,
                63176657,
                None,
                [("1_62634374_G_GA", 1.0)],
                [
                    (
                        True,
                        True,
                        303.2017476882394,
                        1.0,
                        "1_62634374_G_GA",
                        None,
                        None,
                        -0.07779708137213309,
                        None,
                        None,
                    )
                ],
                "SuSiE fine-mapped credible set with out-of-sample LD",
                None,
            ),
            (
                "-1245591334543437941",
                "gwas",
                "1_62725906_C_A",
                "1",
                62725906,
                "1:62275115-62861709",
                "GCST90024601",
                None,
                6.818181818181818,
                1.0845997,
                -12,
                None,
                None,
                None,
                [
                    "Variant not found in LD reference, Study locus finemapped without in-sample LD reference"
                ],
                "SuSiE-inf",
                3,
                903.4374513916813,
                1.0,
                1.0,
                62275115,
                62861709,
                None,
                [("1_62725906_C_A", 1.0)],
                [
                    (
                        True,
                        True,
                        2087.4457573345685,
                        0.9999999999381545,
                        "1_62725906_C_A",
                        None,
                        None,
                        0.20241232721094407,
                        None,
                        None,
                    )
                ],
                "SuSiE fine-mapped credible set with out-of-sample LD",
                None,
            ),
            (
                "-0.20241232721094407",
                "gwas",
                "1_62725906_C_A",
                "1",
                62725906,
                "1:62335572-62883302",
                "GCST90025461",
                None,
                6.363636363636364,
                5.0753098,
                -10,
                None,
                None,
                None,
                [
                    "Variant not found in LD reference, Study locus finemapped without in-sample LD reference"
                ],
                "SuSiE-inf",
                2,
                912.1598183692258,
                1.0,
                1.0,
                62335572,
                62883302,
                None,
                [("1_62725906_C_A", 1.0)],
                [
                    (
                        True,
                        True,
                        2107.38950418228,
                        0.9999999999454303,
                        "1_62725906_C_A",
                        None,
                        None,
                        0.20330391077149534,
                        None,
                        None,
                    )
                ],
                "SuSiE fine-mapped credible set with out-of-sample LD",
                None,
            ),
            (
                "-2271857845883525223",
                "gwas",
                "1_62634374_G_GA",
                "1",
                62634374,
                "1:62192511-63034021",
                "GCST90269580",
                None,
                -15.43232373355239,
                1.0077391,
                -54,
                None,
                None,
                None,
                [
                    "Variant not found in LD reference, Study locus finemapped without in-sample LD reference"
                ],
                "SuSiE-inf",
                2,
                104.77639852123883,
                1.0,
                1.0,
                62192511,
                63034021,
                None,
                [("1_62634374_G_GA", 1.0)],
                [
                    (
                        True,
                        True,
                        249.20354469210795,
                        1.0,
                        "1_62634374_G_GA",
                        None,
                        None,
                        -0.07071263272378725,
                        None,
                        None,
                    )
                ],
                "SuSiE fine-mapped credible set with out-of-sample LD",
                None,
            ),
        ]
        self.credible_set_path = str(tmp_path / "credible_set_datasets")
        session.spark.createDataFrame(
            credible_set_data, schema=StudyLocus.get_schema()
        ).write.parquet(self.credible_set_path)
        self.coloc_path = str(tmp_path / "colocalisation")

    @pytest.mark.parametrize(
        ["label", "expected_method"],
        [
            pytest.param("coloc", Coloc, id="coloc method"),
            pytest.param("ecaviar", ECaviar, id="ecaviar method"),
            pytest.param("ECaviar", ECaviar, id="uppercase label"),
        ],
    )
    def test_get_colocalisation_class(
        self, label: str, expected_method: type[ColocalisationMethodInterface]
    ) -> None:
        """Test _get_colocalisation_class method on ColocalisationStep."""
        method = ColocalisationStep._get_colocalisation_class(label)
        assert method is expected_method, (
            "Incorrect colocalisation class returned by ColocalisationStep._get_colocalisation_class(label)"
        )

    def test_label_with_invalid_method(self) -> None:
        """Test what happens when invalid method_label is passed to the _get_colocalisation_class."""
        with pytest.raises(ValueError):
            ColocalisationStep._get_colocalisation_class("NewMethod")

    @pytest.mark.parametrize(
        ["coloc_method", "expected_data"],
        [
            pytest.param(
                "ecaviar",
                {
                    "clpp": [1.0, 1.0],
                    "colocalisationMethod": ["eCAVIAR", "eCAVIAR"],
                    "leftStudyLocusId": [
                        "-1245591334543437941",
                        "-2271857845883525223",
                    ],
                    "rightStudyLocusId": [
                        "-0.20241232721094407",
                        "-1299941111165481046",
                    ],
                },
                id="ecaviar",
            ),
            pytest.param(
                "coloc",
                {
                    "h4": [1.0, 1.0],
                    "h3": [0.0, 0.0],
                    "h2": [0.0, 0.0],
                    "h1": [0.0, 0.0],
                    "h0": [0.0, 0.0],
                    "colocalisationMethod": ["COLOC", "COLOC"],
                    "leftStudyLocusId": [
                        "-1245591334543437941",
                        "-2271857845883525223",
                    ],
                    "rightStudyLocusId": [
                        "-0.20241232721094407",
                        "-1299941111165481046",
                    ],
                },
                id="coloc",
            ),
        ],
    )
    def test_colocalise(
        self,
        coloc_method: str,
        expected_data: dict[str, list[float] | list[str]],
        session: Session,
    ) -> None:
        """Test colocalise method."""
        ColocalisationStep(
            session=session,
            credible_set_path=self.credible_set_path,
            coloc_path=self.coloc_path,
            colocalisation_method=coloc_method,
        )

        coloc_dataset = Colocalisation.from_parquet(
            session, self.coloc_path, recursiveFileLookup=True
        )
        for column, expected_values in expected_data.items():
            values = [c[column] for c in coloc_dataset.df.collect()]
            for v, e in zip(values, expected_values):
                if isinstance(e, float):
                    assert e == pytest.approx(v, 1e-1), (
                        f"Incorrect value {v} at {column} found in {coloc_method}, expected {e}"
                    )
                else:
                    assert e == v, (
                        f"Incorrect value {v} at {column} found in {coloc_method}, expected {e}"
                    )
