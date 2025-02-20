"""Step to process interval datasets."""

from __future__ import annotations

from functools import reduce

from gentropy.common.Liftover import LiftOverSpark
from gentropy.common.session import Session
from gentropy.dataset.intervals import Intervals
from gentropy.dataset.target_index import TargetIndex


class IntervalStep:
    """Interval step.

    This step aims to generate a dataset that contains multiple pieces of evidence supporting the functional association of specific genomic intervals with genes. Some of the evidence types include:

    1. Chromatin interaction experiments, e.g. Promoter Capture Hi-C (PCHi-C).
    2. Enhancer-TSS activity correlations.
    3. Promoter-Chromatin accessibility correlations.

    Attributes:
        session (Session): Session object.
        target_index_path (str): Input target index path.
        liftover_chain_file_path (str): Path to GRCh37 to GRCh38 chain file.
        liftover_max_length_difference: Maximum length difference for liftover.
        intervals (dict): Dictionary of interval sources.
        processed_interval_path (str): Output path to processed intervals.
    """

    def __init__(
        self,
        session: Session,
        target_index_path: str,
        liftover_chain_file_path: str,
        interval_sources: dict[str, str],
        processed_interval_path: str,
        liftover_max_length_difference: int = 100,
    ) -> None:
        """Run Interval processing step.

        Args:
            session (Session): Session object.
            target_index_path (str): Input target index path.
            liftover_chain_file_path (str): Path to GRCh37 to GRCh38 chain file.
            interval_sources (dict[str, str]): Dictionary of interval sources.
            processed_interval_path (str): Output for processed Intervals path.
            liftover_max_length_difference (int): Maximum length difference for liftover.
        """
        # Read
        target_index = TargetIndex.from_parquet(
            session,
            target_index_path,
        ).persist()
        lift = LiftOverSpark(
            # lift over variants to hg38
            liftover_chain_file_path,
            liftover_max_length_difference,
        )
        intervals = Intervals(
            _df=reduce(
                lambda x, y: x.unionByName(y, allowMissingColumns=True),
                # create interval instances by parsing each source
                [
                    Intervals.from_source(
                        session.spark, source_name, source_path, target_index, lift
                    ).df
                    for source_name, source_path in interval_sources.items()
                ],
            ),
            _schema=Intervals.get_schema(),
        )
        intervals.df.write.mode(session.write_mode).parquet(processed_interval_path)
