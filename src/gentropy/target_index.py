"""Step to generate target index dataset."""

from __future__ import annotations

from gentropy.common.session import Session
from gentropy.datasource.open_targets.target import OpenTargetsTarget


class TargetIndexStep:
    """Target index step.

    This step generates a target index dataset from an Open Targets Platform target dataset.
    """

    def __init__(
        self,
        session: Session,
        target_path: str,
        target_index_path: str,
    ) -> None:
        """Initialize step.

        Args:
            session (Session): Session object.
            target_path (str): Input Open Targets Platform target dataset path.
            target_index_path (str): Output target index dataset path.
        """
        platform_target = session.spark.read.parquet(target_path)
        # Transform
        target_index = OpenTargetsTarget.as_target_index(platform_target)
        # Load
        target_index.df.coalesce(session.output_partitions).write.mode(
            session.write_mode
        ).parquet(target_index_path)
