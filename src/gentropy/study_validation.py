"""Step to validate study index against disease and target index."""

from __future__ import annotations

from pyspark.sql import functions as f

from gentropy.common.session import Session
from gentropy.dataset.biosample_index import BiosampleIndex
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.target_index import TargetIndex


class StudyValidationStep:
    """Study index validation step.

    This step reads and outputs a study index dataset with flagged studies
    when target of disease validation fails.
    """

    def __init__(
        self,
        session: Session,
        study_index_path: list[str],
        target_index_path: str,
        disease_index_path: str,
        biosample_index_path: str,
        valid_study_index_path: str,
        invalid_study_index_path: str,
        invalid_qc_reasons: list[str] = [],
    ) -> None:
        """Initialize step.

        Args:
            session (Session): Session object.
            study_index_path (list[str]): Path to study index file.
            target_index_path (str): Path to target index file.
            disease_index_path (str): Path to disease index file.
            biosample_index_path (str): Path to biosample index file.
            valid_study_index_path (str): Path to write the valid records.
            invalid_study_index_path (str): Path to write the output file.
            invalid_qc_reasons (list[str]): List of invalid quality check reason names from `StudyQualityCheck` (e.g. ['DUPLICATED_STUDY']).
        """
        # Reading datasets:
        target_index = TargetIndex.from_parquet(session, target_index_path)
        biosample_index = BiosampleIndex.from_parquet(session, biosample_index_path)
        # Reading disease index and pre-process.
        # This logic does not belong anywhere, but gentorpy has no disease dataset yet.
        disease_index = (
            session.spark.read.parquet(disease_index_path)
            .select(
                f.col("id").alias("diseaseId"),
                f.explode_outer(
                    f.when(
                        f.col("obsoleteTerms").isNotNull(),
                        f.array_union(f.array("id"), f.col("obsoleteTerms")),
                    )
                ).alias("efo"),
            )
            .withColumn("efo", f.coalesce(f.col("efo"), f.col("diseaseId")))
        )
        study_index = StudyIndex.from_parquet(session, list(study_index_path))

        # Running validation:
        study_index_with_qc = (
            study_index.deconvolute_studies()  # Deconvolute studies where the same study is ingested from multiple sources
            .validate_study_type()  # Flagging non-supported study types
            .validate_target(target_index)  # Flagging QTL studies with invalid targets
            .validate_disease(disease_index)  # Flagging invalid EFOs
            .validate_biosample(
                biosample_index
            )  # Flagging QTL studies with invalid biosamples
        ).persist()  # we will need this for 2 types of outputs

        study_index_with_qc.valid_rows(invalid_qc_reasons, invalid=True).df.coalesce(
            session.output_partitions
        ).write.mode(session.write_mode).parquet(invalid_study_index_path)

        study_index_with_qc.valid_rows(invalid_qc_reasons).df.coalesce(
            session.output_partitions
        ).write.mode(session.write_mode).parquet(valid_study_index_path)
