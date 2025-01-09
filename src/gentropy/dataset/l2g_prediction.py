"""Dataset that contains the Locus to Gene predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from gentropy.common.schemas import parse_spark_schema
from gentropy.common.session import Session
from gentropy.dataset.dataset import Dataset
from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.method.l2g.model import LocusToGeneModel

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class L2GPrediction(Dataset):
    """Dataset that contains the Locus to Gene predictions.

    It is the result of applying the L2G model on a feature matrix, which contains all
    the study/locus pairs and their functional annotations. The score column informs the
    confidence of the prediction that a gene is causal to an association.
    """

    @classmethod
    def get_schema(cls: type[L2GPrediction]) -> StructType:
        """Provides the schema for the L2GPrediction dataset.

        Returns:
            StructType: Schema for the L2GPrediction dataset
        """
        return parse_spark_schema("l2g_predictions.json")

    @classmethod
    def from_credible_set(
        cls: type[L2GPrediction],
        session: Session,
        credible_set: StudyLocus,
        feature_matrix: L2GFeatureMatrix,
        features_list: list[str],
        model_path: str | None,
        hf_token: str | None = None,
        download_from_hub: bool = True,
    ) -> L2GPrediction:
        """Extract L2G predictions for a set of credible sets derived from GWAS.

        Args:
            session (Session): Session object that contains the Spark session
            credible_set (StudyLocus): Dataset containing credible sets from GWAS only
            feature_matrix (L2GFeatureMatrix): Dataset containing all credible sets and their annotations
            features_list (list[str]): List of features to use for the model
            model_path (str | None): Path to the model file. It can be either in the filesystem or the name on the Hugging Face Hub (in the form of username/repo_name).
            hf_token (str | None): Hugging Face token to download the model from the Hub. Only required if the model is private.
            download_from_hub (bool): Whether to download the model from the Hugging Face Hub. Defaults to True.

        Returns:
            L2GPrediction: L2G scores for a set of credible sets.
        """
        # Load the model
        if download_from_hub:
            # Model ID defaults to "opentargets/locus_to_gene" and it assumes the name of the classifier is "classifier.skops".
            model_id = model_path or "opentargets/locus_to_gene"
            l2g_model = LocusToGeneModel.load_from_hub(model_id, hf_token)
        elif model_path:
            l2g_model = LocusToGeneModel.load_from_disk(model_path)

        # Prepare data
        fm = (
            L2GFeatureMatrix(
                _df=(
                    credible_set.df.filter(f.col("studyType") == "gwas")
                    .select("studyLocusId")
                    .join(feature_matrix._df, "studyLocusId")
                    .filter(f.col("isProteinCoding") == 1)
                )
            )
            .fill_na()
            .select_features(features_list)
        )

        return l2g_model.predict(fm, session)

    def to_disease_target_evidence(
        self: L2GPrediction,
        study_locus: StudyLocus,
        study_index: StudyIndex,
        l2g_threshold: float = 0.05,
    ) -> DataFrame:
        """Convert locus to gene predictions to disease target evidence.

        Args:
            study_locus (StudyLocus): Study locus dataset
            study_index (StudyIndex): Study index dataset
            l2g_threshold (float): Threshold to consider a gene as a target. Defaults to 0.05.

        Returns:
            DataFrame: Disease target evidence
        """
        datasource_id = "gwas_credible_sets"
        datatype_id = "genetic_association"

        return (
            self.df.filter(f.col("score") >= l2g_threshold)
            .join(
                study_locus.df.select("studyLocusId", "studyId"),
                on="studyLocusId",
                how="inner",
            )
            .join(
                study_index.df.select("studyId", "diseaseIds"),
                on="studyId",
                how="inner",
            )
            .select(
                f.lit(datatype_id).alias("datatypeId"),
                f.lit(datasource_id).alias("datasourceId"),
                f.col("geneId").alias("targetFromSourceId"),
                f.explode(f.col("diseaseIds")).alias("diseaseFromSourceMappedId"),
                f.col("score").alias("resourceScore"),
                "studyLocusId",
            )
        )

    def add_locus_to_gene_features(
        self: L2GPrediction, feature_matrix: L2GFeatureMatrix, features_list: list[str]
    ) -> L2GPrediction:
        """Add features used to extract the L2G predictions.

        Args:
            feature_matrix (L2GFeatureMatrix): Feature matrix dataset
            features_list (list[str]): List of features used in the model

        Returns:
            L2GPrediction: L2G predictions with additional features
        """
        # Testing if `locusToGeneFeatures` column already exists:
        if "locusToGeneFeatures" in self.df.columns:
            self.df = self.df.drop("locusToGeneFeatures")

        # Aggregating all features into a single map column:
        aggregated_features = (
            feature_matrix._df.withColumn(
                "locusToGeneFeatures",
                f.create_map(
                    *sum(
                        ((f.lit(feature), f.col(feature)) for feature in features_list),
                        (),
                    )
                ),
            )
            .withColumn(
                "locusToGeneFeatures",
                f.expr("map_filter(locusToGeneFeatures, (k, v) -> v != 0)"),
            )
            .drop(*features_list)
        )
        return L2GPrediction(
            _df=self.df.join(
                aggregated_features, on=["studyLocusId", "geneId"], how="left"
            ),
            _schema=self.get_schema(),
        )
