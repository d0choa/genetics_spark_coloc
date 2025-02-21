"""Dataset that contains the Locus to Gene predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
import shap
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from gentropy.common.schemas import parse_spark_schema
from gentropy.common.session import Session
from gentropy.common.spark_helpers import pivot_df
from gentropy.dataset.dataset import Dataset
from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.method.l2g.model import LocusToGeneModel

if TYPE_CHECKING:
    from pandas import DataFrame as pd_dataframe
    from pyspark.sql.types import StructType


@dataclass
class L2GPrediction(Dataset):
    """Dataset that contains the Locus to Gene predictions.

    It is the result of applying the L2G model on a feature matrix, which contains all
    the study/locus pairs and their functional annotations. The score column informs the
    confidence of the prediction that a gene is causal to an association.
    """

    model: LocusToGeneModel | None = field(default=None, repr=False)

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
        model_path: str | None,
        features_list: list[str] | None = None,
        hf_token: str | None = None,
        download_from_hub: bool = True,
    ) -> L2GPrediction:
        """Extract L2G predictions for a set of credible sets derived from GWAS.

        Args:
            session (Session): Session object that contains the Spark session
            credible_set (StudyLocus): Dataset containing credible sets from GWAS only
            feature_matrix (L2GFeatureMatrix): Dataset containing all credible sets and their annotations
            model_path (str | None): Path to the model file. It can be either in the filesystem or the name on the Hugging Face Hub (in the form of username/repo_name).
            features_list (list[str] | None): Default list of features the model uses. Only used if the model is not downloaded from the Hub. CAUTION: This default list can differ from the actual list the model was trained on.
            hf_token (str | None): Hugging Face token to download the model from the Hub. Only required if the model is private.
            download_from_hub (bool): Whether to download the model from the Hugging Face Hub. Defaults to True.

        Returns:
            L2GPrediction: L2G scores for a set of credible sets.

        Raises:
            AttributeError: If `features_list` is not provided and the model is not downloaded from the Hub.
        """
        # Load the model
        if download_from_hub:
            # Model ID defaults to "opentargets/locus_to_gene" and it assumes the name of the classifier is "classifier.skops".
            model_id = model_path or "opentargets/locus_to_gene"
            l2g_model = LocusToGeneModel.load_from_hub(session, model_id, hf_token)
        elif model_path:
            if not features_list:
                raise AttributeError(
                    "features_list is required if the model is not downloaded from the Hub"
                )
            l2g_model = LocusToGeneModel.load_from_disk(
                session, path=model_path, features_list=features_list
            )

        # Prepare data
        fm = (
            L2GFeatureMatrix(
                _df=(
                    credible_set.df.filter(f.col("studyType") == "gwas")
                    .select("studyLocusId")
                    .join(feature_matrix._df, "studyLocusId")
                    .filter(f.col("isProteinCoding") == 1)
                ),
            )
            .fill_na()
            .select_features(l2g_model.features_list)
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

    def explain(
        self: L2GPrediction, feature_matrix: L2GFeatureMatrix | None = None
    ) -> L2GPrediction:
        """Extract Shapley values for the L2G predictions and add them as a map in an additional column.

        Args:
            feature_matrix (L2GFeatureMatrix | None): Feature matrix in case the predictions are missing the feature annotation. If None, the features are fetched from the dataset.

        Returns:
            L2GPrediction: L2GPrediction object with additional column containing feature name to Shapley value mappings

        Raises:
            ValueError: If the model is not set or If feature matrix is not provided and the predictions do not have features
        """
        # Fetch features if they are not present:
        if "features" not in self.df.columns:
            if feature_matrix is None:
                raise ValueError(
                    "Feature matrix is required to explain the L2G predictions"
                )
            self.add_features(feature_matrix)

        if self.model is None:
            raise ValueError("Model not set, explainer cannot be created")

        # Format and pivot the dataframe to pass them before calculating shapley values
        pdf = pivot_df(
            df=self.df.withColumn("feature", f.explode("features")).select(
                "studyLocusId",
                "geneId",
                "score",
                f.col("feature.name").alias("feature_name"),
                f.col("feature.value").alias("feature_value"),
            ),
            pivot_col="feature_name",
            value_col="feature_value",
            grouping_cols=[f.col("studyLocusId"), f.col("geneId"), f.col("score")],
        ).toPandas()
        pdf = pdf.rename(
            # trim the suffix that is added after pivoting the df
            columns={
                col: col.replace("_feature_value", "")
                for col in pdf.columns
                if col.endswith("_feature_value")
            }
        )

        features_list = self.model.features_list  # The matrix needs to present the features in the same order that the model was trained on)
        base_value, shap_values = L2GPrediction._explain(
            model=self.model,
            pdf=pdf.filter(items=features_list),
        )
        for i, feature in enumerate(features_list):
            pdf[f"shap_{feature}"] = [row[i] for row in shap_values]

        spark_session = self.df.sparkSession
        return L2GPrediction(
            _df=(
                spark_session.createDataFrame(pdf.to_dict(orient="records"))
                .withColumn(
                    "features",
                    f.array(
                        *(
                            f.struct(
                                f.lit(feature).alias("name"),
                                f.col(feature).cast("float").alias("value"),
                                f.col(f"shap_{feature}")
                                .cast("float")
                                .alias("shapValue"),
                            )
                            for feature in features_list
                        )
                    ),
                )
                .withColumn("shapBaseValue", f.lit(base_value).cast("float"))
                .select(*L2GPrediction.get_schema().names)
            ),
            _schema=self.get_schema(),
            model=self.model,
        )

    @staticmethod
    def _explain(
        model: LocusToGeneModel, pdf: pd_dataframe
    ) -> tuple[float, list[list[float]]]:
        """Calculate SHAP values. Output is in probability form (approximated from the log odds ratios).

        Args:
            model (LocusToGeneModel): L2G model
            pdf (pd_dataframe): Pandas dataframe containing the feature matrix in the same order that the model was trained on

        Returns:
            tuple[float, list[list[float]]]: A tuple containing:
                - base_value (float): Base value of the model
                - shap_values (list[list[float]]): SHAP values for prediction

        Raises:
            AttributeError: If model.training_data is not set, seed dataset to get shapley values cannot be created.
        """
        if not model.training_data:
            raise AttributeError(
                "`model.training_data` is missing, seed dataset to get shapley values cannot be created."
            )
        background_data = (
            model.training_data._df.select(*model.features_list)
            .toPandas()
            .sample(n=1_000)
        )
        explainer = shap.TreeExplainer(
            model.model,
            data=background_data,
            model_output="probability",
        )
        if pdf.shape[0] >= 10_000:
            logging.warning(
                "Calculating SHAP values for more than 10,000 rows. This may take a while..."
            )
        shap_values = explainer.shap_values(
            pdf.to_numpy(),
            check_additivity=False,
            approximate=True,
        )
        base_value = explainer.expected_value
        return (base_value, shap_values)

    def add_features(
        self: L2GPrediction,
        feature_matrix: L2GFeatureMatrix,
    ) -> L2GPrediction:
        """Add features used to extract the L2G predictions.

        Args:
            feature_matrix (L2GFeatureMatrix): Feature matrix dataset

        Returns:
            L2GPrediction: L2G predictions with additional column `features`

        Raises:
            ValueError: If model is not set, feature list won't be available
        """
        if self.model is None:
            raise ValueError("Model not set, feature annotation cannot be created.")
        # Testing if `features` column already exists:
        if "features" in self.df.columns:
            self.df = self.df.drop("features")

        features_list = self.model.features_list
        feature_expressions = [
            f.struct(f.lit(col).alias("name"), f.col(col).alias("value"))
            for col in features_list
        ]
        self.df = self.df.join(
            feature_matrix._df.select(*features_list, "studyLocusId", "geneId"),
            on=["studyLocusId", "geneId"],
            how="left",
        ).select(
            "studyLocusId",
            "geneId",
            "score",
            f.array(*feature_expressions).alias("features"),
        )
        return self
