"""Utilities to train and apply the Locus to Gene classifier."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from wandb.data_types import Image, Table
from wandb.errors.term import termlog as wandb_termlog
from wandb.sdk.wandb_init import init as wandb_init
from wandb.sdk.wandb_setup import _setup
from wandb.sdk.wandb_sweep import sweep as wandb_sweep
from wandb.sklearn import plot_classifier
from wandb.wandb_agent import agent as wandb_agent

from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.method.l2g.model import LocusToGeneModel

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes
    from shap._explanation import Explanation
    from wandb.sdk.wandb_run import Run


def reset_wandb_env() -> None:
    """Reset Wandb environment variables except for project, entity and API key.

    This is necessary to log multiple runs in the same sweep without overwriting. More context here: https://github.com/wandb/wandb/issues/5119
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in list(os.environ.keys()):
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


@dataclass
class LocusToGeneTrainer:
    """Modelling of what is the most likely causal gene associated with a given locus."""

    model: LocusToGeneModel
    feature_matrix: L2GFeatureMatrix

    # Initialise vars
    features_list: list[str] | None = None
    label_col: str = "goldStandardSet"
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    groups_train: np.ndarray | None = None
    run: Run | None = None
    wandb_l2g_project_name: str = "gentropy-locus-to-gene"

    def __post_init__(self) -> None:
        """Set default features_list to feature_matrix's features_list if not provided."""
        self.features_list = (
            self.feature_matrix.features_list
            if self.features_list is None
            else self.features_list
        )

    def fit(
        self: LocusToGeneTrainer,
    ) -> LocusToGeneModel:
        """Fit the pipeline to the feature matrix dataframe.

        Returns:
            LocusToGeneModel: Fitted model

        Raises:
            ValueError: Train data not set, nothing to fit.
            AssertionError: When x_train_size or y_train_size are not zero.
        """
        if (
            self.x_train is not None
            and self.y_train is not None
            and self.features_list is not None
        ):
            assert self.x_train.size != 0 and self.y_train.size != 0, (
                "Train data not set, nothing to fit."
            )
            fitted_model = self.model.model.fit(X=self.x_train, y=self.y_train)
            self.model = LocusToGeneModel(
                model=fitted_model,
                hyperparameters=fitted_model.get_params(),
                training_data=self.feature_matrix,
                features_list=self.features_list,
            )
            return self.model
        raise ValueError("Train data not set, nothing to fit.")

    def _get_shap_explanation(
        self: LocusToGeneTrainer,
        model: LocusToGeneModel,
    ) -> Explanation:
        """Get the SHAP values for the given model and data. We pass the full X matrix (without the labels) to interpret their shap values.

        Args:
            model (LocusToGeneModel): Model to explain.

        Returns:
                Explanation: SHAP values for the given model and data.

        Raises:
            ValueError: Train data not set, cannot get SHAP values.
            Exception: (ExplanationError) When the additivity check fails.
        """
        if self.x_train is not None and self.x_test is not None:
            training_data = pd.DataFrame(
                np.vstack((self.x_train, self.x_test)),
                columns=self.features_list,
            )
            explainer = shap.TreeExplainer(
                model.model,
                data=training_data,
                feature_perturbation="interventional",
            )
            try:
                return explainer(training_data)
            except Exception as e:
                if "Additivity check failed in TreeExplainer" in repr(e):
                    return explainer(training_data, check_additivity=False)
                else:
                    raise

        raise ValueError("Train data not set.")

    def log_plot_image_to_wandb(
        self: LocusToGeneTrainer, title: str, plot: Axes
    ) -> None:
        """Accepts a plot object, and saves the fig to PNG to then log it in W&B.

        Args:
            title (str): Title of the plot.
            plot (Axes): Shap plot to log.

        Raises:
            ValueError: Run not set, cannot log to W&B.
        """
        if self.run is None:
            raise ValueError("Run not set, cannot log to W&B.")
        if not plot:
            # Scatter plot returns none, so we need to handle this case
            plt.savefig("tmp.png", bbox_inches="tight")
        else:
            plot.figure.savefig("tmp.png", bbox_inches="tight")
        self.run.log({title: Image("tmp.png")})
        plt.close()
        os.remove("tmp.png")

    def log_to_wandb(
        self: LocusToGeneTrainer,
        wandb_run_name: str,
    ) -> None:
        """Log evaluation results and feature importance to W&B to compare between different L2G runs.

        Dashboard is available at https://wandb.ai/open-targets/gentropy-locus-to-gene?nw=nwuseropentargets
        Credentials to access W&B are available at the OT central login sheet.

        Args:
            wandb_run_name (str): Name of the W&B run

        Raises:
            RuntimeError: If dependencies are not available.
            AssertionError: When x_train_size or y_train_size are not zero.
        """
        if (
            self.x_train is None
            or self.x_test is None
            or self.y_train is None
            or self.y_test is None
            or self.features_list is None
        ):
            raise RuntimeError("Train data not set, we cannot log to W&B.")
        assert self.x_train.size != 0 and self.y_train.size != 0, (
            "Train data not set, nothing to evaluate."
        )
        fitted_classifier = self.model.model
        y_predicted = fitted_classifier.predict(self.x_test)
        y_probas = fitted_classifier.predict_proba(self.x_test)
        self.run = wandb_init(
            project=self.wandb_l2g_project_name,
            name=wandb_run_name,
            config=fitted_classifier.get_params(),
        )
        # Track classification plots
        plot_classifier(
            self.model.model,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            y_predicted,
            y_probas,
            labels=list(self.model.label_encoder.values()),
            model_name="L2G-classifier",
            feature_names=self.features_list,
            is_binary=True,
        )
        # Track evaluation metrics
        self.run.log(
            {
                "areaUnderROC": roc_auc_score(
                    self.y_test, y_probas[:, 1], average="weighted"
                )
            }
        )
        self.run.log({"accuracy": accuracy_score(self.y_test, y_predicted)})
        self.run.log(
            {
                "weightedPrecision": precision_score(
                    self.y_test, y_predicted, average="weighted"
                )
            }
        )
        self.run.log(
            {
                "averagePrecision": average_precision_score(
                    self.y_test, y_predicted, average="weighted"
                )
            }
        )
        self.run.log(
            {
                "weightedRecall": recall_score(
                    self.y_test, y_predicted, average="weighted"
                )
            }
        )
        self.run.log({"f1": f1_score(self.y_test, y_predicted, average="weighted")})
        # Track gold standards and their features
        self.run.log(
            {"featureMatrix": Table(dataframe=self.feature_matrix._df.toPandas())}
        )
        # Log feature missingness
        self.run.log(
            {
                "missingnessRates": self.feature_matrix.calculate_feature_missingness_rate()
            }
        )
        # Plot marginal contribution of each feature
        explanation = self._get_shap_explanation(self.model)
        self.log_plot_image_to_wandb(
            "Feature Contribution",
            shap.plots.bar(
                explanation, max_display=len(self.features_list), show=False
            ),
        )
        self.log_plot_image_to_wandb(
            "Beeswarm Plot",
            shap.plots.beeswarm(
                explanation, max_display=len(self.features_list), show=False
            ),
        )
        # Plot correlation between feature values and their importance
        for feature in self.features_list:
            self.log_plot_image_to_wandb(
                f"Effect of {feature} on the predictions",
                shap.plots.scatter(
                    explanation[:, feature],
                    show=False,
                ),
            )
        wandb_termlog("Logged Shapley contributions.")
        self.run.finish()

    def train(
        self: LocusToGeneTrainer,
        wandb_run_name: str,
        cross_validate: bool = True,
        n_splits: int = 5,
        hyperparameter_grid: dict[str, Any] | None = None,
    ) -> LocusToGeneModel:
        """Train the Locus to Gene model.

        If cross_validation is set to True, we implement the following strategy:
            1. Create held-out test set
            2. Perform cross-validation on training set
            3. Train final model on full training set
            4. Evaluate once on test set

        Args:
            wandb_run_name (str): Name of the W&B run. Unless this is provided, the model will not be logged to W&B.
            cross_validate (bool): Whether to run cross-validation. Defaults to True.
            n_splits(int): Number of folds the data is splitted in. The model is trained and evaluated `k - 1` times. Defaults to 5.
            hyperparameter_grid (dict[str, Any] | None): Hyperparameter grid to sweep over. Defaults to None.

        Returns:
            LocusToGeneModel: Fitted model
        """
        data_df = self.feature_matrix._df.toPandas()
        # enforce that data_df is a Pandas DataFrame

        # Encode labels in `goldStandardSet` to a numeric value
        data_df[self.label_col] = data_df[self.label_col].map(self.model.label_encoder)

        X = data_df[self.features_list].apply(pd.to_numeric).values
        y = data_df[self.label_col].apply(pd.to_numeric).values
        gene_trait_groups = (
            data_df["traitFromSourceMappedId"].astype(str)
            + "_"
            + data_df["geneId"].astype(str)
        )  # Group identifier has to be a single string

        # Create hold-out test set separating EFO/Gene pairs between train/test
        train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in train_test_split.split(X, y, gene_trait_groups):
            self.x_train, self.x_test = X[train_idx], X[test_idx]
            self.y_train, self.y_test = y[train_idx], y[test_idx]
            self.groups_train = gene_trait_groups[train_idx]

        # Cross-validation
        if cross_validate:
            self.cross_validate(
                wandb_run_name=f"{wandb_run_name}-cv",
                parameter_grid=hyperparameter_grid,
                n_splits=n_splits,
            )

        # Train final model on full training set
        self.fit()

        # Evaluate once on hold out test set
        self.log_to_wandb(
            wandb_run_name=f"{wandb_run_name}-holdout",
        )

        return self.model

    def cross_validate(
        self: LocusToGeneTrainer,
        wandb_run_name: str,
        parameter_grid: dict[str, Any] | None = None,
        n_splits: int = 5,
    ) -> None:
        """Log results of cross validation and hyperparameter tuning with W&B Sweeps. Metrics for every combination of hyperparameters will be logged to W&B for comparison.

        Args:
            wandb_run_name (str): Name of the W&B run
            parameter_grid (dict[str, Any] | None): Dictionary containing the hyperparameters to sweep over. The keys are the hyperparameter names, and the values are dictionaries containing the values to sweep over.
            n_splits (int): Number of folds the data is splitted in. The model is trained and evaluated `k - 1` times. Defaults to 5.
        """

        def cross_validate_single_fold(
            fold_index: int,
            sweep_id: str,
            sweep_run_name: str,
            config: dict[str, Any],
        ) -> None:
            """Run cross-validation for a single fold.

            Args:
                fold_index (int): Index of the fold to run
                sweep_id (str): ID of the sweep
                sweep_run_name (str): Name of the sweep run
                config (dict[str, Any]): Configuration from the sweep

            Raises:
                ValueError: If training data is not set
            """
            reset_wandb_env()
            train_idx, val_idx = cv_splits[fold_index]

            if (
                self.x_train is None
                or self.y_train is None
                or self.groups_train is None
            ):
                raise ValueError("Training data not set")

            # Initialize a new run for this fold
            os.environ["WANDB_SWEEP_ID"] = sweep_id
            run = wandb_init(
                project=self.wandb_l2g_project_name,
                name=sweep_run_name,
                config=config,
                group=sweep_run_name,
                job_type="fold",
                reinit=True,
            )

            x_fold_train, x_fold_val = (
                self.x_train[train_idx],
                self.x_train[val_idx],
            )
            y_fold_train, y_fold_val = (
                self.y_train[train_idx],
                self.y_train[val_idx],
            )

            fold_model = clone(self.model.model)
            fold_model.set_params(**config)
            fold_model.fit(x_fold_train, y_fold_train)
            y_pred_proba = fold_model.predict_proba(x_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Log metrics
            metrics = {
                "weightedPrecision": precision_score(y_fold_val, y_pred),
                "averagePrecision": average_precision_score(y_fold_val, y_pred_proba),
                "areaUnderROC": roc_auc_score(y_fold_val, y_pred_proba),
                "accuracy": accuracy_score(y_fold_val, y_pred),
                "weightedRecall": recall_score(y_fold_val, y_pred, average="weighted"),
                "f1": f1_score(y_fold_val, y_pred, average="weighted"),
            }

            run.log(metrics)
            wandb_termlog(f"Logged metrics for fold {fold_index + 1}.")
            run.finish()

        # If no grid is provided, use default ones set in the model
        parameter_grid = parameter_grid or {
            param: {"values": [value]}
            for param, value in self.model.hyperparameters.items()
        }
        sweep_config = {
            "method": "grid",
            "name": wandb_run_name,  # Add name to sweep config
            "metric": {"name": "areaUnderROC", "goal": "maximize"},
            "parameters": parameter_grid,
        }
        sweep_id = wandb_sweep(sweep_config, project=self.wandb_l2g_project_name)

        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(self.x_train, self.y_train, self.groups_train))

        def run_all_folds() -> None:
            """Run cross-validation for all folds within a sweep."""
            # Initialize the sweep run and get metadata
            sweep_run = wandb_init(name=wandb_run_name)
            sweep_id = sweep_run.sweep_id or "unknown"
            sweep_url = sweep_run.get_sweep_url()
            project_url = sweep_run.get_project_url()
            sweep_group_url = f"{project_url}/groups/{sweep_id}"
            sweep_run.notes = sweep_group_url
            sweep_run.save()
            config = dict(sweep_run.config)

            # Reset wandb setup to ensure clean state
            _setup(_reset=True)

            # Run all folds
            for fold_index in range(len(cv_splits)):
                cross_validate_single_fold(
                    fold_index=fold_index,
                    sweep_id=sweep_id,
                    sweep_run_name=f"{wandb_run_name}-fold{fold_index + 1}",
                    config=config,
                )

            wandb_termlog(f"Sweep URL: {sweep_url}")
            wandb_termlog(f"Sweep Group URL: {sweep_group_url}")

        wandb_agent(sweep_id, run_all_folds)
