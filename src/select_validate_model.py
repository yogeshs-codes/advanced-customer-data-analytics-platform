"""
Phase 2 - Task 5: Model Selection and Validation

This script performs final model selection using:
    1. Stratified cross-validation
    2. Independent test-set evaluation
    3. Generalization-gap analysis
    4. Stability analysis
    5. Model-size comparison
    6. Final model selection
    7. Performance visualizations
    8. Final validation report

Candidate models:
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Tuned Random Forest

Input:
    output/training_data/train.csv.gz
    output/training_data/test.csv.gz

Models:
    output/models/decision_tree.joblib
    output/models/random_forest.joblib
    output/models/gradient_boosting.joblib
    output/models/tuned_random_forest.joblib

Outputs:
    output/model_results/task5_cross_validation_results.csv
    output/model_results/task5_validation_summary.csv
    output/model_results/task5_test_results.csv
    output/model_results/task5_model_comparison.csv
    output/model_results/task5_final_model.txt
    output/model_results/task5_validation_report.md
    output/model_results/task5_validation.log
    output/model_results/task5_model_comparison.png
    output/model_results/task5_roc_auc_comparison.png
    output/model_results/task5_f1_comparison.png
    output/model_results/task5_generalization_gap.png
    output/models/final_selected_model.joblib

Purpose:
    Select the most reliable model based on validated performance
    rather than relying only on a single train/test metric.
"""

from pathlib import Path
import logging
import shutil
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_data"
    / "train.csv.gz"
)

TEST_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_data"
    / "test.csv.gz"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "output"
    / "model_results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_COLUMN = "target"

ID_COLUMNS = [
    "user_id",
    "product_id",
]

RANDOM_STATE = 42

# Number of rows used for cross-validation.
# A stratified sample keeps the class distribution representative
# while avoiding unnecessary computation on the complete dataset.
CV_SAMPLE_SIZE = 150_000

N_SPLITS = 3

MODEL_FILES = {
    "decision_tree": MODELS_DIR / "decision_tree.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "gradient_boosting": MODELS_DIR / "gradient_boosting.joblib",
    "tuned_random_forest": MODELS_DIR / "tuned_random_forest.joblib",
}


# ============================================================
# 3. Logging
# ============================================================

LOG_FILE = RESULTS_DIR / "task5_validation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. Utility functions
# ============================================================

def get_model_size_mb(model_path):
    """Return serialized model size in MB."""

    if not model_path.exists():
        return np.nan

    size_bytes = model_path.stat().st_size

    return size_bytes / (
        1024 * 1024
    )


def get_predictive_features(
    train_df,
):
    """
    Return the features used by the ML models.

    Identifier columns and target column are excluded.
    """

    excluded_columns = set(
        ID_COLUMNS + [
            TARGET_COLUMN,
        ]
    )

    features = [
        column
        for column in train_df.columns
        if column not in excluded_columns
    ]

    return features


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):
    """
    Calculate classification performance metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
    }


def predict_probability(
    model,
    X,
):
    """
    Return positive-class probabilities.

    Most candidate classifiers expose predict_proba().
    Decision-function models are also supported.
    """

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            X
        )

        if probabilities.ndim == 2:
            return probabilities[:, 1]

        return probabilities

    if hasattr(
        model,
        "decision_function",
    ):
        scores = model.decision_function(
            X
        )

        # Convert arbitrary decision scores into
        # monotonic probabilities for ROC-AUC use.
        scores = np.asarray(
            scores,
            dtype=float,
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(
                scores,
                dtype=float,
            )

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )

    raise AttributeError(
        "Model does not provide "
        "predict_proba() or "
        "decision_function()."
    )


def load_models():
    """
    Load all previously trained candidate models.
    """

    logger.info("=" * 70)
    logger.info(
        "Loading trained candidate models."
    )

    models = {}

    for model_name, model_path in MODEL_FILES.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Required model not found: "
                f"{model_path}"
            )

        logger.info(
            "Loading %s from %s",
            model_name,
            model_path,
        )

        models[model_name] = joblib.load(
            model_path
        )

    return models


# ============================================================
# 5. Load data
# ============================================================

def load_data():
    """
    Load training and independent test datasets.
    """

    logger.info("=" * 70)
    logger.info(
        "Loading training and test data."
    )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: "
            f"{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: "
            f"{TEST_FILE}"
        )

    train_df = pd.read_csv(
        TRAIN_FILE,
        compression="gzip",
    )

    test_df = pd.read_csv(
        TEST_FILE,
        compression="gzip",
    )

    logger.info(
        "Training data shape: %s",
        train_df.shape,
    )

    logger.info(
        "Test data shape: %s",
        test_df.shape,
    )

    if TARGET_COLUMN not in train_df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in test data."
        )

    features = get_predictive_features(
        train_df
    )

    logger.info(
        "Number of predictive features: %d",
        len(features),
    )

    logger.info(
        "Predictive features: %s",
        features,
    )

    logger.info(
        "Training positive class rate: %.4f",
        train_df[TARGET_COLUMN].mean(),
    )

    logger.info(
        "Test positive class rate: %.4f",
        test_df[TARGET_COLUMN].mean(),
    )

    return (
        train_df,
        test_df,
        features,
    )


# ============================================================
# 6. Create stratified CV sample
# ============================================================

def create_cv_sample(
    train_df,
    features,
):
    """
    Create a stratified sample for cross-validation.

    The sample preserves the target class distribution.
    """

    logger.info(
        "Creating stratified CV sample of %d rows.",
        CV_SAMPLE_SIZE,
    )

    if CV_SAMPLE_SIZE >= len(train_df):
        sample_df = train_df.copy()

    else:
        positive_df = train_df[
            train_df[TARGET_COLUMN] == 1
        ]

        negative_df = train_df[
            train_df[TARGET_COLUMN] == 0
        ]

        positive_fraction = (
            len(positive_df)
            / len(train_df)
        )

        positive_size = int(
            CV_SAMPLE_SIZE
            * positive_fraction
        )

        negative_size = (
            CV_SAMPLE_SIZE
            - positive_size
        )

        positive_sample = (
            positive_df.sample(
                n=positive_size,
                random_state=RANDOM_STATE,
            )
        )

        negative_sample = (
            negative_df.sample(
                n=negative_size,
                random_state=RANDOM_STATE,
            )
        )

        sample_df = pd.concat(
            [
                positive_sample,
                negative_sample,
            ],
            axis=0,
        )

        sample_df = sample_df.sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        ).reset_index(
            drop=True
        )

    X_sample = sample_df[
        features
    ].copy()

    y_sample = sample_df[
        TARGET_COLUMN
    ].copy()

    logger.info(
        "CV sample shape: %s",
        X_sample.shape,
    )

    logger.info(
        "CV sample positive class rate: %.4f",
        y_sample.mean(),
    )

    return (
        X_sample,
        y_sample,
    )


# ============================================================
# 7. Cross-validation
# ============================================================

def perform_cross_validation(
    models,
    X,
    y,
):
    """
    Perform stratified K-fold cross-validation.

    IMPORTANT:
        A fresh clone of each trained model is fitted inside
        every fold. This prevents data leakage and produces
        genuine cross-validation estimates.
    """

    logger.info("=" * 70)
    logger.info(
        "Starting %d-fold stratified cross-validation.",
        N_SPLITS,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    for model_name, original_model in models.items():

        logger.info(
            "-" * 70
        )

        logger.info(
            "Cross-validating model: %s",
            model_name,
        )

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            logger.info(
                "%s - Fold %d/%d",
                model_name,
                fold_number,
                N_SPLITS,
            )

            X_train_fold = X.iloc[
                train_indices
            ]

            X_validation_fold = X.iloc[
                validation_indices
            ]

            y_train_fold = y.iloc[
                train_indices
            ]

            y_validation_fold = y.iloc[
                validation_indices
            ]

            # Create an unfitted copy.
            fold_model = clone(
                original_model
            )

            start_time = time.perf_counter()

            # Fit only on the fold training data.
            fold_model.fit(
                X_train_fold,
                y_train_fold,
            )

            predictions = (
                fold_model.predict(
                    X_validation_fold
                )
            )

            probabilities = (
                predict_probability(
                    fold_model,
                    X_validation_fold,
                )
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            metrics = calculate_metrics(
                y_validation_fold,
                predictions,
                probabilities,
            )

            row = {
                "model": model_name,
                "fold": fold_number,
                **metrics,
                "training_time_seconds": elapsed,
            }

            fold_rows.append(
                row
            )

            logger.info(
                "%s fold %d | "
                "ROC-AUC=%.6f | "
                "F1=%.6f | "
                "Recall=%.6f",
                model_name,
                fold_number,
                metrics["roc_auc"],
                metrics["f1_score"],
                metrics["recall"],
            )

    fold_results = pd.DataFrame(
        fold_rows
    )

    output_file = (
        RESULTS_DIR
        / "task5_cross_validation_results.csv"
    )

    fold_results.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved CV fold results: %s",
        output_file,
    )

    return fold_results


# ============================================================
# 8. Summarize cross-validation
# ============================================================

def summarize_cross_validation(
    fold_results,
):
    """
    Calculate mean and standard deviation
    for every CV metric.
    """

    logger.info("=" * 70)
    logger.info(
        "Summarizing cross-validation results."
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
    ]

    summary_rows = []

    for model_name in (
        fold_results[
            "model"
        ].unique()
    ):

        model_rows = fold_results[
            fold_results["model"]
            == model_name
        ]

        row = {
            "model": model_name,
        }

        for metric in metric_columns:

            row[
                f"{metric}_mean"
            ] = model_rows[
                metric
            ].mean()

            row[
                f"{metric}_std"
            ] = model_rows[
                metric
            ].std(
                ddof=1
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        by=[
            "roc_auc_mean",
            "f1_score_mean",
            "recall_mean",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    output_file = (
        RESULTS_DIR
        / "task5_validation_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved CV summary: %s",
        output_file,
    )

    return summary


# ============================================================
# 9. Independent test-set evaluation
# ============================================================

def evaluate_independent_test_set(
    models,
    X_test,
    y_test,
):
    """
    Evaluate the original trained production candidates
    on the untouched independent test set.
    """

    logger.info("=" * 70)
    logger.info(
        "Evaluating models on independent test set."
    )

    rows = []

    for model_name, model in models.items():

        logger.info(
            "Evaluating test performance: %s",
            model_name,
        )

        start_time = (
            time.perf_counter()
        )

        predictions = model.predict(
            X_test
        )

        probabilities = (
            predict_probability(
                model,
                X_test,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        model_path = MODEL_FILES[
            model_name
        ]

        model_size_mb = (
            get_model_size_mb(
                model_path
            )
        )

        row = {
            "model": model_name,
            **metrics,
            "model_size_mb": model_size_mb,
            "test_prediction_time_seconds": elapsed,
        }

        rows.append(
            row
        )

        logger.info(
            "%s | "
            "Accuracy=%.6f | "
            "Precision=%.6f | "
            "Recall=%.6f | "
            "F1=%.6f | "
            "ROC-AUC=%.6f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

    test_results = pd.DataFrame(
        rows
    )

    output_file = (
        RESULTS_DIR
        / "task5_test_results.csv"
    )

    test_results.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved test results: %s",
        output_file,
    )

    return test_results


# ============================================================
# 10. Combine validation evidence
# ============================================================

def create_model_comparison(
    cv_summary,
    test_results,
):
    """
    Combine cross-validation and independent test results.

    The independent-test metric names are explicitly renamed
    before merging to prevent ambiguous pandas suffix behavior.
    """

    logger.info("=" * 70)
    logger.info(
        "Creating final model comparison."
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    # --------------------------------------------------------
    # Explicitly rename test metrics.
    # This guarantees that columns such as roc_auc_test
    # actually exist after the merge.
    # --------------------------------------------------------

    test_results_renamed = (
        test_results.rename(
            columns={
                "accuracy":
                    "accuracy_test",
                "precision":
                    "precision_test",
                "recall":
                    "recall_test",
                "f1_score":
                    "f1_score_test",
                "roc_auc":
                    "roc_auc_test",
                "model_size_mb":
                    "model_size_mb",
                "test_prediction_time_seconds":
                    "test_prediction_time_seconds",
            }
        )
    )

    comparison = cv_summary.merge(
        test_results_renamed,
        on="model",
        how="left",
    )

    # --------------------------------------------------------
    # Generalization gaps
    # --------------------------------------------------------

    comparison[
        "roc_auc_generalization_gap"
    ] = (
        comparison["roc_auc_mean"]
        - comparison["roc_auc_test"]
    )

    comparison[
        "f1_generalization_gap"
    ] = (
        comparison["f1_score_mean"]
        - comparison["f1_score_test"]
    )

    # Absolute gap is useful when measuring
    # generalization stability.
    comparison[
        "abs_roc_auc_generalization_gap"
    ] = comparison[
        "roc_auc_generalization_gap"
    ].abs()

    comparison[
        "abs_f1_generalization_gap"
    ] = comparison[
        "f1_generalization_gap"
    ].abs()

    # --------------------------------------------------------
    # Stability score
    # --------------------------------------------------------

    comparison[
        "roc_auc_stability"
    ] = (
        1
        / (
            comparison[
                "roc_auc_std"
            ]
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Validation rank
    # --------------------------------------------------------

    comparison = comparison.sort_values(
        by=[
            "roc_auc_mean",
            "f1_score_mean",
            "recall_mean",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    comparison[
        "validation_rank"
    ] = np.arange(
        1,
        len(comparison) + 1,
    )

    output_file = (
        RESULTS_DIR
        / "task5_model_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved model comparison: %s",
        output_file,
    )

    return comparison


# ============================================================
# 11. Final model selection
# ============================================================

def select_final_model(
    comparison,
):
    """
    Select the final production model.

    Selection priorities:
        1. Cross-validation ROC-AUC
        2. Cross-validation F1
        3. Cross-validation recall
        4. Low ROC-AUC variability
        5. Independent test ROC-AUC

    ROC-AUC is the primary metric because the original
    customer reorder test set is highly imbalanced.
    """

    logger.info("=" * 70)
    logger.info(
        "Selecting final model."
    )

    ranked = comparison.copy()

    # Primary selection is based on validated CV performance.
    ranked = ranked.sort_values(
        by=[
            "roc_auc_mean",
            "f1_score_mean",
            "recall_mean",
            "roc_auc_std",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    selected_model = (
        ranked.iloc[0]["model"]
    )

    selected_row = ranked.iloc[0]

    logger.info(
        "Selected final model: %s",
        selected_model,
    )

    logger.info(
        "CV ROC-AUC: %.6f",
        selected_row[
            "roc_auc_mean"
        ],
    )

    logger.info(
        "CV F1: %.6f",
        selected_row[
            "f1_score_mean"
        ],
    )

    logger.info(
        "Test ROC-AUC: %.6f",
        selected_row[
            "roc_auc_test"
        ],
    )

    logger.info(
        "Test F1: %.6f",
        selected_row[
            "f1_score_test"
        ],
    )

    return (
        selected_model,
        selected_row,
        ranked,
    )


# ============================================================
# 12. Save final selected model
# ============================================================

def save_final_model(
    selected_model,
    models,
):
    """
    Copy the selected trained production model
    to a standard final model path.
    """

    source_path = MODEL_FILES[
        selected_model
    ]

    destination_path = (
        MODELS_DIR
        / "final_selected_model.joblib"
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    logger.info(
        "Saved final selected model: %s",
        destination_path,
    )

    return destination_path


# ============================================================
# 13. Visualization helpers
# ============================================================

def create_model_comparison_plot(
    comparison,
):
    """
    Create a grouped comparison of CV ROC-AUC and test ROC-AUC.
    """

    ordered = comparison.sort_values(
        "roc_auc_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(ordered)
    )

    ax.barh(
        y_positions - 0.18,
        ordered[
            "roc_auc_mean"
        ],
        height=0.35,
        label="CV ROC-AUC",
    )

    ax.barh(
        y_positions + 0.18,
        ordered[
            "roc_auc_test"
        ],
        height=0.35,
        label="Test ROC-AUC",
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        ordered["model"]
    )

    ax.set_xlabel(
        "ROC-AUC"
    )

    ax.set_title(
        "Model ROC-AUC: Cross-Validation vs Independent Test"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "task5_model_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved model comparison plot: %s",
        output_file,
    )


def create_roc_auc_plot(
    comparison,
):
    """
    Plot cross-validation ROC-AUC with standard deviation.
    """

    ordered = comparison.sort_values(
        "roc_auc_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(ordered)
    )

    ax.barh(
        y_positions,
        ordered[
            "roc_auc_mean"
        ],
        xerr=ordered[
            "roc_auc_std"
        ],
        capsize=5,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        ordered["model"]
    )

    ax.set_xlabel(
        "Cross-Validation ROC-AUC"
    )

    ax.set_title(
        "Cross-Validation ROC-AUC with Standard Deviation"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "task5_roc_auc_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved ROC-AUC plot: %s",
        output_file,
    )


def create_f1_plot(
    comparison,
):
    """
    Compare CV F1 and independent test F1.
    """

    ordered = comparison.sort_values(
        "f1_score_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(ordered)
    )

    ax.barh(
        y_positions - 0.18,
        ordered[
            "f1_score_mean"
        ],
        height=0.35,
        label="CV F1",
    )

    ax.barh(
        y_positions + 0.18,
        ordered[
            "f1_score_test"
        ],
        height=0.35,
        label="Test F1",
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        ordered["model"]
    )

    ax.set_xlabel(
        "F1 Score"
    )

    ax.set_title(
        "Model F1 Score: Cross-Validation vs Independent Test"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "task5_f1_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved F1 comparison plot: %s",
        output_file,
    )


def create_generalization_gap_plot(
    comparison,
):
    """
    Visualize absolute ROC-AUC generalization gaps.
    """

    ordered = comparison.sort_values(
        "abs_roc_auc_generalization_gap",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(ordered)
    )

    ax.barh(
        y_positions,
        ordered[
            "abs_roc_auc_generalization_gap"
        ],
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        ordered["model"]
    )

    ax.set_xlabel(
        "Absolute ROC-AUC Generalization Gap"
    )

    ax.set_title(
        "Model Generalization Stability"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "task5_generalization_gap.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved generalization-gap plot: %s",
        output_file,
    )


# ============================================================
# 14. Save final-model information
# ============================================================

def save_final_model_information(
    selected_model,
    selected_row,
    final_model_path,
):
    """
    Save a concise text summary of the final selection.
    """

    output_file = (
        RESULTS_DIR
        / "task5_final_model.txt"
    )

    lines = [
        "PHASE 2 - TASK 5 FINAL MODEL SELECTION",
        "=" * 55,
        "",
        f"Selected model: {selected_model}",
        "",
        "Cross-validation performance:",
        (
            f"ROC-AUC mean: "
            f"{selected_row['roc_auc_mean']:.6f}"
        ),
        (
            f"ROC-AUC std: "
            f"{selected_row['roc_auc_std']:.6f}"
        ),
        (
            f"F1 mean: "
            f"{selected_row['f1_score_mean']:.6f}"
        ),
        (
            f"F1 std: "
            f"{selected_row['f1_score_std']:.6f}"
        ),
        (
            f"Recall mean: "
            f"{selected_row['recall_mean']:.6f}"
        ),
        "",
        "Independent test performance:",
        (
            f"Accuracy: "
            f"{selected_row['accuracy_test']:.6f}"
        ),
        (
            f"Precision: "
            f"{selected_row['precision_test']:.6f}"
        ),
        (
            f"Recall: "
            f"{selected_row['recall_test']:.6f}"
        ),
        (
            f"F1: "
            f"{selected_row['f1_score_test']:.6f}"
        ),
        (
            f"ROC-AUC: "
            f"{selected_row['roc_auc_test']:.6f}"
        ),
        "",
        "Generalization:",
        (
            f"ROC-AUC generalization gap: "
            f"{selected_row['roc_auc_generalization_gap']:.6f}"
        ),
        (
            f"F1 generalization gap: "
            f"{selected_row['f1_generalization_gap']:.6f}"
        ),
        "",
        "Model:",
        (
            f"Serialized model size: "
            f"{selected_row['model_size_mb']:.4f} MB"
        ),
        f"Final model path: {final_model_path}",
        "",
    ]

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Saved final model information: %s",
        output_file,
    )


# ============================================================
# 15. Generate Markdown validation report
# ============================================================

def generate_validation_report(
    comparison,
    selected_model,
    selected_row,
):
    """
    Generate a complete Markdown report for Task 5.
    """

    report_file = (
        RESULTS_DIR
        / "task5_validation_report.md"
    )

    report_lines = []

    report_lines.append(
        "# Phase 2 - Task 5: Model Selection and Validation"
    )

    report_lines.append("")

    report_lines.append(
        "## Objective"
    )

    report_lines.append("")

    report_lines.append(
        "The objective of Task 5 was to select the most effective "
        "machine learning model for customer-product reorder "
        "prediction using cross-validation, independent test-set "
        "evaluation, and generalization analysis."
    )

    report_lines.append("")

    report_lines.append(
        "## Models Evaluated"
    )

    report_lines.append("")

    report_lines.append(
        "- Decision Tree"
    )

    report_lines.append(
        "- Random Forest"
    )

    report_lines.append(
        "- Gradient Boosting"
    )

    report_lines.append(
        "- Tuned Random Forest"
    )

    report_lines.append("")

    report_lines.append(
        "## Validation Method"
    )

    report_lines.append("")

    report_lines.append(
        f"A stratified sample of {CV_SAMPLE_SIZE:,} training "
        f"observations was evaluated using {N_SPLITS}-fold "
        "StratifiedKFold cross-validation. A fresh clone of "
        "each candidate model was fitted inside every fold to "
        "avoid information leakage."
    )

    report_lines.append("")

    report_lines.append(
        "The final trained candidate models were subsequently "
        "evaluated on the independent test set."
    )

    report_lines.append("")

    report_lines.append(
        "## Model Comparison"
    )

    report_lines.append("")

    report_lines.append(
        "| Model | CV ROC-AUC | CV F1 | CV Recall | "
        "Test ROC-AUC | Test F1 | Test Recall | "
        "ROC-AUC Gap |"
    )

    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in comparison.iterrows():

        report_lines.append(
            f"| {row['model']} | "
            f"{row['roc_auc_mean']:.4f} | "
            f"{row['f1_score_mean']:.4f} | "
            f"{row['recall_mean']:.4f} | "
            f"{row['roc_auc_test']:.4f} | "
            f"{row['f1_score_test']:.4f} | "
            f"{row['recall_test']:.4f} | "
            f"{row['roc_auc_generalization_gap']:.4f} |"
        )

    report_lines.append("")

    report_lines.append(
        "## Selected Model"
    )

    report_lines.append("")

    report_lines.append(
        f"**{selected_model}** was selected as the final model."
    )

    report_lines.append("")

    report_lines.append(
        "The primary selection criterion was mean "
        "cross-validation ROC-AUC because the customer reorder "
        "prediction problem has an imbalanced independent test "
        "distribution. Cross-validation F1 and recall were used "
        "as supporting metrics, while ROC-AUC variability was "
        "considered as a stability indicator."
    )

    report_lines.append("")

    report_lines.append(
        "### Selected Model Performance"
    )

    report_lines.append("")

    report_lines.append(
        f"- CV ROC-AUC: "
        f"**{selected_row['roc_auc_mean']:.4f}**"
    )

    report_lines.append(
        f"- CV ROC-AUC standard deviation: "
        f"**{selected_row['roc_auc_std']:.4f}**"
    )

    report_lines.append(
        f"- CV F1: "
        f"**{selected_row['f1_score_mean']:.4f}**"
    )

    report_lines.append(
        f"- CV Recall: "
        f"**{selected_row['recall_mean']:.4f}**"
    )

    report_lines.append(
        f"- Test Accuracy: "
        f"**{selected_row['accuracy_test']:.4f}**"
    )

    report_lines.append(
        f"- Test Precision: "
        f"**{selected_row['precision_test']:.4f}**"
    )

    report_lines.append(
        f"- Test Recall: "
        f"**{selected_row['recall_test']:.4f}**"
    )

    report_lines.append(
        f"- Test F1: "
        f"**{selected_row['f1_score_test']:.4f}**"
    )

    report_lines.append(
        f"- Test ROC-AUC: "
        f"**{selected_row['roc_auc_test']:.4f}**"
    )

    report_lines.append("")

    report_lines.append(
        "## Generalization Analysis"
    )

    report_lines.append("")

    report_lines.append(
        f"The selected model produced a ROC-AUC generalization "
        f"gap of **{selected_row['roc_auc_generalization_gap']:.4f}** "
        "between cross-validation and independent test evaluation. "
        "A relatively small gap indicates that the model's "
        "performance remains consistent when evaluated on unseen "
        "data."
    )

    report_lines.append("")

    report_lines.append(
        "## Conclusion"
    )

    report_lines.append("")

    report_lines.append(
        f"The validation results support **{selected_model}** "
        "as the final candidate for customer-product reorder "
        "prediction. The selection was based on systematic "
        "cross-validation and independent test-set evidence "
        "rather than a single performance measurement."
    )

    report_lines.append("")

    report_lines.append(
        "## Generated Artifacts"
    )

    report_lines.append("")

    report_lines.append(
        "- `task5_cross_validation_results.csv`"
    )

    report_lines.append(
        "- `task5_validation_summary.csv`"
    )

    report_lines.append(
        "- `task5_test_results.csv`"
    )

    report_lines.append(
        "- `task5_model_comparison.csv`"
    )

    report_lines.append(
        "- `task5_final_model.txt`"
    )

    report_lines.append(
        "- `task5_model_comparison.png`"
    )

    report_lines.append(
        "- `task5_roc_auc_comparison.png`"
    )

    report_lines.append(
        "- `task5_f1_comparison.png`"
    )

    report_lines.append(
        "- `task5_generalization_gap.png`"
    )

    report_lines.append(
        "- `final_selected_model.joblib`"
    )

    report_lines.append("")

    report_file.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved validation report: %s",
        report_file,
    )


# ============================================================
# 16. Print final comparison
# ============================================================

def print_final_summary(
    comparison,
    selected_model,
):
    """
    Print an easy-to-read final summary.
    """

    print()
    print("=" * 80)
    print(
        "PHASE 2 - TASK 5 FINAL MODEL VALIDATION"
    )
    print("=" * 80)

    display_columns = [
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "f1_score_mean",
        "recall_mean",
        "roc_auc_test",
        "f1_score_test",
        "recall_test",
        "roc_auc_generalization_gap",
    ]

    display_df = comparison[
        display_columns
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"FINAL SELECTED MODEL: {selected_model}"
    )

    print(
        "=" * 80
    )

    print()


# ============================================================
# 17. Main workflow
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    logger.info("=" * 70)
    logger.info(
        "Starting Phase 2 - Task 5 model selection and validation."
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        features,
    ) = load_data()

    # --------------------------------------------------------
    # Load trained models
    # --------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------
    # Prepare feature matrices
    # --------------------------------------------------------

    X_test = test_df[
        features
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # Create stratified CV sample
    # --------------------------------------------------------

    (
        X_cv,
        y_cv,
    ) = create_cv_sample(
        train_df,
        features,
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv_fold_results = (
        perform_cross_validation(
            models,
            X_cv,
            y_cv,
        )
    )

    # --------------------------------------------------------
    # CV summary
    # --------------------------------------------------------

    cv_summary = (
        summarize_cross_validation(
            cv_fold_results
        )
    )

    # --------------------------------------------------------
    # Independent test evaluation
    # --------------------------------------------------------

    test_results = (
        evaluate_independent_test_set(
            models,
            X_test,
            y_test,
        )
    )

    # --------------------------------------------------------
    # Combine validation evidence
    # --------------------------------------------------------

    comparison = (
        create_model_comparison(
            cv_summary,
            test_results,
        )
    )

    # --------------------------------------------------------
    # Select final model
    # --------------------------------------------------------

    (
        selected_model,
        selected_row,
        ranked_comparison,
    ) = select_final_model(
        comparison
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        save_final_model(
            selected_model,
            models,
        )
    )

    # --------------------------------------------------------
    # Save final model information
    # --------------------------------------------------------

    save_final_model_information(
        selected_model,
        selected_row,
        final_model_path,
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    create_model_comparison_plot(
        comparison
    )

    create_roc_auc_plot(
        comparison
    )

    create_f1_plot(
        comparison
    )

    create_generalization_gap_plot(
        comparison
    )

    # --------------------------------------------------------
    # Generate Markdown report
    # --------------------------------------------------------

    generate_validation_report(
        comparison,
        selected_model,
        selected_row,
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print_final_summary(
        comparison,
        selected_model,
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    logger.info("=" * 70)
    logger.info(
        "Task 5 completed successfully."
    )

    logger.info(
        "Final selected model: %s",
        selected_model,
    )

    logger.info(
        "Total execution time: %.2f seconds",
        total_elapsed,
    )

    logger.info("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()