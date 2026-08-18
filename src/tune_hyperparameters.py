"""
Phase 2 - Task 2: Hyperparameter Tuning

This script tunes the Random Forest model selected from Phase 2 - Task 1
using RandomizedSearchCV.

Input:
    output/training_data/train.csv.gz
    output/training_data/test.csv.gz

Baseline model:
    Random Forest from Phase 2 - Task 1

Tuning method:
    RandomizedSearchCV with stratified cross-validation

Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC

Outputs:
    output/tuning_results/random_search_results.csv
    output/tuning_results/tuning_metrics.csv
    output/tuning_results/tuning.log
    output/tuning_results/best_parameters.txt
    output/models/tuned_random_forest.joblib
"""

from pathlib import Path
import logging
import time

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)


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

TUNING_RESULTS_DIR = (
    PROJECT_ROOT
    / "output"
    / "tuning_results"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
)

TUNING_RESULTS_DIR.mkdir(
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

# The complete training dataset contains more than 1.3 million
# rows. A representative stratified sample is used during the
# hyperparameter search to keep the search computationally
# practical. The final tuned model is retrained on all training data.
TUNING_SAMPLE_SIZE = 250_000

VALIDATION_SIZE = 0.20

# Number of random hyperparameter combinations evaluated.
N_ITER_SEARCH = 8

# Two-fold CV provides a balance between validation reliability
# and computational cost for this large dataset.
CV_FOLDS = 2

# Limit parallel workers to reduce memory pressure on local machines.
N_JOBS = 2


# ============================================================
# 3. Configure logging
# ============================================================

LOG_FILE = (
    TUNING_RESULTS_DIR
    / "tuning.log"
)

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
# 4. Load data
# ============================================================

def load_data():
    """Load the prepared training and test datasets."""

    logger.info("Loading training data...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    logger.info("Loading test data...")

    test_df = pd.read_csv(
        TEST_FILE
    )

    logger.info(
        "Training data shape: %s",
        train_df.shape,
    )

    logger.info(
        "Test data shape: %s",
        test_df.shape,
    )

    return train_df, test_df


# ============================================================
# 5. Prepare features
# ============================================================

def prepare_features(
    train_df,
    test_df,
):
    """
    Separate target from predictors and remove identifier columns.
    """

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in test data."
        )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in ID_COLUMNS
        and column != TARGET_COLUMN
    ]

    if not feature_columns:
        raise ValueError(
            "No predictive features were found."
        )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    logger.info(
        "Number of predictive features: %d",
        len(feature_columns),
    )

    logger.info(
        "Predictive features: %s",
        feature_columns,
    )

    logger.info(
        "Training target distribution:\n%s",
        y_train.value_counts(),
    )

    logger.info(
        "Test target distribution:\n%s",
        y_test.value_counts(),
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
    )


# ============================================================
# 6. Create tuning sample
# ============================================================

def create_tuning_sample(
    X_train,
    y_train,
):
    """
    Create a representative stratified sample for hyperparameter
    search.

    The sample is drawn only from the training data. The original
    test dataset remains completely untouched.
    """

    sample_size = min(
        TUNING_SAMPLE_SIZE,
        len(X_train),
    )

    if sample_size < len(X_train):

        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    else:

        X_sample = X_train.copy()
        y_sample = y_train.copy()

    logger.info(
        "Hyperparameter tuning sample shape: %s",
        X_sample.shape,
    )

    logger.info(
        "Tuning sample target distribution:\n%s",
        y_sample.value_counts(),
    )

    return X_sample, y_sample


# ============================================================
# 7. Split tuning sample into training and validation sets
# ============================================================

def create_validation_split(
    X_sample,
    y_sample,
):
    """
    Create a stratified validation set from the tuning sample.
    """

    X_tune, X_validation, y_tune, y_validation = train_test_split(
        X_sample,
        y_sample,
        test_size=VALIDATION_SIZE,
        stratify=y_sample,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Tuning training shape: %s",
        X_tune.shape,
    )

    logger.info(
        "Validation shape: %s",
        X_validation.shape,
    )

    logger.info(
        "Validation target distribution:\n%s",
        y_validation.value_counts(),
    )

    return (
        X_tune,
        X_validation,
        y_tune,
        y_validation,
    )


# ============================================================
# 8. Define hyperparameter search space
# ============================================================

def create_search():
    """
    Create RandomizedSearchCV for the Random Forest model.

    The search space covers the main Random Forest parameters:
        - number of trees
        - tree depth
        - minimum samples per leaf
        - minimum samples required to split
        - number of features considered at each split
    """

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    parameter_distributions = {
        "n_estimators": [
            50,
            75,
            100,
            125,
        ],
        "max_depth": [
            10,
            15,
            20,
            25,
            None,
        ],
        "min_samples_split": [
            2,
            5,
            10,
            20,
        ],
        "min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
        ],
        "max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
        "bootstrap": [
            True,
        ],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=2,
        return_train_score=True,
        refit=True,
        pre_dispatch=N_JOBS,
    )

    return search


# ============================================================
# 9. Evaluate model
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):
    """Calculate classification performance metrics."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# 10. Save search results
# ============================================================

def save_search_results(
    search,
):
    """Save all RandomizedSearchCV results."""

    results_df = pd.DataFrame(
        search.cv_results_
    )

    results_file = (
        TUNING_RESULTS_DIR
        / "random_search_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    logger.info(
        "Randomized search results saved to: %s",
        results_file,
    )


# ============================================================
# 11. Main tuning pipeline
# ============================================================

def main():
    """Run the complete Phase 2 Task 2 tuning pipeline."""

    logger.info(
        "Starting Phase 2 - Task 2 hyperparameter tuning."
    )

    pipeline_start = time.time()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_df,
        test_df,
    ) = load_data()

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Create tuning sample
    # --------------------------------------------------------

    (
        X_sample,
        y_sample,
    ) = create_tuning_sample(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Create validation split
    # --------------------------------------------------------

    (
        X_tune,
        X_validation,
        y_tune,
        y_validation,
    ) = create_validation_split(
        X_sample,
        y_sample,
    )

    # --------------------------------------------------------
    # Randomized hyperparameter search
    # --------------------------------------------------------

    logger.info(
        "Starting RandomizedSearchCV."
    )

    logger.info(
        "Search iterations: %d",
        N_ITER_SEARCH,
    )

    logger.info(
        "Cross-validation folds: %d",
        CV_FOLDS,
    )

    search = create_search()

    search_start = time.time()

    search.fit(
        X_tune,
        y_tune,
    )

    search_time = (
        time.time()
        - search_start
    )

    logger.info(
        "RandomizedSearchCV completed in %.2f seconds.",
        search_time,
    )

    logger.info(
        "Best cross-validation ROC-AUC: %.6f",
        search.best_score_,
    )

    logger.info(
        "Best hyperparameters: %s",
        search.best_params_,
    )

    # --------------------------------------------------------
    # Save search results
    # --------------------------------------------------------

    save_search_results(
        search
    )

    # --------------------------------------------------------
    # Evaluate best search model on validation set
    # --------------------------------------------------------

    validation_metrics = evaluate_model(
        search.best_estimator_,
        X_validation,
        y_validation,
    )

    logger.info(
        "Validation Accuracy: %.4f",
        validation_metrics["accuracy"],
    )

    logger.info(
        "Validation Precision: %.4f",
        validation_metrics["precision"],
    )

    logger.info(
        "Validation Recall: %.4f",
        validation_metrics["recall"],
    )

    logger.info(
        "Validation F1 Score: %.4f",
        validation_metrics["f1_score"],
    )

    logger.info(
        "Validation ROC-AUC: %.4f",
        validation_metrics["roc_auc"],
    )

    # --------------------------------------------------------
    # Retrain best model on full training dataset
    # --------------------------------------------------------

    logger.info(
        "Retraining best Random Forest on full training data."
    )

    best_params = search.best_params_

    final_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    final_start = time.time()

    final_model.fit(
        X_train,
        y_train,
    )

    final_training_time = (
        time.time()
        - final_start
    )

    logger.info(
        "Final model training time: %.2f seconds.",
        final_training_time,
    )

    # --------------------------------------------------------
    # Evaluate final model on untouched test data
    # --------------------------------------------------------

    test_metrics = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    logger.info(
        "Final test Accuracy: %.4f",
        test_metrics["accuracy"],
    )

    logger.info(
        "Final test Precision: %.4f",
        test_metrics["precision"],
    )

    logger.info(
        "Final test Recall: %.4f",
        test_metrics["recall"],
    )

    logger.info(
        "Final test F1 Score: %.4f",
        test_metrics["f1_score"],
    )

    logger.info(
        "Final test ROC-AUC: %.4f",
        test_metrics["roc_auc"],
    )

    # --------------------------------------------------------
    # Save final tuned model
    # --------------------------------------------------------

    model_file = (
        MODELS_DIR
        / "tuned_random_forest.joblib"
    )

    joblib.dump(
        final_model,
        model_file,
    )

    logger.info(
        "Saved tuned model: %s",
        model_file,
    )

    # --------------------------------------------------------
    # Save best parameters
    # --------------------------------------------------------

    parameters_file = (
        TUNING_RESULTS_DIR
        / "best_parameters.txt"
    )

    with open(
        parameters_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Phase 2 - Task 2 Best Random Forest Parameters\n"
        )

        file.write(
            "=================================================\n\n"
        )

        for parameter, value in best_params.items():

            file.write(
                f"{parameter}: {value}\n"
            )

        file.write(
            "\nBest cross-validation ROC-AUC: "
            f"{search.best_score_:.6f}\n"
        )

    logger.info(
        "Best parameters saved to: %s",
        parameters_file,
    )

    # --------------------------------------------------------
    # Save validation and test metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "evaluation_stage": "validation",
                **validation_metrics,
            },
            {
                "evaluation_stage": "final_test",
                **test_metrics,
            },
        ]
    )

    metrics_file = (
        TUNING_RESULTS_DIR
        / "tuning_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    logger.info(
        "Tuning metrics saved to: %s",
        metrics_file,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_time = (
        time.time()
        - pipeline_start
    )

    logger.info("=" * 70)
    logger.info(
        "PHASE 2 - TASK 2 COMPLETED"
    )

    logger.info(
        "Best parameters: %s",
        best_params,
    )

    logger.info(
        "Final test metrics:\n%s",
        metrics_df.to_string(index=False),
    )

    logger.info(
        "Total pipeline time: %.2f seconds",
        total_time,
    )

    logger.info(
        "Feature count used: %d",
        len(feature_columns),
    )


# ============================================================
# 12. Entry point
# ============================================================

if __name__ == "__main__":
    main()