"""
Phase 2 - Task 1: Model Training

This script trains and evaluates multiple classification models
for the customer-product reorder prediction problem.

Input:
    output/training_data/train.csv.gz
    output/training_data/test.csv.gz

Models:
    1. Decision Tree
    2. Random Forest
    3. Gradient Boosting

Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC

Outputs:
    output/model_results/model_metrics.csv
    output/model_results/training.log
    output/models/*.joblib
"""

from pathlib import Path
import logging
import time

import joblib
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "output"
    / "model_results"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_COLUMN = "target"

# Customer and product identifiers are retained in the data
# for traceability but are not used as predictive features.
ID_COLUMNS = [
    "user_id",
    "product_id",
]

RANDOM_STATE = 42


# ============================================================
# 3. Configure logging
# ============================================================

LOG_FILE = RESULTS_DIR / "training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. Load training and test data
# ============================================================

def load_data():
    """Load the prepared training and test datasets."""

    logger.info("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)

    logger.info("Loading test data...")
    test_df = pd.read_csv(TEST_FILE)

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
# 5. Prepare model features
# ============================================================

def prepare_features(train_df, test_df):
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

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

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

    return X_train, y_train, X_test, y_test


# ============================================================
# 6. Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calculate classification performance metrics."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# 7. Train and evaluate one model
# ============================================================

def train_and_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train, evaluate, log, and save one model."""

    logger.info("=" * 70)
    logger.info(
        "Training model: %s",
        model_name,
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    model_file = (
        MODELS_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_file,
    )

    logger.info(
        "Training time: %.2f seconds",
        training_time,
    )

    logger.info(
        "Accuracy: %.4f",
        metrics["accuracy"],
    )

    logger.info(
        "Precision: %.4f",
        metrics["precision"],
    )

    logger.info(
        "Recall: %.4f",
        metrics["recall"],
    )

    logger.info(
        "F1 Score: %.4f",
        metrics["f1_score"],
    )

    logger.info(
        "ROC-AUC: %.4f",
        metrics["roc_auc"],
    )

    logger.info(
        "Saved model: %s",
        model_file,
    )

    metrics["model"] = model_name

    metrics["training_time_seconds"] = round(
        training_time,
        2,
    )

    return metrics


# ============================================================
# 8. Main training pipeline
# ============================================================

def main():
    """Run the complete Phase 2 Task 1 pipeline."""

    logger.info(
        "Starting Phase 2 - Task 1 model training."
    )

    pipeline_start = time.time()

    train_df, test_df = load_data()

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Model definitions
    # --------------------------------------------------------

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    # --------------------------------------------------------
    # Train all models
    # --------------------------------------------------------

    for model_name, model in models.items():

        result = train_and_evaluate(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "training_time_seconds",
        ]
    ]

    results_file = (
        RESULTS_DIR
        / "model_metrics.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    total_time = (
        time.time()
        - pipeline_start
    )

    logger.info("=" * 70)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info(
        "\n%s",
        results_df.to_string(index=False),
    )

    logger.info(
        "Total pipeline time: %.2f seconds",
        total_time,
    )

    logger.info(
        "Results saved to: %s",
        results_file,
    )

    logger.info(
        "Phase 2 - Task 1 completed successfully."
    )


# ============================================================
# 9. Entry point
# ============================================================

if __name__ == "__main__":
    main()