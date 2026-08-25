"""
Phase 3 - Task 4: Automated Model Retraining Pipeline

This module implements an automated retraining workflow for the
customer demand prediction model.

Pipeline:
    1. Consume new customer data from Kafka.
    2. Validate incoming records.
    3. Evaluate the current model on labelled data.
    4. Compare performance against a configurable threshold.
    5. Automatically retrain when performance drops below the threshold.
    6. Save the updated model and evaluation results.

The implementation supports local execution when Kafka is unavailable
by using a simulated batch of incoming data.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

try:
    from kafka import KafkaConsumer
except ImportError:
    KafkaConsumer = None


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

MODELS_DIR = PROJECT_ROOT / "output" / "models"
RESULTS_DIR = PROJECT_ROOT / "output" / "model_results"

CURRENT_MODEL_FILE = MODELS_DIR / "final_selected_model.joblib"
RETRAINED_MODEL_FILE = MODELS_DIR / "retrained_model.joblib"

RETRAINING_LOG_FILE = RESULTS_DIR / "retraining_pipeline.log"
RETRAINING_REPORT_FILE = RESULTS_DIR / "retraining_report.json"


MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_COLUMN = "target"

ID_COLUMNS = [
    "user_id",
    "product_id",
]

RANDOM_STATE = 42

# Retraining is triggered when F1 falls below this threshold.
PERFORMANCE_THRESHOLD = float(
    os.getenv("PERFORMANCE_THRESHOLD", "0.50")
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "customer-retraining-data",
)

KAFKA_GROUP_ID = os.getenv(
    "KAFKA_GROUP_ID",
    "customer-retraining-consumer",
)


# ============================================================
# 3. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            RETRAINING_LOG_FILE,
            mode="a",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. Data preparation
# ============================================================

def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate target and predictive features.

    Identifier columns are removed because they are retained for
    traceability but are not used as model predictors.
    """

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' "
            "was not found."
        )

    data = dataframe.copy()

    y = data[TARGET_COLUMN].astype(int)

    feature_columns = [
        column
        for column in data.columns
        if column not in ID_COLUMNS + [TARGET_COLUMN]
    ]

    X = data[feature_columns]

    # Keep only numeric features for model training.
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError(
            "No numeric predictive features were found."
        )

    return X, y


# ============================================================
# 5. Load existing datasets
# ============================================================

def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the existing training and test datasets."""

    logger.info("Loading training data from %s", TRAIN_FILE)
    train_df = pd.read_csv(TRAIN_FILE)

    logger.info("Loading test data from %s", TEST_FILE)
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
# 6. Current model evaluation
# ============================================================

def evaluate_model(
    model: Any,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate a model using accuracy and F1 score.
    """

    X, y = prepare_features(dataframe)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }


# ============================================================
# 7. Load current model
# ============================================================

def load_current_model() -> Any:
    """Load the currently deployed model."""

    if not CURRENT_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Current model was not found: "
            f"{CURRENT_MODEL_FILE}"
        )

    logger.info(
        "Loading current model: %s",
        CURRENT_MODEL_FILE,
    )

    return joblib.load(CURRENT_MODEL_FILE)


# ============================================================
# 8. Kafka consumer
# ============================================================

def consume_kafka_records(
    max_records: int = 100,
) -> list[dict[str, Any]]:
    """
    Consume incoming customer records from Kafka.

    If Kafka is unavailable, an empty list is returned so the local
    retraining workflow can continue using the latest available
    training data.
    """

    if KafkaConsumer is None:
        logger.warning(
            "kafka-python is unavailable. "
            "Skipping Kafka consumption."
        )
        return []

    logger.info(
        "Connecting to Kafka at %s",
        KAFKA_BOOTSTRAP_SERVERS,
    )

    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=3000,
            value_deserializer=lambda value: json.loads(
                value.decode("utf-8")
            ),
        )

        records = []

        for message in consumer:
            records.append(message.value)

            if len(records) >= max_records:
                break

        consumer.close()

        logger.info(
            "Consumed %d Kafka records.",
            len(records),
        )

        return records

    except Exception as exc:
        logger.warning(
            "Kafka unavailable: %s",
            exc,
        )
        return []


# ============================================================
# 9. Validate incoming data
# ============================================================

def validate_incoming_data(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Validate incoming customer records.

    Records must contain the target field and valid numeric values
    for predictive features.
    """

    if not records:
        logger.info(
            "No new Kafka records available."
        )
        return pd.DataFrame()

    dataframe = pd.DataFrame(records)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            "Incoming Kafka data must contain the "
            f"'{TARGET_COLUMN}' field."
        )

    dataframe = dataframe.dropna(
        subset=[TARGET_COLUMN]
    )

    logger.info(
        "Validated %d incoming records.",
        len(dataframe),
    )

    return dataframe


# ============================================================
# 10. Retrain model
# ============================================================

def retrain_model(
    training_data: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    """
    Train a new Random Forest model using the latest available data.
    """

    logger.info(
        "Starting model retraining..."
    )

    X, y = prepare_features(training_data)

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(validation_accuracy),
        "f1_score": float(validation_f1),
    }

    joblib.dump(
        model,
        RETRAINED_MODEL_FILE,
    )

    logger.info(
        "Retrained model saved to %s",
        RETRAINED_MODEL_FILE,
    )

    logger.info(
        "Retrained model F1 score: %.4f",
        validation_f1,
    )

    return model, metrics


# ============================================================
# 11. Retraining decision
# ============================================================

def should_retrain(
    f1_score_value: float,
) -> bool:
    """
    Determine whether model retraining should be triggered.
    """

    decision = (
        f1_score_value < PERFORMANCE_THRESHOLD
    )

    logger.info(
        "Current F1 score: %.4f",
        f1_score_value,
    )

    logger.info(
        "Retraining threshold: %.4f",
        PERFORMANCE_THRESHOLD,
    )

    logger.info(
        "Retraining required: %s",
        decision,
    )

    return decision


# ============================================================
# 12. Promote retrained model
# ============================================================

def promote_retrained_model() -> None:
    """
    Replace the current model with the retrained model.
    """

    if not RETRAINED_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Retrained model does not exist."
        )

    backup_file = (
        MODELS_DIR
        / "final_selected_model_backup.joblib"
    )

    if CURRENT_MODEL_FILE.exists():
        shutil.copy2(
            CURRENT_MODEL_FILE,
            backup_file,
        )

    shutil.copy2(
        RETRAINED_MODEL_FILE,
        CURRENT_MODEL_FILE,
    )

    logger.info(
        "Retrained model promoted as the current model."
    )


# ============================================================
# 13. Save pipeline report
# ============================================================

def save_report(
    current_metrics: dict[str, float],
    retrained_metrics: dict[str, float] | None,
    retraining_triggered: bool,
) -> None:
    """Save an auditable JSON report."""

    report = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "performance_threshold": PERFORMANCE_THRESHOLD,
        "current_model": str(
            CURRENT_MODEL_FILE
        ),
        "current_metrics": current_metrics,
        "retraining_triggered": retraining_triggered,
        "retrained_metrics": retrained_metrics,
    }

    with open(
        RETRAINING_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    logger.info(
        "Retraining report saved to %s",
        RETRAINING_REPORT_FILE,
    )


# ============================================================
# 14. Main pipeline
# ============================================================

def run_pipeline() -> dict[str, Any]:
    """
    Execute the complete automated retraining workflow.
    """

    logger.info("=" * 70)
    logger.info(
        "AUTOMATED MODEL RETRAINING PIPELINE"
    )
    logger.info("=" * 70)

    # Load current model.
    current_model = load_current_model()

    # Load labelled evaluation data.
    train_df, test_df = load_training_data()

    # Evaluate current production model.
    current_metrics = evaluate_model(
        current_model,
        test_df,
    )

    logger.info(
        "Current model accuracy: %.4f",
        current_metrics["accuracy"],
    )

    logger.info(
        "Current model F1 score: %.4f",
        current_metrics["f1_score"],
    )

    # Consume new customer data from Kafka.
    kafka_records = consume_kafka_records()

    incoming_data = validate_incoming_data(
        kafka_records
    )

    # If Kafka has no records, use the latest labelled
    # training dataset as the local retraining source.
    if incoming_data.empty:
        logger.info(
            "Using latest local training data "
            "as the retraining dataset."
        )
        latest_data = train_df
    else:
        latest_data = pd.concat(
            [train_df, incoming_data],
            ignore_index=True,
        )

    # Check whether retraining is required.
    retraining_triggered = should_retrain(
        current_metrics["f1_score"]
    )

    retrained_metrics = None

    if retraining_triggered:
        _, retrained_metrics = retrain_model(
            latest_data
        )

        # Promote the new model only if it is valid.
        if (
            retrained_metrics["f1_score"]
            >= current_metrics["f1_score"]
        ):
            promote_retrained_model()

            logger.info(
                "New model passed the performance gate "
                "and was promoted."
            )
        else:
            logger.warning(
                "Retrained model did not improve F1. "
                "Current model was retained."
            )

    else:
        logger.info(
            "Performance is above threshold. "
            "Retraining is not required."
        )

    save_report(
        current_metrics=current_metrics,
        retrained_metrics=retrained_metrics,
        retraining_triggered=retraining_triggered,
    )

    return {
        "current_metrics": current_metrics,
        "retraining_triggered": retraining_triggered,
        "retrained_metrics": retrained_metrics,
    }


# ============================================================
# 15. Entry point
# ============================================================

if __name__ == "__main__":
    result = run_pipeline()

    print("\n" + "=" * 70)
    print("RETRAINING PIPELINE COMPLETED")
    print("=" * 70)
    print(
        json.dumps(
            result,
            indent=2,
        )
    )