"""
Phase 3 - Task 2: A/B Testing for Customer Demand Models

This module runs a controlled 50/50 A/B experiment between:

    Model A: Gradient Boosting
    Model B: Tuned Random Forest

Architecture:

    Test data
        |
        v
    Kafka Producer
        |
        v
    customer-demand-ab topic
        |
        v
    A/B Consumer
       / \
      /   \
     v     v
   Model A  Model B
      \     /
       \   /
        v v
       Redis
        |
        v
   A/B Metrics

The experiment uses the labeled test dataset so that predictions
can be evaluated against the actual target.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import redis
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
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

MODEL_A_PATH = MODELS_DIR / "gradient_boosting.joblib"
MODEL_B_PATH = MODELS_DIR / "tuned_random_forest.joblib"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "customer-demand-ab"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

RANDOM_SEED = 42

FEATURE_NAMES = [
    "user_product_purchase_count",
    "user_product_reorder_count",
    "user_product_last_order_number",
    "user_product_reorder_rate",
    "user_product_avg_cart_position",
    "user_product_recency_orders",
    "department_id",
    "user_department_purchase_count",
    "user_department_purchase_share",
    "aisle_id",
    "user_aisle_purchase_count",
    "user_aisle_purchase_share",
    "user_total_orders",
    "user_avg_days_between_orders",
    "user_avg_order_hour",
    "user_avg_order_dow",
    "product_total_purchases",
    "product_unique_users",
    "product_reorder_rate",
]

TARGET_COLUMN = "target"

DEFAULT_SAMPLE_SIZE = 100


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# Model loading
# ============================================================================


def load_models() -> dict[str, Any]:
    """
    Load both trained models and verify their feature schemas.
    """

    if not MODEL_A_PATH.exists():
        raise FileNotFoundError(
            f"Model A not found: {MODEL_A_PATH}"
        )

    if not MODEL_B_PATH.exists():
        raise FileNotFoundError(
            f"Model B not found: {MODEL_B_PATH}"
        )

    logger.info("Loading Model A: %s", MODEL_A_PATH)
    model_a = joblib.load(MODEL_A_PATH)

    logger.info("Loading Model B: %s", MODEL_B_PATH)
    model_b = joblib.load(MODEL_B_PATH)

    model_a_features = list(
        getattr(model_a, "feature_names_in_", [])
    )

    model_b_features = list(
        getattr(model_b, "feature_names_in_", [])
    )

    if model_a_features != FEATURE_NAMES:
        raise RuntimeError(
            "Model A feature schema does not match the expected "
            "19-feature schema."
        )

    if model_b_features != FEATURE_NAMES:
        raise RuntimeError(
            "Model B feature schema does not match the expected "
            "19-feature schema."
        )

    logger.info(
        "Both models validated successfully with %d features.",
        len(FEATURE_NAMES),
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
    }


# ============================================================================
# Kafka
# ============================================================================


def create_kafka_producer() -> KafkaProducer:
    """
    Create a Kafka producer using JSON serialization.
    """

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(
            value
        ).encode("utf-8"),
        acks="all",
        retries=3,
    )


def create_kafka_consumer(group_id: str) -> KafkaConsumer:
    """
    Create a Kafka consumer for the A/B testing topic.
    """

    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )


# ============================================================================
# Redis
# ============================================================================


def create_redis_client() -> redis.Redis:
    """
    Create and validate the Redis connection.
    """

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )

    client.ping()

    logger.info(
        "Redis connection established at %s:%s.",
        REDIS_HOST,
        REDIS_PORT,
    )

    return client


# ============================================================================
# Experiment identifiers
# ============================================================================


def create_experiment_id() -> str:
    """
    Create a unique experiment identifier.
    """

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    return f"ab_{timestamp}_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Data validation
# ============================================================================


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the test data contains the expected columns.
    """

    required_columns = (
        ["user_id", "product_id"]
        + FEATURE_NAMES
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


# ============================================================================
# Kafka producer
# ============================================================================


def produce_test_records(
    producer: KafkaProducer,
    df: pd.DataFrame,
    experiment_id: str,
) -> int:
    """
    Send labeled test records to Kafka.

    The consumer performs the actual A/B assignment. Keeping assignment
    at the consumer makes the routing decision part of the serving layer.
    """

    records_sent = 0

    for row in df.itertuples(index=False):
        row_data = row._asdict()

        message = {
            "experiment_id": experiment_id,
            "user_id": int(row_data["user_id"]),
            "product_id": int(row_data["product_id"]),
            "features": {
                feature: float(row_data[feature])
                for feature in FEATURE_NAMES
            },
            "target": int(row_data[TARGET_COLUMN]),
            "created_at": time.time(),
        }

        producer.send(
            KAFKA_TOPIC,
            value=message,
        )

        records_sent += 1

    producer.flush()

    logger.info(
        "Produced %d records to Kafka topic '%s'.",
        records_sent,
        KAFKA_TOPIC,
    )

    return records_sent


# ============================================================================
# Prediction
# ============================================================================


def predict_with_model(
    model: Any,
    features: dict[str, float],
) -> tuple[int, float]:
    """
    Generate prediction and positive-class probability.
    """

    input_df = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    class_to_probability = {
        int(class_value): float(probability)
        for class_value, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    probability = class_to_probability.get(
        1,
        0.0,
    )

    return prediction, probability


# ============================================================================
# Redis result storage
# ============================================================================


def store_result(
    redis_client: redis.Redis,
    experiment_id: str,
    result: dict[str, Any],
) -> None:
    """
    Store one prediction result in Redis.

    Each prediction is stored as a Redis hash.
    """

    result_id = result["result_id"]

    key = (
        f"ab:{experiment_id}:result:"
        f"{result_id}"
    )

    redis_client.hset(
        key,
        mapping={
            "experiment_id": experiment_id,
            "result_id": result_id,
            "model": result["model"],
            "user_id": result["user_id"],
            "product_id": result["product_id"],
            "prediction": result["prediction"],
            "probability": result["probability"],
            "target": result["target"],
            "correct": result["correct"],
            "latency_ms": result["latency_ms"],
            "timestamp": result["timestamp"],
        },
    )

    redis_client.sadd(
        f"ab:{experiment_id}:results",
        result_id,
    )


# ============================================================================
# A/B consumer
# ============================================================================


def consume_and_predict(
    consumer: KafkaConsumer,
    redis_client: redis.Redis,
    models: dict[str, Any],
    experiment_id: str,
    expected_records: int,
) -> dict[str, int]:
    """
    Consume Kafka records and randomly assign each request to Model A
    or Model B with a deterministic 50/50 assignment.

    Returns counts for processed records and each model.
    """

    rng = random.Random(RANDOM_SEED)

    processed = 0
    model_a_count = 0
    model_b_count = 0

    while processed < expected_records:

        records = consumer.poll(
            timeout_ms=2000,
            max_records=100,
        )

        if not records:
            logger.warning(
                "Waiting for Kafka records..."
            )
            continue

        for _, messages in records.items():

            for message in messages:

                if processed >= expected_records:
                    break

                payload = message.value

                assignment = (
                    "model_a"
                    if rng.random() < 0.5
                    else "model_b"
                )

                selected_model = models[
                    assignment
                ]

                model_name = (
                    "gradient_boosting"
                    if assignment == "model_a"
                    else "tuned_random_forest"
                )

                start_time = time.perf_counter()

                prediction, probability = (
                    predict_with_model(
                        selected_model,
                        payload["features"],
                    )
                )

                latency_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                target = int(
                    payload["target"]
                )

                correct = int(
                    prediction == target
                )

                result = {
                    "result_id": uuid.uuid4().hex,
                    "experiment_id": experiment_id,
                    "model": model_name,
                    "user_id": payload["user_id"],
                    "product_id": payload["product_id"],
                    "prediction": prediction,
                    "probability": probability,
                    "target": target,
                    "correct": correct,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                }

                store_result(
                    redis_client,
                    experiment_id,
                    result,
                )

                processed += 1

                if assignment == "model_a":
                    model_a_count += 1
                else:
                    model_b_count += 1

                if processed % 25 == 0:
                    logger.info(
                        "Processed %d/%d records | "
                        "Model A: %d | Model B: %d",
                        processed,
                        expected_records,
                        model_a_count,
                        model_b_count,
                    )

    return {
        "processed": processed,
        "model_a": model_a_count,
        "model_b": model_b_count,
    }


# ============================================================================
# Metrics
# ============================================================================


def calculate_metrics(
    redis_client: redis.Redis,
    experiment_id: str,
) -> pd.DataFrame:
    """
    Load experiment results from Redis and calculate model metrics.
    """

    result_ids = redis_client.smembers(
        f"ab:{experiment_id}:results"
    )

    rows: list[dict[str, Any]] = []

    for result_id in result_ids:

        key = (
            f"ab:{experiment_id}:result:"
            f"{result_id}"
        )

        result = redis_client.hgetall(key)

        if result:
            rows.append(result)

    if not rows:
        raise RuntimeError(
            "No A/B test results were found in Redis."
        )

    results_df = pd.DataFrame(rows)

    numeric_columns = [
        "prediction",
        "probability",
        "target",
        "correct",
        "latency_ms",
    ]

    for column in numeric_columns:
        results_df[column] = pd.to_numeric(
            results_df[column]
        )

    metrics = []

    for model_name, group in results_df.groupby(
        "model"
    ):
        tp = (
            (group["prediction"] == 1)
            & (group["target"] == 1)
        ).sum()

        fp = (
            (group["prediction"] == 1)
            & (group["target"] == 0)
        ).sum()

        fn = (
            (group["prediction"] == 0)
            & (group["target"] == 1)
        ).sum()

        tn = (
            (group["prediction"] == 0)
            & (group["target"] == 0)
        ).sum()

        accuracy = (
            (tp + tn)
            / len(group)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.append(
            {
                "model": model_name,
                "samples": len(group),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_probability": group[
                    "probability"
                ].mean(),
                "mean_latency_ms": group[
                    "latency_ms"
                ].mean(),
                "p95_latency_ms": group[
                    "latency_ms"
                ].quantile(0.95),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )

    return pd.DataFrame(metrics)


# ============================================================================
# Save report
# ============================================================================


def save_metrics(
    metrics_df: pd.DataFrame,
    experiment_id: str,
) -> Path:
    """
    Save A/B experiment metrics to CSV.
    """

    output_dir = (
        PROJECT_ROOT
        / "output"
        / "ab_testing"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{experiment_id}_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment(
    sample_size: int,
) -> None:
    """
    Run one A/B experiment.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    experiment_id = create_experiment_id()

    logger.info(
        "Starting A/B experiment: %s",
        experiment_id,
    )

    logger.info(
        "Reading %d records from test data.",
        sample_size,
    )

    df = pd.read_csv(
        TEST_DATA_PATH,
        nrows=sample_size,
    )

    validate_dataframe(df)

    models = load_models()

    producer = create_kafka_producer()

    redis_client = create_redis_client()

    consumer_group = (
        f"ab-testing-{experiment_id}"
    )

    consumer = create_kafka_consumer(
        consumer_group
    )

    try:

        # Produce the evaluation records.
        records_sent = produce_test_records(
            producer,
            df,
            experiment_id,
        )

        # Consume and evaluate the records.
        counts = consume_and_predict(
            consumer,
            redis_client,
            models,
            experiment_id,
            records_sent,
        )

        logger.info(
            "Experiment processing complete: %s",
            counts,
        )

        # Calculate and save metrics.
        metrics_df = calculate_metrics(
            redis_client,
            experiment_id,
        )

        output_path = save_metrics(
            metrics_df,
            experiment_id,
        )

        print()
        print("=" * 80)
        print("A/B TEST RESULTS")
        print("=" * 80)
        print()
        print(metrics_df.to_string(index=False))
        print()
        print(f"Experiment ID: {experiment_id}")
        print(f"Results: {output_path}")
        print()

    finally:

        producer.close()

        consumer.close()

        redis_client.close()


# ============================================================================
# Command-line interface
# ============================================================================


def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a 50/50 Kafka + Redis A/B test "
            "between Gradient Boosting and Tuned "
            "Random Forest."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of labeled test records to evaluate. "
            "Default: 100."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()