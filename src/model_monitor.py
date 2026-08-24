"""
Model monitoring utilities for prediction logging.

Phase 3 - Task 2: Model Monitoring

Tracks:
- Prediction distribution
- Prediction probabilities
- Prediction latency
- Basic anomaly/drift indicators
- Recent prediction history

Redis is used for fast metric storage and Kafka is used
for streaming prediction events.
"""

import json
from datetime import datetime, timezone
from statistics import mean

import redis
from kafka import KafkaProducer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST = "localhost"
REDIS_PORT = 6379

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "model-predictions"

REDIS_PREDICTION_KEY = "model:predictions"
REDIS_TOTAL_PREDICTIONS_KEY = "model:total_predictions"
REDIS_POSITIVE_PREDICTIONS_KEY = "model:positive_predictions"
REDIS_NEGATIVE_PREDICTIONS_KEY = "model:negative_predictions"

# Monitoring thresholds
LATENCY_WARNING_THRESHOLD_MS = 500.0
POSITIVE_RATE_DRIFT_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


# ---------------------------------------------------------------------------
# Kafka producer
# ---------------------------------------------------------------------------

kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------

def log_prediction(
    prediction: int,
    probability_future_purchase: float,
    model_version: str,
    latency_ms: float,
) -> dict:
    """
    Log a model prediction to Redis and Kafka.
    """

    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": timestamp,
        "prediction": int(prediction),
        "probability_future_purchase": float(
            probability_future_purchase
        ),
        "model": model_version,
        "latency_ms": float(latency_ms),
    }

    # Store recent predictions in Redis.
    redis_client.lpush(
        REDIS_PREDICTION_KEY,
        json.dumps(record),
    )

    # Keep the latest 1000 records.
    redis_client.ltrim(
        REDIS_PREDICTION_KEY,
        0,
        999,
    )

    # Update prediction counters.
    redis_client.incr(REDIS_TOTAL_PREDICTIONS_KEY)

    if prediction == 1:
        redis_client.incr(REDIS_POSITIVE_PREDICTIONS_KEY)
    else:
        redis_client.incr(REDIS_NEGATIVE_PREDICTIONS_KEY)

    # Publish prediction event to Kafka.
    kafka_producer.send(
        KAFKA_TOPIC,
        record,
    )

    kafka_producer.flush()

    return record


# ---------------------------------------------------------------------------
# Retrieve recent predictions
# ---------------------------------------------------------------------------

def get_recent_predictions(limit: int = 100) -> list:
    """
    Retrieve recent prediction records from Redis.
    """

    limit = max(1, min(limit, 1000))

    records = redis_client.lrange(
        REDIS_PREDICTION_KEY,
        0,
        limit - 1,
    )

    return [
        json.loads(record)
        for record in records
    ]


# ---------------------------------------------------------------------------
# Monitoring statistics
# ---------------------------------------------------------------------------

def get_monitoring_stats() -> dict:
    """
    Return monitoring statistics calculated from Redis.
    """

    total_predictions = int(
        redis_client.get(
            REDIS_TOTAL_PREDICTIONS_KEY
        ) or 0
    )

    positive_predictions = int(
        redis_client.get(
            REDIS_POSITIVE_PREDICTIONS_KEY
        ) or 0
    )

    negative_predictions = int(
        redis_client.get(
            REDIS_NEGATIVE_PREDICTIONS_KEY
        ) or 0
    )

    recent_predictions = get_recent_predictions(100)

    latencies = [
        float(record["latency_ms"])
        for record in recent_predictions
        if "latency_ms" in record
    ]

    probabilities = [
        float(record["probability_future_purchase"])
        for record in recent_predictions
        if "probability_future_purchase" in record
    ]

    recent_positive_predictions = sum(
        1
        for record in recent_predictions
        if int(record.get("prediction", 0)) == 1
    )

    recent_count = len(recent_predictions)

    positive_rate = (
        positive_predictions / total_predictions
        if total_predictions > 0
        else 0.0
    )

    recent_positive_rate = (
        recent_positive_predictions / recent_count
        if recent_count > 0
        else 0.0
    )

    average_latency = (
        mean(latencies)
        if latencies
        else 0.0
    )

    minimum_latency = (
        min(latencies)
        if latencies
        else 0.0
    )

    maximum_latency = (
        max(latencies)
        if latencies
        else 0.0
    )

    average_probability = (
        mean(probabilities)
        if probabilities
        else 0.0
    )

    # Basic latency anomaly detection.
    latency_anomaly = (
        average_latency > LATENCY_WARNING_THRESHOLD_MS
    )

    # Basic prediction-distribution drift detection.
    distribution_drift = (
        abs(recent_positive_rate - positive_rate)
        > POSITIVE_RATE_DRIFT_THRESHOLD
        if total_predictions > 0 and recent_count > 0
        else False
    )

    alerts = []

    if latency_anomaly:
        alerts.append(
            "Average prediction latency exceeds "
            f"{LATENCY_WARNING_THRESHOLD_MS:.0f} ms."
        )

    if distribution_drift:
        alerts.append(
            "Recent positive prediction rate differs "
            "significantly from the overall rate."
        )

    return {
        "total_predictions": total_predictions,
        "positive_predictions": positive_predictions,
        "negative_predictions": negative_predictions,
        "positive_prediction_rate": positive_rate,
        "negative_prediction_rate": (
            negative_predictions / total_predictions
            if total_predictions > 0
            else 0.0
        ),
        "recent_prediction_count": recent_count,
        "recent_positive_prediction_rate": recent_positive_rate,
        "average_latency_ms": average_latency,
        "minimum_latency_ms": minimum_latency,
        "maximum_latency_ms": maximum_latency,
        "average_probability_future_purchase": (
            average_probability
        ),
        "latency_anomaly": latency_anomaly,
        "prediction_distribution_drift": distribution_drift,
        "alerts": alerts,
        "recent_predictions": recent_predictions[:10],
    }
