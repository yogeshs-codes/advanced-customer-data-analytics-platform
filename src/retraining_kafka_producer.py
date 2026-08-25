"""
Phase 3 - Task 4: Kafka Producer for Retraining Pipeline

Publishes labelled customer-product records to Kafka so that the
automated retraining pipeline can consume newly available data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_data"
    / "train.csv.gz"
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "customer-retraining-data",
)

MAX_RECORDS = int(
    os.getenv("MAX_RECORDS", "100")
)


def create_producer() -> KafkaProducer:
    """Create a Kafka producer."""

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(
            value
        ).encode("utf-8"),
    )


def publish_records(
    max_records: int = MAX_RECORDS,
) -> int:
    """
    Read labelled customer-product records and publish them
    to the retraining Kafka topic.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    print(
        f"Loading training data from {TRAIN_FILE}"
    )

    dataframe = pd.read_csv(
        TRAIN_FILE,
        nrows=max_records,
    )

    if "target" not in dataframe.columns:
        raise ValueError(
            "Training data must contain the 'target' column."
        )

    producer = create_producer()

    published = 0

    try:
        for record in dataframe.to_dict(
            orient="records"
        ):
            producer.send(
                KAFKA_TOPIC,
                record,
            )
            published += 1

        producer.flush()

    finally:
        producer.close()

    print(
        f"Published {published} records "
        f"to Kafka topic '{KAFKA_TOPIC}'."
    )

    return published


if __name__ == "__main__":
    publish_records()