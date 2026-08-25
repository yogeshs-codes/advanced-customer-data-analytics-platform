"""
Optimized inference engine for the customer demand prediction model.

Phase 3 - Task 5: Inference Optimization
"""

from pathlib import Path
import time

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "models"
    / "final_selected_model_backup.joblib"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

load_start = time.perf_counter()
model = joblib.load(MODEL_PATH)
MODEL_LOAD_TIME_MS = (time.perf_counter() - load_start) * 1000

FEATURE_NAMES = list(model.feature_names_in_)


# ---------------------------------------------------------------------------
# Sample inference input
# ---------------------------------------------------------------------------

SAMPLE_VALUES = [
    1,
    1,
    1,
    0.5,
    2,
    1,
    1,
    5,
    0.5,
    1,
    5,
    0.5,
    20,
    3,
    12,
    2,
    50,
    20,
    0.6,
]


def create_input() -> pd.DataFrame:
    """
    Create a DataFrame using the exact feature names and order
    expected by the trained model.
    """
    return pd.DataFrame(
        [SAMPLE_VALUES],
        columns=FEATURE_NAMES,
    )


def run_inference():
    """
    Run one optimized model inference.
    """
    input_data = create_input()

    start = time.perf_counter()

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    latency_ms = (time.perf_counter() - start) * 1000

    probability_future_purchase = float(
        probabilities[list(model.classes_).index(1)]
    )

    return prediction, probability_future_purchase, latency_ms


def benchmark(iterations: int = 100):
    """
    Benchmark repeated inference calls and report latency statistics.
    """
    latencies = []

    input_data = create_input()

    for _ in range(iterations):
        start = time.perf_counter()

        model.predict(input_data)
        model.predict_proba(input_data)

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    series = pd.Series(latencies)

    return {
        "iterations": iterations,
        "average_latency_ms": float(series.mean()),
        "median_latency_ms": float(series.median()),
        "p95_latency_ms": float(series.quantile(0.95)),
        "min_latency_ms": float(series.min()),
        "max_latency_ms": float(series.max()),
    }


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("OPTIMIZED INFERENCE ENGINE TEST")
    print("=" * 70)

    prediction, probability, latency = run_inference()

    print(f"Model: {type(model).__name__}")
    print(f"Prediction: {prediction}")
    print(f"Future purchase probability: {probability:.6f}")
    print(f"Inference latency: {latency:.4f} ms")

    print()
    print("Benchmarking...")

    results = benchmark(iterations=100)

    for key, value in results.items():
        if key == "iterations":
            print(f"{key}: {value:.0f}")
        else:
            print(f"{key}: {value:.4f}")

    print(f"model_load_time_ms: {MODEL_LOAD_TIME_MS:.4f}")
