"""
FastAPI serving layer for the final customer-product purchase prediction model.

Phase 3 - Task 1: Model Serving API
"""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "output" / "models" / "final_selected_model.joblib"

MODEL_VERSION = "gradient_boosting_v1"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Unable to load model from {MODEL_PATH}: {exc}"
    ) from exc


# Get the exact feature order stored by the trained model.
FEATURE_NAMES = list(model.feature_names_in_)

EXPECTED_FEATURE_COUNT = 19


if len(FEATURE_NAMES) != EXPECTED_FEATURE_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_FEATURE_COUNT} model features, "
        f"but found {len(FEATURE_NAMES)}."
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Demand Prediction API",
    description=(
        "Serving API for the final Gradient Boosting model used "
        "for future customer-product purchase prediction."
    ),
    version=MODEL_VERSION,
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Input schema for one customer-product prediction.
    """

    user_product_purchase_count: float = Field(ge=0)
    user_product_reorder_count: float = Field(ge=0)
    user_product_last_order_number: float = Field(ge=0)
    user_product_reorder_rate: float = Field(ge=0, le=1)
    user_product_avg_cart_position: float = Field(ge=0)
    user_product_recency_orders: float = Field(ge=0)

    department_id: float = Field(ge=0)

    user_department_purchase_count: float = Field(ge=0)
    user_department_purchase_share: float = Field(ge=0, le=1)

    aisle_id: float = Field(ge=0)

    user_aisle_purchase_count: float = Field(ge=0)
    user_aisle_purchase_share: float = Field(ge=0, le=1)

    user_total_orders: float = Field(ge=0)
    user_avg_days_between_orders: float = Field(ge=0)
    user_avg_order_hour: float = Field(ge=0, le=23)
    user_avg_order_dow: float = Field(ge=0, le=6)

    product_total_purchases: float = Field(ge=0)
    product_unique_users: float = Field(ge=0)
    product_reorder_rate: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_future_purchase: float
    model: str
    feature_count: int


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, object]:
    """
    Health check endpoint used to verify that the API and model are ready.
    """

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": type(model).__name__,
        "model_version": MODEL_VERSION,
        "feature_count": len(FEATURE_NAMES),
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate a future-purchase prediction for one customer-product record.
    """

    try:
        # Convert validated Pydantic input into a dictionary.
        input_data = request.model_dump()

        # Build DataFrame using the exact feature order expected by the model.
        input_df = pd.DataFrame(
            [[input_data[feature] for feature in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        # Generate class prediction.
        prediction = int(model.predict(input_df)[0])

        # Generate probability for class 1 (Future Purchase).
        probabilities = model.predict_proba(input_df)[0]

        class_to_probability = {
            int(class_value): float(probability)
            for class_value, probability in zip(
                model.classes_,
                probabilities,
            )
        }

        probability_future_purchase = class_to_probability.get(1, 0.0)

        prediction_label = (
            "Future Purchase"
            if prediction == 1
            else "No Future Purchase"
        )

        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            probability_future_purchase=probability_future_purchase,
            model=MODEL_VERSION,
            feature_count=len(FEATURE_NAMES),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc