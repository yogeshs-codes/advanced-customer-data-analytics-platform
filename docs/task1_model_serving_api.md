# Task 1 - Model Serving API

## 1. Overview

This task implements a production-oriented model serving API for the final customer-product purchase prediction model developed in the Customer Demand Analysis project.

The API is implemented using **FastAPI** and serves predictions from the trained **Gradient Boosting Classifier** model.

The application provides:

- A health-check endpoint to verify API and model readiness.
- A prediction endpoint for customer-product purchase prediction.
- Request validation using Pydantic.
- Consistent JSON responses.
- Docker-based containerization.
- Interactive API documentation through Swagger UI.

---

## 2. Technology Stack

The serving application uses:

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Docker

---

## 3. Model Used

The API loads the final selected model from:

```text
output/models/final_selected_model.joblib
```

The model version exposed by the API is:

```text
gradient_boosting_v1
```

The API verifies that the trained model contains exactly 19 expected features before serving predictions.

---

## 4. Project Files

The model-serving implementation consists of the following files:

```text
customer_demand_analysis/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── src/
│   └── serving_api.py
├── output/
│   └── models/
│       └── final_selected_model.joblib
└── docs/
    └── task1_model_serving_api.md
```

---

## 5. API Endpoints

### 5.1 GET /health

**Purpose**

The health endpoint verifies that the API is running and that the trained model has been loaded successfully.

**Request**

```http
GET /health
```

No request parameters or request body are required.

**Example Response**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "GradientBoostingClassifier",
  "model_version": "gradient_boosting_v1",
  "feature_count": 19
}
```

**Successful Status Code**

`200 OK`

---

## 6. POST /predict

**Purpose**

The `/predict` endpoint generates a future-purchase prediction for one customer-product record.

**Request**

```http
POST /predict
Content-Type: application/json
```

The endpoint requires a JSON request body containing all 19 model features.

### Required Input Fields

| Field | Type | Validation |
| :--- | :--- | :--- |
| `user_product_purchase_count` | float | >= 0 |
| `user_product_reorder_count` | float | >= 0 |
| `user_product_last_order_number` | float | >= 0 |
| `user_product_reorder_rate` | float | 0 to 1 |
| `user_product_avg_cart_position` | float | >= 0 |
| `user_product_recency_orders` | float | >= 0 |
| `department_id` | float | >= 0 |
| `user_department_purchase_count` | float | >= 0 |
| `user_department_purchase_share` | float | 0 to 1 |
| `aisle_id` | float | >= 0 |
| `user_aisle_purchase_count` | float | >= 0 |
| `user_aisle_purchase_share` | float | 0 to 1 |
| `user_total_orders` | float | >= 0 |
| `user_avg_days_between_orders` | float | >= 0 |
| `user_avg_order_hour` | float | 0 to 23 |
| `user_avg_order_dow` | float | 0 to 6 |
| `product_total_purchases` | float | >= 0 |
| `product_unique_users` | float | >= 0 |
| `product_reorder_rate` | float | 0 to 1 |

### Example Request

```json
{
  "user_product_purchase_count": 0,
  "user_product_reorder_count": 0,
  "user_product_last_order_number": 0,
  "user_product_reorder_rate": 1,
  "user_product_avg_cart_position": 0,
  "user_product_recency_orders": 0,
  "department_id": 0,
  "user_department_purchase_count": 0,
  "user_department_purchase_share": 1,
  "aisle_id": 0,
  "user_aisle_purchase_count": 0,
  "user_aisle_purchase_share": 1,
  "user_total_orders": 0,
  "user_avg_days_between_orders": 0,
  "user_avg_order_hour": 23,
  "user_avg_order_dow": 6,
  "product_total_purchases": 0,
  "product_unique_users": 0,
  "product_reorder_rate": 1
}
```

### Example Successful Response

```json
{
  "prediction": 0,
  "prediction_label": "No Future Purchase",
  "probability_future_purchase": 0.4677504844854496,
  "model": "gradient_boosting_v1",
  "feature_count": 19
}
```

### Response Fields

| Field | Description |
| :--- | :--- |
| `prediction` | Binary model prediction: 0 or 1 |
| `prediction_label` | Human-readable prediction label |
| `probability_future_purchase` | Probability of future purchase |
| `model` | Model version used for prediction |
| `feature_count` | Number of model features used |

The prediction labels are:

- `0` -> No Future Purchase
- `1` -> Future Purchase

---

## 7. Input Validation

Request validation is implemented using Pydantic. The API validates:

- Required fields.
- Numeric data types.
- Non-negative values where required.
- Reorder and purchase rates between 0 and 1.
- Order hour between 0 and 23.
- Day-of-week value between 0 and 6.

Invalid requests are rejected before model prediction is performed.

For example, an invalid request produces:

`422 Unprocessable Entity`

This confirms that the API validation layer is working correctly.

---

## 8. Model Inference

The API uses the exact feature order expected by the trained model. The validated request is converted into a Pandas DataFrame and passed to the trained model.

The API generates:

- A binary class prediction.
- The probability of the future-purchase class.
- A human-readable prediction label.

The model's feature names are obtained from the trained model using:

```python
model.feature_names_in_
```

This helps ensure that inference uses the same feature ordering as model training.

---

## 9. Error Handling

The API provides validation and inference error handling.

### Validation Errors

Invalid request data is rejected by Pydantic and FastAPI with:

`422 Unprocessable Entity`

### Prediction Errors

Unexpected errors during model inference are converted into an HTTP error response:

`500 Internal Server Error`

---

## 10. Docker Containerization

The API is containerized using the following Docker configuration:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY output/models ./output/models

EXPOSE 8000

CMD ["uvicorn", "src.serving_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Docker Image

From the project root:

```bash
docker build -t customer-demand-api .
```

### Run Container

```bash
docker run --name customer-demand-container -p 8000:8000 customer-demand-api
```

The API is then available at:

`http://127.0.0.1:8000`

---

## 11. Interactive API Documentation

FastAPI automatically provides interactive Swagger documentation.

Open:

`http://127.0.0.1:8000/docs`

The Swagger interface provides access to:

- `GET /health`
- `POST /predict`

It also displays request schemas, validation requirements, response schemas, and allows endpoints to be tested directly from the browser.

---

## 12. API Testing

The Dockerized API was tested locally using the Swagger UI.

### Health Check

The following request was successfully tested:

```http
GET /health
```

**Result:** `200 OK`

The response confirmed:

- `status: healthy`
- `model_loaded: true`
- `model: GradientBoostingClassifier`
- `model_version: gradient_boosting_v1`
- `feature_count: 19`

### Valid Prediction

A valid request was sent to:

```http
POST /predict
```

**Result:** `200 OK`

The API returned a prediction, prediction label, future-purchase probability, model version, and feature count.

### Invalid Prediction

An invalid request was also tested against:

```http
POST /predict
```

**Result:** `422 Unprocessable Entity`

This confirms that request validation is enforced by the API.

---

## 13. Docker Runtime Verification

The API container successfully started Uvicorn with:

```text
Uvicorn running on http://0.0.0.0:8000
```

The Docker logs showed successful requests including:

```text
GET /docs HTTP/1.1 200 OK
GET /openapi.json HTTP/1.1 200 OK
GET /health HTTP/1.1 200 OK
POST /predict HTTP/1.1 200 OK
POST /predict HTTP/1.1 422 Unprocessable Entity
```

These results demonstrate that the API, model loading, prediction endpoint, documentation endpoint, and request validation are functioning correctly inside the Docker container.

---

## 14. Task Deliverables

The implementation provides the required deliverables:

- **FastAPI Application:** Implemented in `src/serving_api.py`
- **API Documentation:** Provided in `docs/task1_model_serving_api.md`
- **Dockerfile:** Provided at `Dockerfile`
- **Dependencies:** Provided in `requirements.txt`
- **Docker Ignore Configuration:** Provided in `.dockerignore`

---

## 15. Conclusion

The customer demand prediction model has been successfully exposed through a FastAPI serving layer and containerized using Docker.

The API provides validated prediction requests, consistent JSON responses, model health monitoring, interactive Swagger documentation, and successful local Docker-based testing.

The implementation satisfies the core requirements for model serving, request validation, response formatting, and containerized deployment.
