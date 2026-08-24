# Model Monitoring

## Phase 3 - Task 2: Model Monitoring

### 1. Objective

The model monitoring component records prediction activity from the deployed customer demand prediction API.

The monitoring implementation provides:

- Prediction event logging
- Prediction count tracking
- Positive prediction rate tracking
- Prediction latency recording
- Recent prediction history
- Kafka-based prediction event streaming
- Redis-based monitoring statistics

The monitoring system is implemented in:

```text
src/model_monitor.py
```

---

## 2. Monitoring Architecture

The monitoring flow is:

```text
Client
   |
   v
FastAPI /predict
   |
   v
Gradient Boosting Model
   |
   +----------------------+
   |                      |
   v                      v
Redis                  Kafka
   |                      |
   v                      v
Statistics &          Prediction
Recent History        Event Stream
```

The FastAPI prediction endpoint measures prediction latency and sends the resulting monitoring record to the monitoring module.

---

## 3. Prediction Monitoring Record

Each prediction generates a monitoring record containing:

| Field | Description |
|---|---|
| `timestamp` | UTC timestamp when the prediction was generated |
| `prediction` | Predicted class (`0` or `1`) |
| `probability_future_purchase` | Probability of a future purchase |
| `model` | Model version used for prediction |
| `latency_ms` | Prediction processing latency in milliseconds |

Example monitoring record:

```json
{
  "timestamp": "2026-08-24T14:55:22.953156+00:00",
  "prediction": 1,
  "probability_future_purchase": 0.5290054925742836,
  "model": "gradient_boosting_v1",
  "latency_ms": 71.7263999977149
}
```

---

## 4. Redis Monitoring

Redis is used for real-time monitoring statistics and recent prediction storage.

### Redis keys

| Redis Key | Purpose |
|---|---|
| `model:predictions` | Stores recent prediction records |
| `model:total_predictions` | Stores total number of predictions |
| `model:positive_predictions` | Stores number of positive predictions |

The prediction history is maintained as a Redis list.

Only the latest 1,000 prediction records are retained.

### Positive Prediction Rate

The positive prediction rate is calculated as:

```text
positive_prediction_rate =
positive_predictions / total_predictions
```

If no predictions have been recorded, the rate is returned as `0.0`.

---

## 5. Kafka Monitoring

Kafka is used to publish prediction events to the:

```text
model-predictions
```

topic.

Each prediction record is serialized as JSON and published to Kafka.

This allows prediction events to be consumed by monitoring or analytics components independently from the prediction API.

The Kafka broker used during testing was:

```text
localhost:9092
```

---

## 6. Integration with the Serving API

The monitoring module is integrated into:

```text
src/serving_api.py
```

The prediction endpoint measures latency using:

```python
start_time = time.perf_counter()
```

After generating the prediction, latency is calculated as:

```python
latency_ms = (time.perf_counter() - start_time) * 1000
```

The monitoring record is then passed to:

```python
log_prediction(
    prediction=prediction,
    probability_future_purchase=probability_future_purchase,
    model_version=MODEL_VERSION,
    latency_ms=latency_ms,
)
```

This means every successful `/predict` request is recorded in Redis and published to Kafka.

---

## 7. Validation and Testing

The monitoring implementation was tested after starting the Redis service, Kafka broker, and FastAPI serving API.

### Redis connectivity

Redis connectivity was verified successfully:

```text
PING: True
```

The Redis server was running on:

```text
localhost:6379
```

### Model monitoring module

The monitoring module was imported successfully:

```text
Model monitor imported successfully
```

A direct monitoring test generated the following record:

```text
{
  'timestamp': '2026-08-24T14:37:09.219559+00:00',
  'prediction': 1,
  'probability_future_purchase': 0.5290054925742836,
  'model': 'gradient_boosting_v1',
  'latency_ms': 120.5
}
```

The monitoring statistics were:

```text
total_predictions: 1
positive_predictions: 1
positive_prediction_rate: 1.0
```

---

## 8. End-to-End API Monitoring Test

The FastAPI `/health` endpoint returned:

```text
status        : healthy
model_loaded  : True
model         : GradientBoostingClassifier
model_version : gradient_boosting_v1
feature_count : 19
```

A prediction request to `/predict` returned:

```text
prediction                  : 1
prediction_label            : Future Purchase
probability_future_purchase : 0.5290054925742836
model                       : gradient_boosting_v1
feature_count               : 19
```

After this API request, Redis reported:

```text
model:total_predictions = 2
model:positive_predictions = 2
```

The Redis prediction history contained the latest monitoring event:

```json
{
  "timestamp": "2026-08-24T14:55:22.953156+00:00",
  "prediction": 1,
  "probability_future_purchase": 0.5290054925742836,
  "model": "gradient_boosting_v1",
  "latency_ms": 71.7263999977149
}
```

This confirms that the API prediction was successfully recorded by Redis.

---

## 9. Kafka Validation

Kafka consumer testing confirmed that prediction events were published successfully.

The Kafka topic:

```text
model-predictions
```

contained the monitoring events generated by the API.

The latest event included:

```json
{
  "timestamp": "2026-08-24T14:55:22.953156+00:00",
  "prediction": 1,
  "probability_future_purchase": 0.5290054925742836,
  "model": "gradient_boosting_v1",
  "latency_ms": 71.7263999977149
}
```

Therefore, the end-to-end monitoring pipeline was validated as:

```text
API Prediction
      |
      v
Prediction Record
      |
      +---------> Redis
      |             |
      |             +--> Prediction count
      |             +--> Positive count
      |             +--> Recent predictions
      |
      +---------> Kafka
                    |
                    +--> model-predictions topic
```

---

## 10. Dependencies

The following monitoring dependencies were added to `requirements.txt`:

```text
redis
kafka-python
```

The project already uses FastAPI, Uvicorn, pandas, NumPy, scikit-learn, joblib, and Pydantic for the serving and machine learning components.

---

## 11. Result

The model monitoring component successfully provides basic operational monitoring for the deployed prediction service.

The implementation can:

1. Record every successful prediction.
2. Track total prediction volume.
3. Track positive prediction volume.
4. Calculate positive prediction rate.
5. Record prediction latency.
6. Maintain recent prediction history in Redis.
7. Publish prediction events to Kafka.
8. Support independent consumption of prediction events from the `model-predictions` topic.

The end-to-end testing confirms that the FastAPI serving layer, Redis monitoring storage, and Kafka event stream are successfully integrated.
