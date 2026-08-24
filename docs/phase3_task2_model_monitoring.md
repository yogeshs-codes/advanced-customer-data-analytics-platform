# Phase 3 – Task 2: Model Monitoring

## Overview

This document describes the model monitoring implementation added to the Customer Demand Analysis project.

The monitoring layer captures prediction activity from the production serving API and provides operational statistics for monitoring model behavior, prediction distribution, inference latency, and prediction confidence.

## Monitoring Architecture

The monitoring flow is:

```text
Client
   |
   v
FastAPI /predict
   |
   +--> Gradient Boosting Model
   |
   +--> Prediction + Probability + Latency
   |
   +--> Redis
   |      |
   |      +--> Prediction history
   |      +--> Prediction counters
   |      +--> Monitoring statistics
   |
   +--> Kafka
          |
          +--> model-predictions topic
```

A React/Vite dashboard consumes the `/monitoring` endpoint to present the monitoring information visually.

## Implemented Components

### 1. Prediction Logging

The monitoring module records:

- UTC timestamp
- prediction value
- future-purchase probability
- model version
- prediction latency in milliseconds

Each prediction is stored as a monitoring record.

### 2. Redis Monitoring

Redis is used for real-time monitoring storage.

The implementation maintains:

- total prediction count
- positive prediction count
- recent prediction records
- positive prediction rate
- negative prediction rate
- latency statistics
- average prediction probability

Recent predictions are retained in Redis for monitoring purposes.

### 3. Kafka Prediction Events

Each prediction is also published to the Kafka topic:

```text
model-predictions
```

This provides an event-streaming mechanism for prediction monitoring and downstream processing.

### 4. Latency Monitoring

The FastAPI serving layer measures prediction latency using:

```python
time.perf_counter()
```

The monitoring statistics calculate:

- average latency
- minimum latency
- maximum latency

A latency anomaly flag is also exposed by the monitoring layer.

### 5. Prediction Distribution Monitoring

The monitoring layer calculates:

- positive predictions
- negative predictions
- positive prediction rate
- negative prediction rate
- recent prediction count
- recent positive prediction rate

A prediction-distribution drift flag is exposed when the configured monitoring condition is triggered.

### 6. Prediction Confidence Monitoring

The system tracks:

```text
average_probability_future_purchase
```

This provides visibility into the confidence of recent future-purchase predictions.

### 7. Monitoring API

The FastAPI application exposes:

```text
GET /monitoring
```

The endpoint returns the current monitoring statistics, including:

- prediction counts
- prediction rates
- recent prediction count
- latency statistics
- average prediction probability
- anomaly/drift flags
- alerts
- recent predictions

The existing health endpoint remains available:

```text
GET /health
```

and reports model readiness and model metadata.

### 8. Monitoring Dashboard

A React/Vite dashboard was added under:

```text
dashboard/
```

The dashboard provides a visual monitoring interface using React and Recharts.

It is designed to display production monitoring information such as:

- total predictions
- prediction distribution
- prediction rates
- latency metrics
- probability information
- recent prediction activity
- model status

## Validation

The monitoring implementation was validated using the running FastAPI service.

A successful prediction request produced:

```text
prediction: 1
prediction_label: Future Purchase
probability_future_purchase: 0.5290054925742836
model: gradient_boosting_v1
feature_count: 19
```

Redis monitoring confirmed:

```text
total_predictions: 2
positive_predictions: 2
```

The monitoring endpoint returned:

```text
total_predictions: 2
positive_predictions: 2
negative_predictions: 0
positive_prediction_rate: 1.0
negative_prediction_rate: 0.0
recent_prediction_count: 2
recent_positive_prediction_rate: 1.0
average_latency_ms: 96.11319999885745
minimum_latency_ms: 71.7263999977149
maximum_latency_ms: 120.5
average_probability_future_purchase: 0.5290054925742836
latency_anomaly: false
prediction_distribution_drift: false
alerts: []
```

Kafka validation also confirmed prediction events in the:

```text
model-predictions
```

topic.

## Files

The main implementation files are:

```text
src/model_monitor.py
src/model_monitoring.py
src/serving_api.py
docs/model_monitoring.md
docs/phase3_task2_model_monitoring.md
dashboard/
```

## Technologies

- Python
- FastAPI
- Redis
- Kafka
- kafka-python
- React
- Vite
- Recharts
- Pydantic
- scikit-learn
- PowerShell
- WSL/Ubuntu

## Task Completion

Phase 3 – Task 2 implements an end-to-end model monitoring layer covering:

1. Prediction logging
2. Redis-based monitoring storage
3. Kafka prediction events
4. Prediction counters
5. Prediction distribution monitoring
6. Latency monitoring
7. Prediction confidence monitoring
8. Monitoring alerts/flags
9. `/monitoring` API endpoint
10. React monitoring dashboard
11. Documentation and validation
