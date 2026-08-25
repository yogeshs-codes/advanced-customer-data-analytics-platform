# Task 5 - Inference Optimization

## Overview

This task implements and evaluates an optimized inference engine for the customer demand prediction model. The objective is to provide a lightweight inference workflow and measure prediction latency under repeated execution.

The implementation uses the trained `GradientBoostingClassifier` model and preserves the exact feature order expected by the model.

## Implementation

The optimized inference engine is implemented in:

`src/inference_optimizer.py`

The implementation performs the following steps:

1. Loads the trained Gradient Boosting model using Joblib.
2. Reads the model's `feature_names_in_` attribute to preserve the expected feature order.
3. Creates prediction input as a Pandas DataFrame using the model's exact feature names.
4. Generates both class predictions and future-purchase probabilities.
5. Measures individual inference latency.
6. Runs a 100-iteration benchmark.
7. Calculates average, median, P95, minimum, and maximum inference latency.
8. Records model loading time separately from inference latency.

Using a Pandas DataFrame with the correct feature names also prevents feature-name validation warnings during inference.

## Model Used

The model used for the inference benchmark is:

`GradientBoostingClassifier`

The model contains 19 input features and supports binary classification with classes:

- `0` - No Future Purchase
- `1` - Future Purchase

## Test Prediction

A sample customer-product record was passed through the optimized inference engine.

Result:

- Prediction: `0`
- Future purchase probability: `0.487409`

The result is consistent with the earlier direct model inference test.

## Benchmark Configuration

The inference benchmark performs:

- 100 inference iterations
- Prediction and probability calculation for each iteration
- Latency measurement using high-resolution performance timing

The model loading time is measured separately so that one-time model initialization overhead does not affect the repeated inference benchmark.

## Benchmark Results

| Metric | Result |
|---|---:|
| Iterations | 100 |
| Average latency | 1.7304 ms |
| Median latency | 1.5792 ms |
| P95 latency | 2.6743 ms |
| Minimum latency | 1.3193 ms |
| Maximum latency | 4.1724 ms |
| Model load time | 1685.8317 ms |

## Performance Analysis

The benchmark demonstrates low repeated inference latency.

The average inference latency was approximately `1.73 ms`, while the median latency was approximately `1.58 ms`. The P95 latency was `2.67 ms`, showing that most inference requests completed within a few milliseconds.

The maximum measured inference latency was `4.17 ms`.

The model loading time was approximately `1.69 seconds`. This is a one-time startup cost and is separated from the repeated inference measurements.

Compared with the earlier benchmark, the optimized implementation reduced average inference latency from approximately `2.13 ms` to `1.73 ms`.

## Optimization Approach

The optimization focuses on efficient inference execution rather than retraining the model.

Key improvements include:

- Loading the model once at application startup.
- Reusing the already-loaded model for repeated predictions.
- Preserving the model's exact feature order.
- Using a Pandas DataFrame with valid feature names.
- Avoiding repeated model loading inside the inference loop.
- Separating model initialization time from prediction latency.
- Benchmarking repeated inference to obtain stable performance measurements.

## Validation

The implementation was validated using Python compilation and direct execution.

Syntax validation:

```text
python -m py_compile src/inference_optimizer.py
```

The script executed successfully and completed all 100 benchmark iterations without errors or feature-name warnings.

## Conclusion

Task 5 successfully implements an optimized inference engine for the customer demand prediction model.

The final benchmark achieved an average inference latency of `1.7304 ms`, median latency of `1.5792 ms`, and P95 latency of `2.6743 ms` across 100 iterations.

The implementation provides a clean and reusable inference workflow suitable for integration into the model serving layer and future performance monitoring.
