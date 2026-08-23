# Task 2 — A/B Testing with Kafka and Redis

## Overview

This document records the A/B testing implementation for the customer demand prediction platform.

The A/B testing pipeline compares two trained classification models using a controlled experiment:

- **Model A:** Gradient Boosting
- **Model B:** Tuned Random Forest

The experiment uses Kafka for event/message delivery and Redis for supporting experiment-time state and connectivity.

## Implementation

The A/B testing implementation is located at:

`src/ab_testing.py`

The pipeline:

1. Loads a sample of customer-product records from the test dataset.
2. Loads both candidate models.
3. Validates that both models use the expected 19 features.
4. Publishes test records to the Kafka topic `customer-demand-ab`.
5. Consumes the records through a Kafka consumer group.
6. Randomly assigns records between Model A and Model B.
7. Generates predictions and probabilities.
8. Uses Redis as part of the experiment infrastructure.
9. Calculates model performance and inference-latency metrics.
10. Saves the experiment results as a CSV file.

## Infrastructure

### Kafka

Kafka was configured locally using Apache Kafka 4.3.1 in WSL2.

Kafka topic:

`customer-demand-ab`

Topic configuration:

- Partitions: 1
- Replication factor: 1

Kafka connectivity was verified successfully on:

`localhost:9092`

### Redis

Redis was installed and started in WSL2.

The Redis server was verified with:

`redis-cli ping`

Result:

`PONG`

Redis connectivity from the Python environment was also verified successfully.

## Experiment Configuration

The executed experiment used:

- Sample size: 100 records
- Model A: Gradient Boosting
- Model B: Tuned Random Forest
- Allocation: 50 records per model
- Kafka topic: `customer-demand-ab`

Experiment ID:

`ab_20260823_153319_c9d178b2`

## Results

The experiment produced the following results:

| Metric | Gradient Boosting | Tuned Random Forest |
|---|---:|---:|
| Samples | 50 | 50 |
| Accuracy | 0.74 | 0.72 |
| Precision | 0.2353 | 0.1765 |
| Recall | 1.00 | 1.00 |
| F1 | 0.3810 | 0.3000 |
| Mean Probability | 0.4006 | 0.3654 |
| Mean Latency (ms) | 2.7474 | 29.3163 |
| P95 Latency (ms) | 4.1941 | 47.1312 |
| True Positives | 4 | 3 |
| False Positives | 13 | 14 |
| False Negatives | 0 | 0 |
| True Negatives | 33 | 33 |

## Result Interpretation

For this 100-record A/B experiment, Gradient Boosting performed better than Tuned Random Forest on the measured metrics.

Gradient Boosting achieved:

- Higher accuracy: **0.74 vs 0.72**
- Higher precision: **0.2353 vs 0.1765**
- Higher F1 score: **0.3810 vs 0.3000**
- Lower mean latency: **2.747 ms vs 29.316 ms**
- Lower P95 latency: **4.194 ms vs 47.131 ms**

Both models achieved 100% recall on this sample.

The latency difference is particularly significant: Gradient Boosting had substantially lower inference latency than Tuned Random Forest in this experiment.

Because the experiment contains only 50 observations per model, these results should be treated as an initial A/B test rather than a definitive production conclusion. A larger sample would provide stronger evidence for model selection.

## Output

The experiment generated:

`output/ab_testing/ab_20260823_153319_c9d178b2_metrics.csv`

The output contains model-level performance, latency, and confusion-matrix metrics.

## Validation

The implementation was successfully executed with:

`python src\ab_testing.py --sample-size 100`

The run successfully:

- Connected to Kafka.
- Connected to Redis.
- Published 100 records.
- Consumed and processed all 100 records.
- Assigned 50 records to each model.
- Generated predictions.
- Calculated A/B test metrics.
- Saved the experiment results.

## Git Version Control

The A/B testing implementation was committed and pushed to the project repository.

Commit:

`ff5d135`

Commit message:

`Add Kafka Redis A/B testing pipeline`

The working tree was clean after the push.

## Conclusion

The completed A/B testing pipeline provides an event-driven method for comparing candidate customer-demand prediction models. Based on the executed 100-record experiment, Gradient Boosting was the stronger candidate, providing better accuracy, precision, F1 score, and substantially lower inference latency while maintaining the same recall as the Tuned Random Forest.
