# Task 4 – Automated Model Retraining Pipeline

## Overview

This task implements an automated model retraining pipeline for the customer demand analysis project. The pipeline monitors model performance, checks for new customer data through Kafka, triggers retraining when model performance falls below a predefined threshold, validates the retrained model, and promotes the new model only when it passes the performance gate.

## Pipeline Workflow

The retraining workflow consists of the following stages:

1. Load the current production model.
2. Load the latest training and test datasets.
3. Evaluate the current model using accuracy and F1 score.
4. Connect to Kafka and check for new customer records.
5. Use new Kafka data when available.
6. Fall back to the latest local training data when Kafka is unavailable.
7. Compare the current F1 score with the predefined retraining threshold.
8. Trigger model retraining when the F1 score is below the threshold.
9. Evaluate the retrained model on the test dataset.
10. Promote the retrained model only when it passes the performance gate.
11. Save a JSON retraining report containing the evaluation results.

## Kafka Integration

The pipeline uses Kafka to support streaming customer data ingestion. The file `src/retraining_kafka_producer.py` provides a Kafka producer for sending customer records, while `src/retraining_pipeline.py` contains the Kafka consumer logic used by the retraining workflow.

The configured Kafka endpoint is:

`localhost:9092`

If Kafka is unavailable, the pipeline handles the connection error gracefully and continues using the latest local training data. This allows the retraining process to complete without failing because of temporary Kafka unavailability.

## Validation Gate

A performance threshold is used to determine whether retraining is required.

The configured F1 threshold is:

`0.5000`

During testing, the current model produced:

- Accuracy: `0.7333`
- F1 score: `0.2576`

Since the F1 score was below the threshold of `0.5000`, the pipeline automatically triggered retraining.

## Retraining Results

The retrained model achieved:

- Accuracy: `0.7379`
- F1 score: `0.7389`

The retrained F1 score was significantly higher than the previous F1 score and exceeded the required threshold. Therefore, the new model passed the performance gate and was promoted as the current model.

The retraining report was saved to:

`output/model_results/retraining_report.json`

The promoted model is stored under:

`output/models/`

## Docker Implementation

The retraining pipeline is containerized using `Dockerfile.retraining`.

The Docker image was successfully built using:

```powershell
docker build -f Dockerfile.retraining -t customer-demand-retraining:latest .
```

The resulting Docker image was verified with:

```powershell
docker images customer-demand-retraining
```

The image command was also verified as:

```text
python src/retraining_pipeline.py
```

The container was executed using:

```powershell
docker run --name customer-demand-retraining customer-demand-retraining:latest
```

The container completed successfully with:

```text
ExitCode=0
```

## Docker Build Optimization

The `.dockerignore` file was configured to exclude unnecessary project files from the Docker build context. This reduced the build context substantially and avoided copying large datasets and development environments unnecessarily.

The Docker image includes the required Python dependencies, source code, training data, and model files needed by the retraining pipeline.

## Error Handling

The pipeline includes handling for Kafka connection failures. During testing, Kafka was not running on `localhost:9092`, resulting in a connection refusal. Instead of terminating, the pipeline logged the issue and used the latest local training data as a fallback.

This allowed the retraining workflow to continue and successfully complete model retraining.

## Verification

The complete Dockerized retraining workflow was successfully tested.

The final test demonstrated:

- Current model evaluation
- Kafka availability check
- Automatic detection of low model performance
- Automatic retraining
- Retrained model evaluation
- Performance gate validation
- Model promotion
- Retraining report generation
- Successful Docker container completion

### Final Performance Comparison

| Metric | Current Model | Retrained Model |
|---|---:|---:|
| Accuracy | 0.7333 | 0.7379 |
| F1 Score | 0.2576 | 0.7389 |

The retrained model therefore passed the validation gate and was promoted successfully.

## Files Implemented

- `src/retraining_pipeline.py` – automated retraining workflow
- `src/retraining_kafka_producer.py` – Kafka data producer
- `Dockerfile.retraining` – Docker configuration for the retraining pipeline
- `.dockerignore` – Docker build context optimization
- `output/model_results/retraining_report.json` – retraining evaluation report

## Conclusion

Task 4 successfully implements an automated, validated, and containerized model retraining pipeline. The system can detect model performance degradation, trigger retraining automatically, validate the new model, and promote it only when it meets the required performance threshold. Dockerization makes the pipeline reproducible and easier to deploy.
