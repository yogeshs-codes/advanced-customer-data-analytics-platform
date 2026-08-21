# Phase 2 - Task 4: Experiment Tracking with MLflow

## Objective

MLflow was used to track the trained customer-product reorder prediction models from Phase 2. The tracking captures model parameters, evaluation metrics, model artifacts, inference performance, and experiment metadata.

## MLflow Configuration

- Experiment: `Customer Demand Reorder Prediction`
- Tracking backend: SQLite
- Database: `mlflow.db`
- Models tracked: 4
- Predictive features: 19
- Test observations: 2,661,591

## Tracked Models

1. Decision Tree
2. Random Forest
3. Gradient Boosting
4. Tuned Random Forest

## Evaluation Results

| model               |   accuracy |   precision |   recall |   f1_score |   roc_auc |   inference_ms_per_row |   model_size_mb |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|-----------------------:|----------------:|
| decision_tree       |     0.7238 |      0.1514 |   0.7456 |     0.2517 |    0.8067 |               0.000528 |            0.51 |
| random_forest       |     0.7311 |      0.1558 |   0.7507 |     0.258  |    0.8152 |               0.006765 |          197.52 |
| gradient_boosting   |     0.7308 |      0.1556 |   0.7502 |     0.2577 |    0.8152 |               0.003315 |            0.46 |
| tuned_random_forest |     0.7317 |      0.1554 |   0.7461 |     0.2572 |    0.8131 |               0.009349 |           11.15 |

## Best Model

The highest ROC-AUC was achieved by **random_forest**, with a ROC-AUC of **0.8152**.

## MLflow Tracking Details

Each model was logged as a separate MLflow run. The runs contain model parameters, classification metrics, inference latency, model size, model artifacts, and relevant Task 2 tuning artifacts where applicable.

The tuned Random Forest run additionally records that the model originated from the RandomizedSearchCV hyperparameter tuning process completed in Phase 2 Task 2.

## Reproducibility

All tracked models were loaded from the versioned model artifacts produced by the previous Phase 2 tasks. The untouched test dataset was used for consistent evaluation across all tracked models.

## Conclusion

MLflow provides a centralized experiment record for comparing the baseline and tuned models. The tracking information can be inspected locally through the MLflow UI using the generated SQLite tracking database.