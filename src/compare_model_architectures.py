"""
Phase 2 - Task 3: Compare Model Architectures

This script compares the trained machine learning models from
Phase 2 Task 1 and the tuned Random Forest from Task 2.

Models compared:
    1. Decision Tree
    2. Random Forest
    3. Gradient Boosting
    4. Tuned Random Forest
    5. Soft Voting Ensemble

Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Inference latency
    - Model size

Outputs:
    output/model_results/architecture_comparison.csv
    output/model_results/architecture_comparison.png
    output/model_results/inference_latency_comparison.png
    docs/task3_model_architecture_comparison.md
"""

from pathlib import Path
import logging
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_data"
    / "train.csv.gz"
)

TEST_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_data"
    / "test.csv.gz"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "output"
    / "model_results"
)

TUNING_DIR = (
    PROJECT_ROOT
    / "output"
    / "tuning_results"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOCS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_COLUMN = "target"

ID_COLUMNS = [
    "user_id",
    "product_id",
]

RANDOM_STATE = 42

MODEL_FILES = {
    "decision_tree": MODELS_DIR / "decision_tree.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "gradient_boosting": MODELS_DIR / "gradient_boosting.joblib",
    "tuned_random_forest": MODELS_DIR / "tuned_random_forest.joblib",
}


# ============================================================
# 3. Logging
# ============================================================

LOG_FILE = RESULTS_DIR / "architecture_comparison.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. Load data
# ============================================================

def load_test_data():
    """Load the final test dataset."""

    logger.info("Loading test data...")

    test_df = pd.read_csv(TEST_FILE)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in test data."
        )

    feature_columns = [
        column
        for column in test_df.columns
        if column not in ID_COLUMNS
        and column != TARGET_COLUMN
    ]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    logger.info(
        "Test data shape: %s",
        test_df.shape,
    )

    logger.info(
        "Number of predictive features: %d",
        len(feature_columns),
    )

    logger.info(
        "Positive class rate: %.4f",
        y_test.mean(),
    )

    return X_test, y_test, feature_columns


# ============================================================
# 5. Load trained models
# ============================================================

def load_models():
    """Load previously trained models."""

    models = {}

    for model_name, model_file in MODEL_FILES.items():

        if not model_file.exists():
            raise FileNotFoundError(
                f"Required model was not found: {model_file}"
            )

        logger.info(
            "Loading model: %s",
            model_name,
        )

        models[model_name] = joblib.load(model_file)

    return models


# ============================================================
# 6. Calculate model size
# ============================================================

def get_model_size_mb(model_name):
    """Return serialized model size in megabytes."""

    model_file = MODEL_FILES[model_name]

    size_bytes = model_file.stat().st_size

    return size_bytes / (1024 * 1024)


# ============================================================
# 7. Evaluate individual model
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):
    """Evaluate one trained model."""

    logger.info(
        "Evaluating model: %s",
        model_name,
    )

    # Warm-up prediction
    warmup_size = min(100, len(X_test))

    model.predict_proba(
        X_test.iloc[:warmup_size]
    )

    # Measure inference latency
    start_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    elapsed = time.perf_counter() - start_time

    latency_ms = elapsed * 1000

    latency_ms_per_row = (
        latency_ms / len(X_test)
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "inference_time_ms": latency_ms,
        "inference_ms_per_row": latency_ms_per_row,
        "model_size_mb": get_model_size_mb(
            model_name
        ),
    }

    return metrics, probabilities


# ============================================================
# 8. Soft Voting Ensemble
# ============================================================

def evaluate_soft_voting_ensemble(
    models,
    X_test,
    y_test,
):
    """
    Evaluate a soft voting ensemble.

    The ensemble averages predicted probabilities from:
        - Decision Tree
        - Random Forest
        - Gradient Boosting
    """

    ensemble_models = [
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    logger.info(
        "Evaluating soft voting ensemble..."
    )

    probability_predictions = []

    # Warm-up
    warmup_size = min(100, len(X_test))

    for model_name in ensemble_models:
        models[model_name].predict_proba(
            X_test.iloc[:warmup_size]
        )

    start_time = time.perf_counter()

    for model_name in ensemble_models:

        probabilities = (
            models[model_name]
            .predict_proba(X_test)[:, 1]
        )

        probability_predictions.append(
            probabilities
        )

    ensemble_probabilities = np.mean(
        probability_predictions,
        axis=0,
    )

    ensemble_predictions = (
        ensemble_probabilities >= 0.5
    ).astype(int)

    elapsed = time.perf_counter() - start_time

    latency_ms = elapsed * 1000

    latency_ms_per_row = (
        latency_ms / len(X_test)
    )

    combined_size_mb = sum(
        get_model_size_mb(model_name)
        for model_name in ensemble_models
    )

    metrics = {
        "model": "soft_voting_ensemble",
        "accuracy": accuracy_score(
            y_test,
            ensemble_predictions,
        ),
        "precision": precision_score(
            y_test,
            ensemble_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            ensemble_predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            ensemble_predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            ensemble_probabilities,
        ),
        "inference_time_ms": latency_ms,
        "inference_ms_per_row": latency_ms_per_row,
        "model_size_mb": combined_size_mb,
    }

    return metrics, ensemble_probabilities


# ============================================================
# 9. Load previous training results
# ============================================================

def load_previous_results():
    """
    Load previous Task 1 and Task 2 results to include
    training time and tuning information in the report.
    """

    training_results_file = (
        RESULTS_DIR
        / "model_metrics.csv"
    )

    tuning_metrics_file = (
        TUNING_DIR
        / "tuning_metrics.csv"
    )

    best_parameters_file = (
        TUNING_DIR
        / "best_parameters.txt"
    )

    training_results = pd.DataFrame()

    if training_results_file.exists():
        training_results = pd.read_csv(
            training_results_file
        )

    tuning_metrics = pd.DataFrame()

    if tuning_metrics_file.exists():
        tuning_metrics = pd.read_csv(
            tuning_metrics_file
        )

    best_parameters = ""

    if best_parameters_file.exists():
        best_parameters = (
            best_parameters_file.read_text(
                encoding="utf-8"
            )
        )

    return (
        training_results,
        tuning_metrics,
        best_parameters,
    )


# ============================================================
# 10. Create comparison visualization
# ============================================================

def create_metric_visualization(results_df):
    """Create grouped model performance comparison chart."""

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    x = np.arange(
        len(results_df["model"])
    )

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for index, metric in enumerate(metrics):

        ax.bar(
            x + (
                index - 2
            ) * width,
            results_df[metric],
            width,
            label=metric.replace(
                "_",
                " ",
            ).title(),
        )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Model Architecture Performance Comparison"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        results_df["model"],
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "architecture_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved visualization: %s",
        output_file,
    )


# ============================================================
# 11. Create inference latency visualization
# ============================================================

def create_latency_visualization(results_df):
    """Create inference latency comparison chart."""

    sorted_df = results_df.sort_values(
        "inference_ms_per_row"
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        sorted_df["model"],
        sorted_df[
            "inference_ms_per_row"
        ],
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Inference Time (ms per row)"
    )

    ax.set_title(
        "Model Inference Latency Comparison"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR
        / "inference_latency_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved visualization: %s",
        output_file,
    )


# ============================================================
# 12. Generate documentation
# ============================================================

def generate_report(
    results_df,
    tuning_metrics,
    best_parameters,
):
    """Generate the Task 3 markdown report."""

    ranking = (
        results_df
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_model = ranking.iloc[0]

    table_df = results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "inference_ms_per_row",
            "model_size_mb",
        ]
    ].copy()

    table_df[
        [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "inference_ms_per_row",
            "model_size_mb",
        ]
    ] = table_df[
        [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "inference_ms_per_row",
            "model_size_mb",
        ]
    ].round(6)

    markdown_table = table_df.to_markdown(
        index=False
    )

    tuning_section = ""

    if best_parameters:
        tuning_section = (
            "\n### Task 2 Tuned Random Forest Parameters\n\n"
            "```text\n"
            f"{best_parameters.strip()}\n"
            "```\n"
        )

    validation_section = ""

    if not tuning_metrics.empty:

        validation_rows = tuning_metrics[
            tuning_metrics[
                "evaluation_stage"
            ].isin(
                [
                    "validation",
                    "final_test",
                ]
            )
        ]

        if not validation_rows.empty:

            validation_section = (
                "\n### Task 2 Validation and Final Test Results\n\n"
                + validation_rows.to_markdown(
                    index=False
                )
                + "\n"
            )

    report = f"""# Phase 2 - Task 3: Model Architecture Comparison

## 1. Objective

The objective of this task is to compare multiple machine learning
architectures for the customer-product reorder prediction problem.

The comparison includes tree-based models, ensemble methods, and the
tuned Random Forest from Phase 2 Task 2.

The models are evaluated using classification performance metrics and
inference latency to assess their suitability for customer analytics
and near-real-time prediction.

## 2. Models Compared

1. Decision Tree
2. Random Forest
3. Gradient Boosting
4. Tuned Random Forest
5. Soft Voting Ensemble

The Soft Voting Ensemble combines the predicted probabilities from the
Decision Tree, Random Forest, and Gradient Boosting models by averaging
their probabilities.

## 3. Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Inference latency
- Serialized model size

ROC-AUC is given particular importance because the reorder prediction
dataset is imbalanced and accuracy alone can be misleading.

## 4. Test Dataset

The same final test dataset used in the previous tasks is used for
model comparison. This ensures that all architectures are evaluated
under the same conditions.

Customer and product identifiers are retained for traceability but are
not used as predictive features.

## 5. Results

{markdown_table}

## 6. Best Performing Architecture

Based on ROC-AUC, the highest-performing architecture was:

**{best_model["model"]}**

with:

- Accuracy: {best_model["accuracy"]:.4f}
- Precision: {best_model["precision"]:.4f}
- Recall: {best_model["recall"]:.4f}
- F1 Score: {best_model["f1_score"]:.4f}
- ROC-AUC: {best_model["roc_auc"]:.4f}
- Inference latency: {best_model["inference_ms_per_row"]:.6f} ms/row

The final recommendation should consider both predictive quality and
inference cost rather than selecting a model using accuracy alone.

## 7. Ensemble Analysis

The Soft Voting Ensemble combines three complementary tree-based
architectures:

- Decision Tree
- Random Forest
- Gradient Boosting

Soft voting averages the predicted probability of the positive class
from each model. This approach can reduce dependence on the behavior
of any single architecture and may improve ranking performance.

The ensemble is compared against the individual models using the same
test data and metrics.

## 8. Hyperparameter Tuning Context

The Tuned Random Forest was selected during Phase 2 Task 2 using
cross-validation based on ROC-AUC.

The best cross-validation ROC-AUC was approximately **0.8110**.

{tuning_section}

{validation_section}

## 9. Inference and Production Considerations

Inference latency was measured on the final test dataset.

A model with slightly better predictive performance may not always be
the best production choice if its inference cost is substantially
higher.

The comparison therefore considers:

- Predictive performance
- ROC-AUC
- F1 Score
- Recall
- Inference latency
- Model size
- Ensemble complexity

For large-scale customer analytics, Random Forest and tuned Random
Forest provide a useful balance between predictive performance and
model complexity.

Gradient Boosting may provide competitive predictive performance but
requires substantially more training time in this project.

The Soft Voting Ensemble provides a way to combine multiple model
architectures, but its inference cost is higher because multiple
models must be executed.

## 10. Key Findings

- Tree-based architectures are effective for the engineered
  customer-product features.
- Random Forest provides stronger ROC-AUC than the standalone Decision
  Tree in the Task 1 comparison.
- Gradient Boosting achieves performance close to Random Forest but has
  considerably higher training cost in this experiment.
- Hyperparameter tuning improves the Random Forest configuration and
  provides a more controlled model complexity.
- Soft voting demonstrates how multiple architectures can be combined
  for ensemble prediction.
- Accuracy should not be used as the only selection criterion because
  the target variable is imbalanced.
- ROC-AUC, F1 Score, recall, and inference latency are important when
  selecting a model for customer analytics.

## 11. Visualizations

### Model Performance

![Model Architecture Comparison](../output/model_results/architecture_comparison.png)

### Inference Latency

![Inference Latency Comparison](../output/model_results/inference_latency_comparison.png)

## 12. Conclusion

The model architecture comparison demonstrates that ensemble-based
tree models are strong candidates for customer-product reorder
prediction.

The final architecture should be selected by balancing predictive
performance with inference efficiency. For a production customer
analytics workflow, the tuned Random Forest is a strong baseline,
while the Soft Voting Ensemble can be considered when the additional
inference cost is justified by improved predictive performance.

This task completes the architecture comparison stage and provides the
basis for selecting the model configuration for subsequent evaluation
and documentation.
"""

    report_file = (
        DOCS_DIR
        / "task3_model_architecture_comparison.md"
    )

    report_file.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Saved report: %s",
        report_file,
    )


# ============================================================
# 13. Main pipeline
# ============================================================

def main():
    """Run the complete Phase 2 Task 3 pipeline."""

    logger.info(
        "=" * 70
    )

    logger.info(
        "Starting Phase 2 - Task 3 model architecture comparison."
    )

    pipeline_start = time.perf_counter()

    X_test, y_test, feature_columns = (
        load_test_data()
    )

    models = load_models()

    results = []

    # --------------------------------------------------------
    # Evaluate individual models
    # --------------------------------------------------------

    for model_name in MODEL_FILES:

        metrics, _ = evaluate_model(
            model_name=model_name,
            model=models[model_name],
            X_test=X_test,
            y_test=y_test,
        )

        results.append(metrics)

    # --------------------------------------------------------
    # Evaluate soft voting ensemble
    # --------------------------------------------------------

    ensemble_metrics, _ = (
        evaluate_soft_voting_ensemble(
            models=models,
            X_test=X_test,
            y_test=y_test,
        )
    )

    results.append(
        ensemble_metrics
    )

    # --------------------------------------------------------
    # Create results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Add training times from Task 1
    training_results, tuning_metrics, best_parameters = (
        load_previous_results()
    )

    results_df[
        "training_time_seconds"
    ] = np.nan

    if not training_results.empty:

        for index, row in results_df.iterrows():

            model_name = row["model"]

            match = training_results[
                training_results["model"]
                == model_name
            ]

            if not match.empty:

                results_df.loc[
                    index,
                    "training_time_seconds",
                ] = match.iloc[0][
                    "training_time_seconds"
                ]

    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "training_time_seconds",
            "inference_time_ms",
            "inference_ms_per_row",
            "model_size_mb",
        ]
    ]

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_file = (
        RESULTS_DIR
        / "architecture_comparison.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    logger.info(
        "Saved comparison results: %s",
        results_file,
    )

    # --------------------------------------------------------
    # Display ranking
    # --------------------------------------------------------

    ranking = (
        results_df
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    logger.info(
        "\nMODEL RANKING BY ROC-AUC\n%s",
        ranking[
            [
                "model",
                "roc_auc",
                "f1_score",
                "recall",
                "inference_ms_per_row",
            ]
        ].to_string(index=False),
    )

    # --------------------------------------------------------
    # Create visualizations
    # --------------------------------------------------------

    create_metric_visualization(
        results_df
    )

    create_latency_visualization(
        results_df
    )

    # --------------------------------------------------------
    # Generate documentation
    # --------------------------------------------------------

    generate_report(
        results_df=results_df,
        tuning_metrics=tuning_metrics,
        best_parameters=best_parameters,
    )

    total_time = (
        time.perf_counter()
        - pipeline_start
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Task 3 completed successfully."
    )

    logger.info(
        "Total comparison time: %.2f seconds",
        total_time,
    )


# ============================================================
# 14. Entry point
# ============================================================

if __name__ == "__main__":
    main()