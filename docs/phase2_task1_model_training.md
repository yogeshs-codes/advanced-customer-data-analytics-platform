# Phase 2 - Task 1: Model Training

## 1. Objective

The objective of Phase 2 Task 1 is to train and evaluate multiple machine learning classification models for the customer-product reorder prediction problem.

The prepared training and test datasets generated during Phase 1 are used as the input to the model training pipeline.

The models trained in this task are:

- Decision Tree
- Random Forest
- Gradient Boosting

The models are evaluated using multiple classification metrics to provide a reliable comparison of predictive performance.

---

## 2. Input Data

The model training pipeline uses the prepared datasets generated during Phase 1:

```text
output/training_data/train.csv.gz
output/training_data/test.csv.gz
```

The datasets contain customer-product level features and a binary target variable named:

```text
target
```

The target represents whether the customer-product combination corresponds to a reorder outcome.

### Dataset Sizes

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 1,326,118 | 22 |
| Test | 2,661,591 | 22 |

The training dataset contains an equal number of positive and negative target observations:

```text
Target 0: 663,059
Target 1: 663,059
```

The test dataset preserves the natural class imbalance:

```text
Target 0: 2,495,826
Target 1:   165,765
```

Because of this class imbalance in the test set, accuracy is not considered sufficient as the only evaluation metric.

---

## 3. Feature Preparation

The target column is separated from the predictive variables before model training.

The following identifier columns are excluded from model training:

- `user_id`
- `product_id`

These identifiers are retained in the source data for traceability but are not treated as predictive features.

A total of **19 predictive features** are used.

### Predictive Features

```text
user_product_purchase_count
user_product_reorder_count
user_product_last_order_number
user_product_reorder_rate
user_product_avg_cart_position
user_product_recency_orders
department_id
user_department_purchase_count
user_department_purchase_share
aisle_id
user_aisle_purchase_count
user_aisle_purchase_share
user_total_orders
user_avg_days_between_orders
user_avg_order_hour
user_avg_order_dow
product_total_purchases
product_unique_users
product_reorder_rate
```

---

## 4. Models Trained

Three supervised classification algorithms were trained.

### 4.1 Decision Tree

The Decision Tree was configured as follows:

```text
max_depth = 12
min_samples_leaf = 10
random_state = 42
```

The Decision Tree provides a fast baseline model and is capable of capturing nonlinear relationships between customer-product features.

---

### 4.2 Random Forest

The Random Forest was configured as follows:

```text
n_estimators = 100
max_depth = 15
min_samples_leaf = 5
random_state = 42
n_jobs = -1
```

Random Forest combines multiple decision trees to provide a more robust ensemble prediction than a single decision tree.

---

### 4.3 Gradient Boosting

The Gradient Boosting model was configured as follows:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 5
random_state = 42
```

Gradient Boosting builds an ensemble of sequential decision trees, with each stage attempting to improve on the errors of the previous stages.

---

## 5. Evaluation Metrics

The following metrics are calculated for every trained model.

### Accuracy

Accuracy measures the proportion of all predictions that are correct.

### Precision

Precision measures the proportion of observations predicted as positive that are actually positive.

### Recall

Recall measures the proportion of actual positive observations that are correctly identified by the model.

### F1 Score

F1 Score provides a balance between precision and recall.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between positive and negative classes across different classification thresholds.

Because the test dataset is highly imbalanced, particular attention is given to precision, recall, F1 score, and ROC-AUC rather than relying on accuracy alone.

---

## 6. Model Performance

The three models were trained on the prepared training dataset and evaluated against the prepared test dataset.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time (sec) |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.7238 | 0.1514 | 0.7456 | 0.2517 | 0.8067 | 52.38 |
| **Random Forest** | **0.7311** | **0.1558** | **0.7507** | **0.2580** | **0.8152** | **399.45** |
| Gradient Boosting | 0.7308 | 0.1556 | 0.7502 | 0.2577 | 0.8152 | 2187.38 |

The values above are taken from the completed model training run and the generated `model_metrics.csv` file.

---

## 7. Model Comparison

### Decision Tree

The Decision Tree completed training in approximately **52.38 seconds**.

Performance:

- Accuracy: **72.38%**
- Precision: **15.14%**
- Recall: **74.56%**
- F1 Score: **25.17%**
- ROC-AUC: **80.67%**

The Decision Tree provides a useful baseline but performs below the two ensemble models on the main evaluation metrics.

---

### Random Forest

Random Forest completed training in approximately **399.45 seconds**.

Performance:

- Accuracy: **73.11%**
- Precision: **15.58%**
- Recall: **75.07%**
- F1 Score: **25.80%**
- ROC-AUC: **81.52%**

Random Forest achieved the highest accuracy, precision, recall, F1 Score, and ROC-AUC among the three evaluated models.

---

### Gradient Boosting

Gradient Boosting completed training in approximately **2,187.38 seconds**.

Performance:

- Accuracy: **73.08%**
- Precision: **15.56%**
- Recall: **75.02%**
- F1 Score: **25.77%**
- ROC-AUC: **81.52%**

Gradient Boosting achieved performance very close to Random Forest.

However, it required substantially more training time:

- Random Forest: **399.45 seconds**
- Gradient Boosting: **2,187.38 seconds**

Therefore, Gradient Boosting did not provide a meaningful performance improvement over Random Forest in this training run.

---

## 8. Selected Model

Based on the results of this task, **Random Forest is selected as the preferred model among the three evaluated models**.

The main reasons are:

1. It achieved the highest accuracy.
2. It achieved the highest precision.
3. It achieved the highest recall.
4. It achieved the highest F1 Score.
5. It achieved an ROC-AUC of approximately **0.8152**.
6. It achieved nearly identical ROC-AUC to Gradient Boosting while requiring substantially less training time.

Random Forest therefore provides the best balance between predictive performance and computational cost among the models evaluated in this task.

The selected model can be further evaluated and optimized in subsequent Phase 2 tasks.

---

## 9. Class Distribution and Metric Interpretation

The training and test datasets have different target distributions.

### Training Dataset

```text
Target 0: 663,059
Target 1: 663,059
```

The training data is balanced between the two target classes.

### Test Dataset

```text
Target 0: 2,495,826
Target 1:   165,765
```

The test dataset contains substantially more negative observations than positive observations.

Because of this imbalance, a model could obtain relatively high accuracy while still performing poorly at identifying positive reorder cases.

For this reason, precision, recall, F1 Score, and ROC-AUC are important for interpreting model performance.

The selected Random Forest achieved:

```text
Recall:  0.7507
F1:      0.2580
ROC-AUC: 0.8152
```

These metrics provide a more informative view of the model's ability to identify positive reorder outcomes.

---

## 10. Model Artifacts

The training pipeline saves the trained models locally under:

```text
output/models/
```

Generated model files:

```text
decision_tree.joblib
random_forest.joblib
gradient_boosting.joblib
```

The model-performance comparison is saved to:

```text
output/model_results/model_metrics.csv
```

The training log is saved to:

```text
output/model_results/training.log
```

The `output/` directory is excluded from version control through `.gitignore`.

This prevents large generated model artifacts from being committed to the Git repository.

In particular, the Random Forest model is approximately **207 MB**, which is larger than GitHub's standard individual file size limit. Therefore, the trained model files remain local rather than being committed to the repository.

---

## 11. Training Pipeline

The complete model training pipeline is implemented in:

```text
src/train_models.py
```

The pipeline performs the following steps:

```text
Load training and test data
        ↓
Separate target variable
        ↓
Remove identifier columns
        ↓
Prepare predictive features
        ↓
Train Decision Tree
        ↓
Evaluate Decision Tree
        ↓
Save Decision Tree model
        ↓
Train Random Forest
        ↓
Evaluate Random Forest
        ↓
Save Random Forest model
        ↓
Train Gradient Boosting
        ↓
Evaluate Gradient Boosting
        ↓
Save Gradient Boosting model
        ↓
Compare model performance
        ↓
Save evaluation metrics
        ↓
Save training log
```

---

## 12. Reproducibility

The training pipeline uses a fixed random seed:

```text
random_state = 42
```

This provides reproducible initialization and training behavior for the configured models.

The training pipeline can be executed from the project root using:

```powershell
python src\train_models.py
```

Required input files:

```text
output/training_data/train.csv.gz
output/training_data/test.csv.gz
```

Generated outputs:

```text
output/models/
output/model_results/model_metrics.csv
output/model_results/training.log
```

---

## 13. Training Run Summary

The complete model training pipeline successfully completed all three model-training stages.

Total pipeline execution time:

```text
2849.05 seconds
```

Approximately:

```text
47 minutes 29 seconds
```

Individual training times were:

| Model | Training Time |
|---|---:|
| Decision Tree | 52.38 seconds |
| Random Forest | 399.45 seconds |
| Gradient Boosting | 2187.38 seconds |

All three models were successfully trained, evaluated, and saved locally.

The model comparison results were successfully written to:

```text
output/model_results/model_metrics.csv
```

The training execution details were successfully written to:

```text
output/model_results/training.log
```

---

## 14. Project Structure After Task 1

The relevant project structure is:

```text
customer_demand_analysis/
│
├── data/
│   ├── aisles.csv
│   ├── departments.csv
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   └── README.md
│
├── docs/
│   ├── feature_engineering.md
│   ├── task3_training_data_preparation.md
│   ├── task5_evaluation_and_documentation.md
│   └── phase2_task1_model_training.md
│
├── notebooks/
│   ├── task2_feature_engineering.ipynb
│   └── task4_baseline_model.ipynb
│
├── output/
│   ├── training_data/
│   │   ├── train.csv.gz
│   │   └── test.csv.gz
│   │
│   ├── models/
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   │   └── gradient_boosting.joblib
│   │
│   └── model_results/
│       ├── model_metrics.csv
│       └── training.log
│
├── src/
│   ├── feature_pipeline.py
│   ├── prepare_training_data.py
│   ├── split_feature_file.py
│   └── train_models.py
│
├── .gitignore
└── README.md
```

Note that the `output/` directory is excluded from version control.

---

## 15. Conclusion

Phase 2 Task 1 successfully established a multi-model training and evaluation pipeline for customer-product reorder prediction.

Three classification algorithms were trained:

- Decision Tree
- Random Forest
- Gradient Boosting

The models were evaluated using accuracy, precision, recall, F1 Score, and ROC-AUC.

Random Forest achieved the strongest overall performance among the evaluated models, with:

```text
Accuracy:  73.11%
Precision: 15.58%
Recall:    75.07%
F1 Score:  25.80%
ROC-AUC:   81.52%
```

It also required substantially less training time than Gradient Boosting while achieving essentially the same ROC-AUC.

Therefore, **Random Forest is selected as the preferred model from this task** and can be carried forward to subsequent model optimization and evaluation activities in Phase 2.