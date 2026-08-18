# Phase 2 - Task 2: Hyperparameter Tuning

## 1. Objective

The objective of Phase 2 - Task 2 is to improve the performance of the classification model by tuning its hyperparameters.

The Random Forest model was selected for hyperparameter tuning because it achieved the strongest overall ROC-AUC and F1 score among the baseline models trained in Phase 2 - Task 1.

The tuning process uses `RandomizedSearchCV` to efficiently explore different combinations of Random Forest hyperparameters.

---

## 2. Input Data

The hyperparameter tuning pipeline uses the prepared training and test datasets generated during Phase 1.

### Training Data

`output/training_data/train.csv.gz`

**Training dataset shape:**
* 1,326,118 rows
* 22 columns

### Test Data

`output/training_data/test.csv.gz`

**Test dataset shape:**
* 2,661,591 rows
* 22 columns

### Target & ID Columns

* **Target variable:** `target`
* **Identifiers (retained for traceability, excluded from training):**
  * `user_id`
  * `product_id`

Therefore, **19 predictive features** are used for model tuning.

---

## 3. Predictive Features

The following 19 features were used:

* `user_product_purchase_count`
* `user_product_reorder_count`
* `user_product_last_order_number`
* `user_product_reorder_rate`
* `user_product_avg_cart_position`
* `user_product_recency_orders`
* `department_id`
* `user_department_purchase_count`
* `user_department_purchase_share`
* `aisle_id`
* `user_aisle_purchase_count`
* `user_aisle_purchase_share`
* `user_total_orders`
* `user_avg_days_between_orders`
* `user_avg_order_hour`
* `user_avg_order_dow`
* `product_total_purchases`
* `product_unique_users`
* `product_reorder_rate`

---

## 4. Model Selected for Tuning

The Random Forest classifier was selected based on the Phase 2 - Task 1 baseline results.

### Baseline Random Forest Metrics

| Metric | Baseline Random Forest |
| :--- | :--- |
| **Accuracy** | 0.7311 |
| **Precision** | 0.1558 |
| **Recall** | 0.7507 |
| **F1 Score** | 0.2580 |
| **ROC-AUC** | 0.8152 |

The Random Forest provided the strongest baseline ROC-AUC and F1 score among the three models evaluated in Task 1. Therefore, Random Forest was selected for further hyperparameter optimization.

---

## 5. Hyperparameter Search Method

`RandomizedSearchCV` from scikit-learn was used for hyperparameter tuning.

Randomized search was selected instead of an exhaustive grid search because the training dataset contains more than 1.3 million rows. Randomized search allows multiple combinations of hyperparameters to be evaluated while keeping the computational cost manageable.

**Tuning Configuration:**
* **Search method:** `RandomizedSearchCV`
* **Number of iterations:** 8
* **Cross-validation folds:** 2
* **Scoring metric:** ROC-AUC
* **Random state:** 42

---

## 6. Tuning Dataset

Because the complete training dataset is large, a representative balanced subset was used during the hyperparameter search.

* **Tuning sample:** 250,000 rows
* **Target distribution:**
  * Class 0: 125,000
  * Class 1: 125,000

The tuning sample was divided into:
* **Tuning training set:** 200,000 rows
* **Validation set:** 50,000 rows

The validation set was kept separate from the tuning training data and was used to evaluate the selected hyperparameter configuration before final retraining.

---

## 7. Hyperparameters Explored

The Random Forest search explored different values for the following parameters:

* `n_estimators`: Controls the number of decision trees in the forest.
  * Explored: `50`, `75`, `100`, `125`
* `max_depth`: Controls the maximum depth of each decision tree.
  * Explored: `10`, `15`, `25`, `None`
* `min_samples_split`: Controls the minimum number of samples required to split an internal node.
  * Explored: `2`, `5`, `10`, `20`
* `min_samples_leaf`: Controls the minimum number of samples required at a leaf node.
  * Explored: `1`, `5`, `10`, `20`
* `max_features`: Controls the number of features considered when looking for the best split.
  * Explored: `log2`, `0.5`
* `bootstrap`: Controls whether bootstrap samples are used when building trees.
  * Explored: `True`

---

## 8. Best Hyperparameters

The best configuration identified by `RandomizedSearchCV` was:

* `n_estimators`: `75`
* `max_depth`: `10`
* `min_samples_split`: `10`
* `min_samples_leaf`: `5`
* `max_features`: `log2`
* `bootstrap`: `True`

**Best Cross-Validation ROC-AUC:** `0.810995`

These parameters were selected based on the highest cross-validation ROC-AUC obtained during the randomized search.

---

## 9. Validation Performance

The best Random Forest configuration was evaluated on the held-out validation dataset.

### Validation Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.7408 |
| **Precision** | 0.7362 |
| **Recall** | 0.7506 |
| **F1 Score** | 0.7433 |
| **ROC-AUC** | 0.8145 |

The validation dataset was balanced between the two target classes. Therefore, the validation metrics should be interpreted separately from the final test metrics, because the final test dataset has an imbalanced class distribution.

---

## 10. Final Model Training

After selecting the best hyperparameters, a new Random Forest model was trained using the complete Phase 1 training dataset.

**Final Model Configuration:**
* `n_estimators`: 75
* `max_depth`: 10
* `min_samples_split`: 10
* `min_samples_leaf`: 5
* `max_features`: log2
* `bootstrap`: True
* `random_state`: 42

The final model was then evaluated on the original test dataset.

---

## 11. Final Test Performance

The tuned Random Forest achieved the following performance on the final test dataset:

| Metric | Final Tuned Random Forest |
| :--- | :--- |
| **Accuracy** | 0.7317 |
| **Precision** | 0.1554 |
| **Recall** | 0.7461 |
| **F1 Score** | 0.2572 |
| **ROC-AUC** | 0.8131 |

The final test ROC-AUC of approximately 0.8131 indicates that the tuned model retains good ranking ability on unseen data.

---

## 12. Baseline vs Tuned Model

| Metric | Baseline Random Forest | Tuned Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 0.7311 | 0.7317 |
| **Precision** | 0.1558 | 0.1554 |
| **Recall** | 0.7507 | 0.7461 |
| **F1 Score** | 0.2580 | 0.2572 |
| **ROC-AUC** | 0.8152 | 0.8131 |

The tuned model produced very similar performance to the baseline Random Forest on the final test dataset. The small differences indicate that the baseline configuration was already strong for this dataset. Hyperparameter tuning did not produce a meaningful improvement in test ROC-AUC or F1 score.

However, the tuning process successfully identified a more constrained Random Forest configuration with fewer trees and a maximum depth of 10, resulting in improved computational efficiency and smaller model size.

---

## 13. Interpretation of Results

The hyperparameter tuning process successfully explored multiple Random Forest configurations using randomized search and cross-validation.

* **Selected configuration:** `n_estimators=75`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features='log2'`, `bootstrap=True`
* **Best cross-validation ROC-AUC:** `0.810995`
* **Final test ROC-AUC:** `0.813112`

Compared with the baseline Random Forest, the tuned model does not provide a significant numerical improvement on the final test dataset. This demonstrates that hyperparameter optimization does not automatically translate to massive performance gains when feature signal is the primary bottleneck. Later phases can use the tuned model due to its reduced complexity.

---

## 14. Output Files

The hyperparameter tuning pipeline generates the following files:

* **Tuned Model:** `output/models/tuned_random_forest.joblib`
  * Contains the final Random Forest model trained on the complete training set with optimal hyperparameters.
* **Best Parameters:** `output/tuning_results/best_parameters.txt`
  * Records the selected hyperparameters and the best cross-validation ROC-AUC.
* **Randomized Search Results:** `output/tuning_results/random_search_results.csv`
  * Detailed trial-by-trial logs and scores from the search iterations.
* **Tuning Metrics:** `output/tuning_results/tuning_metrics.csv`
  * Holds validation and final test performance metrics.
* **Tuning Log:** `output/tuning_results/tuning.log`
  * Records execution logs, dataset information, search progress, and execution time.

---

## 15. Implementation

The complete hyperparameter tuning pipeline is implemented in:

`src/tune_hyperparameters.py`

**Pipeline Execution Steps:**
1. Loads prepared training and test datasets.
2. Separates target variable from predictive features.
3. Removes customer and product identifiers (`user_id`, `product_id`).
4. Creates a balanced tuning sample.
5. Splits the tuning sample into tuning-training and validation subsets.
6. Defines the parameter search space.
7. Executes `RandomizedSearchCV`.
8. Selects the configuration with the highest cross-validation ROC-AUC.
9. Evaluates the selected model on the validation dataset.
10. Retrains the model on the full training set.
11. Evaluates performance on the final test set.
12. Exports tuned artifacts, parameters, metrics, and logs.

---

## 16. Reproducibility

A fixed random seed of `42` was used throughout the pipeline to ensure deterministic sampling, cross-validation splits, and tree construction.

**Execution Environment:**
* **Python Environment:** `.venv`
* **pandas:** `3.0.5`
* **scikit-learn:** `1.9.0`
* **joblib:** `1.5.3`

---

## 17. Execution

Activate the virtual environment and run the tuning script from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python src\tune_hyperparameters.py