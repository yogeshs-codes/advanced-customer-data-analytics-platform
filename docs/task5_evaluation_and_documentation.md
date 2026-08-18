# Task 5 — Evaluate & Document

## 1. Objective

The objective of Task 5 is to evaluate the machine learning baseline model using relevant performance metrics and document the complete model training and evaluation methodology.

The Logistic Regression model developed in Task 4 is used as the baseline classifier for predicting whether a customer will make a future purchase of a product.

This baseline establishes a reference point for future model improvements.

---

## 2. Dataset and Data Preparation

The project uses the prepared customer-product datasets generated during the earlier data preparation stages.

The Task 3 process produced separate training and testing datasets.

The training dataset contains:

- **1,326,118 rows**
- **19 selected features**

The testing dataset contains:

- **2,661,591 rows**
- **19 selected features**

The target variable is:

- `0` — No Future Purchase
- `1` — Future Purchase

### Target Distribution

The training dataset was balanced through random undersampling:

| Target | Count | Proportion |
|---|---:|---:|
| No Future Purchase (0) | 663,059 | 50.00% |
| Future Purchase (1) | 663,059 | 50.00% |

The testing dataset retained the original class distribution:

| Target | Count | Proportion |
|---|---:|---:|
| No Future Purchase (0) | 2,495,826 | 93.772% |
| Future Purchase (1) | 165,765 | 6.228% |

Maintaining the original distribution in the test dataset provides a more realistic evaluation of how the baseline model performs on the expected class distribution.

---

## 3. Selected Features

A total of **19 features** were selected for model training.

### Categorical Features

The categorical features were:

- `department_id`
- `aisle_id`

### Numerical Features

The numerical features were:

- `user_product_purchase_count`
- `user_product_reorder_count`
- `user_product_last_order_number`
- `user_product_reorder_rate`
- `user_product_avg_cart_position`
- `user_product_recency_orders`
- `user_department_purchase_count`
- `user_department_purchase_share`
- `user_aisle_purchase_count`
- `user_aisle_purchase_share`
- `user_total_orders`
- `user_avg_days_between_orders`
- `user_avg_order_hour`
- `user_avg_order_dow`
- `product_total_purchases`
- `product_unique_users`
- `product_reorder_rate`

No missing values were present in either the training or testing feature datasets.

---

## 4. Baseline Model

The baseline model used for this task is **Logistic Regression**.

Logistic Regression was selected as a baseline classification model because it provides a simple and interpretable reference point for the future-purchase prediction problem.

The model was trained using the prepared training dataset and then used to generate predictions for the untouched testing dataset.

The implementation is available in:

`notebooks/task4_baseline_model.ipynb`

---

## 5. Preprocessing Methodology

A preprocessing pipeline was created before model training.

### Numerical Features

Numerical features were transformed using:

**StandardScaler**

Standardization was applied so that numerical features were placed on a comparable scale before being provided to the Logistic Regression model.

### Categorical Features

Categorical features were transformed using:

**OneHotEncoder**

This converted the categorical department and aisle identifiers into numerical representations suitable for Logistic Regression.

The preprocessing transformations were integrated into the model pipeline.

This ensured that the same preprocessing approach was consistently applied during training and testing.

---

## 6. Model Training

The Logistic Regression baseline was trained using the Task 3 training dataset containing:

- **1,326,118 observations**
- **19 selected features**
- **Balanced target classes**

The model training completed successfully.

After training, predictions were generated for all:

**2,661,591 testing observations**

The number of predictions matched the number of actual test labels.

---

## 7. Evaluation Methodology

The model was evaluated on the untouched testing dataset.

The following performance metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score

A confusion matrix and classification report were also generated to provide a more detailed view of model performance.

Because the testing dataset is highly imbalanced, accuracy was not considered sufficient by itself.

Particular attention was therefore given to the precision, recall, and F1-score of the positive class:

**Future Purchase**

---

## 8. Performance Metrics

### 8.1 Accuracy

The baseline Logistic Regression model achieved:

**Accuracy: 0.7140 (71.40%)**

Accuracy represents the proportion of all test observations that were classified correctly.

However, the test dataset contains 93.772% negative examples and only 6.228% positive examples. Therefore, accuracy alone does not provide a complete picture of the model's ability to identify future purchases.

---

### 8.2 Precision

For the **Future Purchase** class:

**Precision: 0.15 (15%)**

Precision measures how many of the observations predicted as future purchases were actually future purchases.

The relatively low precision indicates that the model produces a substantial number of false-positive predictions.

---

### 8.3 Recall

For the **Future Purchase** class:

**Recall: 0.75 (75%)**

Recall measures the proportion of actual future purchases that were successfully identified by the model.

The 75% recall indicates that the baseline model successfully identifies a substantial proportion of customers/product combinations that result in a future purchase.

---

### 8.4 F1-score

For the **Future Purchase** class:

**F1-score: 0.2454**

The F1-score combines precision and recall into a single metric using their harmonic mean.

The relatively low F1-score is mainly influenced by the low precision of the positive class.

---

## 9. Confusion Matrix

The Logistic Regression baseline produced the following confusion matrix on the test dataset:

| Actual / Predicted | No Future Purchase | Future Purchase |
|---|---:|---:|
| **No Future Purchase** | 1,776,495 | 719,331 |
| **Future Purchase** | 41,951 | 123,814 |

The four outcomes can be interpreted as follows:

- **True Negatives:** 1,776,495
- **False Positives:** 719,331
- **False Negatives:** 41,951
- **True Positives:** 123,814

The model correctly identified 123,814 future purchases while missing 41,951 actual future purchases.

At the same time, 719,331 observations were incorrectly predicted as future purchases, which explains the relatively low precision.

---

## 10. Classification Report

The complete classification report was:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Future Purchase | 0.98 | 0.71 | 0.82 | 2,495,826 |
| Future Purchase | 0.15 | 0.75 | 0.25 | 165,765 |
| **Accuracy** | | | **0.71** | 2,661,591 |
| **Macro Average** | 0.56 | 0.73 | 0.53 | 2,661,591 |
| **Weighted Average** | 0.93 | 0.71 | 0.79 | 2,661,591 |

The positive-class F1-score is reported as approximately 0.25 in the classification report, while the directly calculated F1-score used for baseline comparison is **0.2454**.

---

## 11. Results Interpretation

The Logistic Regression baseline provides a useful starting point for the future-purchase prediction problem.

The model achieved a **75% recall** for the Future Purchase class. This means that it was able to identify a substantial proportion of the actual future purchases in the test dataset.

However, the model's **15% precision** indicates that many of its future-purchase predictions were false positives.

This behavior is also visible in the confusion matrix, where the number of false positives (**719,331**) is considerably larger than the number of true positives (**123,814**).

The model therefore favors identifying more potential future purchases at the cost of producing many incorrect positive predictions.

The **0.2454 F1-score** reflects the difficulty of simultaneously achieving high precision and high recall for the positive class.

---

## 12. Class Imbalance Consideration

Class imbalance is an important consideration in this evaluation.

The training dataset was balanced to provide the Logistic Regression model with an equal number of positive and negative training examples.

In contrast, the test dataset retained the original distribution:

- **93.772% No Future Purchase**
- **6.228% Future Purchase**

This difference between training and testing distributions affects the evaluation results.

For this reason, accuracy should not be interpreted independently. Precision, recall, F1-score, and the confusion matrix provide more useful information about the model's ability to identify future purchases.

The baseline evaluation therefore uses the untouched test distribution rather than evaluating the model on a balanced test sample.

---

## 13. Challenges and Considerations

Several considerations were important during model evaluation.

### 13.1 Class Imbalance

The future-purchase class represents only 6.228% of the testing dataset. This makes accuracy less informative as a standalone metric.

### 13.2 False Positive Predictions

The model generated a large number of false-positive predictions. This resulted in a precision of only 15% for the Future Purchase class.

### 13.3 Training and Testing Distribution

The training dataset was balanced using random undersampling, while the testing dataset retained the original class distribution.

This approach allows the model to learn from a balanced training set while testing its performance under a more realistic class distribution.

### 13.4 Baseline Model Limitations

Logistic Regression provides a useful baseline, but the results indicate substantial room for improvement, particularly in positive-class precision and F1-score.

Future iterations can investigate more advanced models, improved feature engineering, class-weighting strategies, probability threshold optimization, or other approaches to improve the balance between precision and recall.

---

## 14. Evaluation Summary

The baseline model performance is summarized below:

| Metric | Result |
|---|---:|
| Accuracy | **71.40%** |
| Future Purchase Precision | **15%** |
| Future Purchase Recall | **75%** |
| Future Purchase F1-score | **0.2454** |
| Test Samples | **2,661,591** |

The baseline successfully identifies many future purchases, as demonstrated by its 75% recall.

However, its low precision results in a large number of false-positive predictions. Therefore, improving positive-class precision while maintaining useful recall is an important direction for future model iterations.

---

## 15. Conclusion

A Logistic Regression model was successfully trained and evaluated as the baseline classifier for future customer-product purchase prediction.

The model used standardized numerical features and one-hot encoded categorical features through a preprocessing pipeline.

Evaluation was performed on the untouched test dataset containing the original class distribution.

The baseline achieved:

- **71.40% accuracy**
- **15% precision** for Future Purchase
- **75% recall** for Future Purchase
- **0.2454 F1-score** for Future Purchase

The results demonstrate that the baseline model can identify a substantial proportion of future purchases, but its relatively low precision leads to many false-positive predictions.

Therefore, the Logistic Regression model serves as a useful reference point for subsequent model development and improvement.

---

## 16. Reference Implementation

The complete baseline model implementation, including:

- Data loading
- Feature selection
- Preprocessing
- Logistic Regression training
- Test-set prediction
- Performance metrics
- Classification report
- Confusion matrix
- Exploratory visualizations
- Model interpretation

is available in:

`notebooks/task4_baseline_model.ipynb`

This documentation complements the implementation by providing a concise description of the training methodology, evaluation approach, performance results, and key findings.

---

## 17. Task 5 Deliverables Checklist

The following Task 5 requirements have been addressed:

- [x] Accuracy calculated
- [x] Precision calculated
- [x] Recall calculated
- [x] F1-score calculated
- [x] Classification report documented
- [x] Confusion matrix documented
- [x] Model training methodology documented
- [x] Preprocessing methodology documented
- [x] Feature engineering/feature selection documented
- [x] Dataset and target distributions documented
- [x] Evaluation methodology documented
- [x] Challenges and considerations documented
- [x] Results interpreted
- [x] Baseline conclusion documented
- [x] Reference implementation identified