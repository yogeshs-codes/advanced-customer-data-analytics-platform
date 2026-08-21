# Task 5 — Evaluate & Document

## 1. Objective

The objective of Task 5 is to evaluate the candidate machine learning models, select the best-performing model, validate its performance using cross-validation, evaluate it on an independent test dataset, and document the final model selection.

The task focuses on predicting whether a customer will make a future purchase of a product.

Model selection was based primarily on **ROC-AUC** and **F1-score**, while **recall** was also considered because identifying future purchases is an important objective of the problem.

After model evaluation and validation, **Gradient Boosting** was selected as the final model.

---

## 2. Dataset and Data Preparation

The project uses the prepared customer-product datasets generated during the earlier data preparation stages.

The Task 3 process produced separate training and testing datasets.

- **Training dataset:** 1,326,118 rows
- **Testing dataset:** 2,661,591 rows
- **Selected features:** 19

**Target variable:**
- `0` — No Future Purchase
- `1` — Future Purchase

### Target Distribution

The training dataset was balanced through random undersampling:

| Target Class | Count | Proportion |
| :--- | ---: | ---: |
| **No Future Purchase (0)** | 663,059 | 50.00% |
| **Future Purchase (1)** | 663,059 | 50.00% |
| **Total** | **1,326,118** | **100.00%** |

The testing dataset retained the original class distribution:

| Target Class | Count | Proportion |
| :--- | ---: | ---: |
| **No Future Purchase (0)** | 2,495,826 | 93.772% |
| **Future Purchase (1)** | 165,765 | 6.228% |
| **Total** | **2,661,591** | **100.000%** |

Maintaining the original distribution in the test dataset provides a more realistic evaluation of how the final model performs on the expected class distribution.

---

## 3. Selected Features

A total of **19 features** were used by the final model.

### Categorical / Identifier Features

- `department_id`
- `aisle_id`

### Numerical Features

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

No missing values were present in the training or testing feature datasets.

---

## 4. Final Model Selection and Storage

Candidate models were evaluated using the model-selection workflow developed for Task 5. The final model was selected based on cross-validation performance, with emphasis on ROC-AUC and F1-score while also considering recall.

**Gradient Boosting** was selected as the final model because it achieved strong and consistent cross-validation performance and maintained a very small ROC-AUC difference between cross-validation and independent test evaluation.

The final model configuration is:

```text
GradientBoostingClassifier(max_depth=5, random_state=42)
```

The final serialized model is stored at:

```text
output/models/final_selected_model.joblib
```

The saved artifact was independently loaded and verified successfully.

---

## 5. Validation Methodology

Cross-validation was used to estimate the final model's performance across multiple validation splits.

The following metrics were evaluated:

- **ROC-AUC:** Measures the model's ability to distinguish between positive and negative classes across classification thresholds.
- **F1-score:** Evaluates the balance between precision and recall for the positive (*Future Purchase*) class.
- **Recall:** Monitored because identifying actual future purchases is an important objective.

### Cross-Validation Results — Gradient Boosting

| Metric | Mean | Standard Deviation |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.812667 | 0.003678 |
| **F1-score** | 0.740168 | 0.002504 |
| **Recall** | 0.745800 | — |

The relatively small standard deviations indicate that the model produced consistent performance across the validation folds.

---

## 6. Independent Test Evaluation

After cross-validation, the selected Gradient Boosting model was evaluated on the independent test dataset containing **2,661,591 observations**.

The test dataset was not balanced and retained the original class distribution.

### Test Dataset Performance

| Metric | Test Performance |
| :--- | :---: |
| **Accuracy** | 0.730815 (73.08%) |
| **Precision** | 0.155557 (15.56%) |
| **Recall** | 0.750170 (75.02%) |
| **F1-score** | 0.257680 |
| **ROC-AUC** | 0.815208 |

These results provide an objective evaluation of the final model under the original, highly imbalanced class distribution.

---

## 7. Performance Interpretation

### 7.1 Accuracy

**Result:** `0.730815` (73.08%)

Accuracy represents the proportion of all test observations classified correctly. Because the test dataset contains a large majority of negative examples, accuracy should not be interpreted as the sole measure of model quality.

### 7.2 Precision

**Result:** `0.155557` (15.56%)

For the *Future Purchase* class, only a relatively small proportion of observations predicted as future purchases were true positives. The low precision is strongly influenced by the severe class imbalance in the independent test dataset.

### 7.3 Recall

**Result:** `0.750170` (75.02%)

The model successfully identified approximately 75% of actual future purchases in the independent test dataset. This strong recall aligns with the objective of identifying potential future purchases.

### 7.4 F1-score

**Result:** `0.257680`

F1-score combines precision and recall using their harmonic mean. Although the model achieved strong recall, the low precision reduces the overall F1-score. Improving precision while maintaining useful recall remains an important area for future improvement.

### 7.5 ROC-AUC

**Result:** `0.815208`

The ROC-AUC result indicates good ability to distinguish between future-purchase and no-future-purchase cases. The test ROC-AUC is also very close to the cross-validation ROC-AUC, indicating stable generalization of the model's ranking ability.

---

## 8. Cross-Validation vs. Independent Test Performance

| Metric | Cross-Validation Mean | Independent Test | Generalization Gap |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.812667 | 0.815208 | -0.002540 |
| **Recall** | 0.745800 | 0.750170 | -0.004370 |
| **F1-score** | 0.740168 | 0.257680 | 0.482487 |

### Analysis

- **ROC-AUC consistency:** Cross-validation ROC-AUC (0.812667) and test ROC-AUC (0.815208) differ by only -0.002540, indicating stable ranking performance on unseen data.
- **Recall consistency:** Cross-validation recall (0.745800) and test recall (0.750170) are also very close.
- **F1-score difference:** The F1-score is substantially lower on the independent test dataset because the validation/training data were balanced while the independent test dataset retains the original 93.772% negative and 6.228% positive distribution. The resulting increase in false positives reduces precision and therefore F1-score.

---

## 9. Generalization Analysis

### ROC-AUC Generalization Gap

`-0.002540`

The near-zero ROC-AUC gap indicates that the final model did not show meaningful degradation in its ability to distinguish between the two classes when evaluated on unseen test data.

### F1 Generalization Gap

`0.482487`

The larger F1-score difference is primarily associated with the change from the balanced training/validation distribution to the highly imbalanced independent test distribution. Therefore, this difference should not be interpreted by itself as evidence of severe overfitting.

Overall, the ROC-AUC and recall results provide stronger evidence that the final model generalizes reasonably well to the independent test dataset.

---

## 10. Class Imbalance Consideration

Class imbalance is a major consideration in this project.

### Training Dataset

- **No Future Purchase:** 50.00%
- **Future Purchase:** 50.00%

### Independent Test Dataset

- **No Future Purchase:** 93.772%
- **Future Purchase:** 6.228%

The training dataset was balanced through random undersampling, while the independent test dataset retained the original distribution.

This difference affects metrics such as precision and F1-score. A model trained on a balanced dataset can achieve strong recall during validation, but when evaluated on the original negative-heavy distribution, false-positive predictions can become much more prominent.

For this reason, ROC-AUC, precision, recall, and F1-score should be considered together rather than relying only on accuracy.

---

## 11. Model Size and Serialization

The final Gradient Boosting model was serialized using `joblib`.

- **Artifact path:** `output/models/final_selected_model.joblib`
- **Serialized model size:** 0.4642 MB
- **Model class:** `<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>`
- **Configuration:** `GradientBoostingClassifier(max_depth=5, random_state=42)`
- **Input feature count:** 19

The saved model was independently reloaded and verified successfully. The loaded object was confirmed as a `GradientBoostingClassifier`, and the model was confirmed to contain 19 input features.

This confirms that the final model artifact is available for future inference and deployment workflows.

---

## 12. Challenges and Considerations

### 12.1 Strong Class Imbalance

The independent test dataset contains only 6.228% positive cases. Therefore, accuracy is less informative as a standalone metric.

### 12.2 Low Positive-Class Precision

The final model achieved 15.56% precision for the positive class. This indicates that a substantial number of positive predictions are false positives.

### 12.3 Training and Testing Distribution

The training dataset was balanced through random undersampling, while the independent test dataset retained the original distribution. This explains the considerable difference between validation F1-score and independent-test F1-score.

### 12.4 Metric Selection

Multiple complementary metrics were considered:

- ROC-AUC for class discrimination and ranking ability
- Recall for the proportion of actual future purchases identified
- Precision for the reliability of positive predictions
- F1-score for the balance between precision and recall
- Accuracy as an overall classification measure

### 12.5 Model Complexity

Gradient Boosting provides a nonlinear model capable of capturing relationships that a simple linear baseline may not capture.

The selected configuration uses:

```text
max_depth=5
random_state=42
```

The final model was validated using cross-validation and then evaluated on an independent test dataset, reducing reliance on training performance alone.

---

## 13. Final Model Evaluation Summary

### Core Performance Metrics

| Metric | Cross-Validation | Independent Test |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.812667 | 0.815208 |
| **F1-score** | 0.740168 | 0.257680 |
| **Recall** | 0.745800 | 0.750170 |
| **Accuracy** | — | 0.730815 |
| **Precision** | — | 0.155557 |

### Validation and Artifact Summary

| Measure | Result |
| :--- | :---: |
| **ROC-AUC CV Standard Deviation** | 0.003678 |
| **F1 CV Standard Deviation** | 0.002504 |
| **ROC-AUC Generalization Gap** | -0.002540 |
| **F1 Generalization Gap** | 0.482487 |
| **Serialized Model Size** | 0.4642 MB |
| **Number of Features** | 19 |

---

## 14. Conclusion

The model evaluation and validation process established **Gradient Boosting** as the final selected model for future customer-product purchase prediction.

The final model achieved:

- **Test ROC-AUC:** `0.815208`
- **Test Recall:** `0.750170`
- **Test Precision:** `0.155557`
- **Test F1-score:** `0.257680`
- **Test Accuracy:** `0.730815`

The test ROC-AUC was highly consistent with cross-validation performance, with a generalization gap of only `-0.002540`. This indicates stable class-separation performance on unseen data.

The model also achieved approximately 75% recall, meaning it identified a substantial proportion of actual future purchases. However, the positive-class precision was relatively low at 15.56%, resulting in a lower F1-score.

The main factor affecting precision and F1-score is the strong difference between the balanced training distribution and the naturally imbalanced independent test distribution.

Overall, Gradient Boosting provides a stronger and more flexible final model than the initial baseline approach and demonstrates good ROC-AUC and recall performance.

Future improvements can focus on increasing positive-class precision and F1-score through probability-threshold optimization, class-weighting or cost-sensitive strategies where appropriate, and additional behavioral feature engineering.

---

## 15. Final Model Artifact

```text
Model File:           output/models/final_selected_model.joblib
Model Type:           GradientBoostingClassifier
Configuration:        GradientBoostingClassifier(max_depth=5, random_state=42)
Input Features:       19
Serialized Size:      0.4642 MB
Verification Status:  Successfully loaded and validated via joblib
```

---

## 16. Task 5 Deliverables Checklist

- [x] Candidate model evaluation completed
- [x] Final model selected
- [x] Cross-validation performed
- [x] ROC-AUC calculated
- [x] Accuracy calculated
- [x] Precision calculated
- [x] Recall calculated
- [x] F1-score calculated
- [x] Independent test evaluation completed
- [x] Generalization performance analyzed
- [x] Class imbalance considered
- [x] Model configuration documented
- [x] Feature set documented
- [x] Final model serialized
- [x] Serialized model verified
- [x] Model size documented
- [x] Results interpreted
- [x] Challenges and limitations documented
- [x] Final model conclusion documented
