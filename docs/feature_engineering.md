# Task 2 --- Feature Engineering Documentation

## 1. Task Overview

This document records the completed work for **Phase 1 --- Task 2: Build
Feature Pipeline** of the Advanced Customer Data Analytics project.

The task objective is to create a feature pipeline that extracts,
transforms, and engineers relevant features from the Instacart customer
purchasing data for later predictive analytics.

The Task 2 specification requires: 1. Collecting raw customer data. 2.
Cleaning the data using Python/Pandas. 3. Engineering relevant customer
behavior features. 4. Storing the transformed features for further
analysis or real-time processing.

## 2. Data Source

The project uses the supplied **Instacart Market Basket Analysis**
dataset.

The pipeline uses historical orders, products, aisles, departments, and
order-product information. The engineered dataset has the analytical
grain:

**One row = one unique customer-product pair.**

## 3. Feature Pipeline

The reusable implementation is stored in:

`src/feature_pipeline.py`

The large historical purchase input is processed incrementally. Progress
is reported during processing so that the pipeline can handle the large
source file without requiring the entire file to be loaded at once.

The final feature dataset contains:

-   **13,307,953 rows**
-   **21 columns**
-   **0 duplicate customer-product pairs**
-   **0 missing values**
-   **0 infinite values**

## 4. Engineered Features

### Customer-product features

  -----------------------------------------------------------------------
  Feature                             Rationale
  ----------------------------------- -----------------------------------
  `user_product_purchase_count`       Measures customer-product
                                      purchasing affinity.

  `user_product_reorder_count`        Captures repeat purchasing
                                      behavior.

  `user_product_last_order_number`    Provides a recency-related signal.

  `user_product_reorder_rate`         Measures the customer's tendency to
                                      reorder the product.

  `user_product_avg_cart_position`    Captures the product's typical
                                      position in the customer's cart.

  `user_product_recency_orders`       Measures how many orders have
                                      passed since the product was last
                                      purchased.
  -----------------------------------------------------------------------

### Customer-category features

  -----------------------------------------------------------------------
  Feature                             Rationale
  ----------------------------------- -----------------------------------
  `department_id`                     Identifies the product department.

  `user_department_purchase_count`    Measures customer activity within
                                      the department.

  `user_department_purchase_share`    Measures the department's share of
                                      customer purchases.

  `aisle_id`                          Identifies the product aisle.

  `user_aisle_purchase_count`         Measures customer activity within
                                      the aisle.

  `user_aisle_purchase_share`         Measures the aisle's share of
                                      customer purchases.
  -----------------------------------------------------------------------

### Customer-level features

  -----------------------------------------------------------------------
  Feature                             Rationale
  ----------------------------------- -----------------------------------
  `user_total_orders`                 Represents overall customer
                                      purchasing activity.

  `user_avg_days_between_orders`      Represents the customer's ordering
                                      cadence.

  `user_avg_order_hour`               Captures typical ordering time.

  `user_avg_order_dow`                Captures ordering day-of-week
                                      behavior.
  -----------------------------------------------------------------------

### Product-level features

  -----------------------------------------------------------------------
  Feature                             Rationale
  ----------------------------------- -----------------------------------
  `product_total_purchases`           Measures overall product
                                      popularity.

  `product_unique_users`              Measures breadth of product
                                      adoption.

  `product_reorder_rate`              Measures product-level
                                      repeat-purchase behavior.
  -----------------------------------------------------------------------

The complete 21-column dataset also includes `user_id` and `product_id`
as the customer and product identifiers.

## 5. Data Quality Validation

The completed pipeline performs validation before saving the dataset.

### Dataset shape

``` text
(13,307,953, 21)
```

### Uniqueness

``` text
Duplicate customer-product pairs: 0
```

### Missing and infinite values

``` text
Total missing values: 0
Total infinite values: 0
```

### Rate/share checks

``` text
user_product_reorder_rate:
    min = 0.0000
    max = 0.9899

user_department_purchase_share:
    min = 0.0003
    max = 1.0000

user_aisle_purchase_share:
    min = 0.0003
    max = 1.0000

product_reorder_rate:
    min = 0.0000
    max = 0.9412
```

### Negative-value checks

All checked count/order/product features contained zero negative values.

## 6. Local Output

The pipeline successfully generated:

`output/customer_product_features.csv.gz`

Because the compressed file was approximately 494 MB, it exceeded the
direct Snowflake browser upload limit. The file was therefore split into
seven compressed parts for upload.

``` text
output/feature_parts/
    customer_product_features_part_1.csv.gz
    customer_product_features_part_2.csv.gz
    customer_product_features_part_3.csv.gz
    customer_product_features_part_4.csv.gz
    customer_product_features_part_5.csv.gz
    customer_product_features_part_6.csv.gz
    customer_product_features_part_7.csv.gz
```

The seven parts together contain the complete **13,307,953-row**
engineered dataset.

## 7. Snowflake Storage

The engineered dataset was loaded into Snowflake.

``` text
Database: CUSTOMER_DEMAND
Schema:   FEATURE_ENGINEERING
Table:    CUSTOMER_PRODUCT_FEATURES
```

Post-load validation returned:

``` text
ROW_COUNT                       13,307,953
UNIQUE_CUSTOMER_PRODUCT_PAIRS  13,307,953
NULL_USER_IDS                  0
NULL_PRODUCT_IDS               0
```

This confirms that the Snowflake table contains the expected number of
rows and unique customer-product pairs, with no NULL customer or product
identifiers.

## 8. Feature Selection Rationale

The features cover several important dimensions of purchasing behavior:

-   **Recency:** recent interaction with a product.
-   **Frequency:** how often the customer orders and buys a product.
-   **Repeat purchasing:** reorder counts and reorder rates.
-   **Customer preference:** department and aisle purchase shares.
-   **Product popularity:** total purchases and unique customers.
-   **Ordering behavior:** order interval, hour, and day-of-week
    patterns.

This combines customer-product, customer-category, customer-level, and
product-level signals for subsequent predictive modeling.

## 9. Efficiency and Scalability

The pipeline was implemented as a reusable Python script and processes
the large historical input incrementally.

The workflow: - processes the large input in chunks; - reports progress
during execution; - creates compressed output; - splits large output
only when required for transfer; - stores the engineered data in
Snowflake for scalable analytical access.

## 10. Reproducibility

The feature engineering logic is implemented in:

`src/feature_pipeline.py`

The generated feature dataset is validated before completion and is
stored locally and in Snowflake.

This makes the feature extraction and transformation process
reproducible from the available project data.

## 11. Step 2 Completion

**Build Feature Pipeline --- COMPLETED**

Completed deliverables:

1.  **Python feature extraction/transformation pipeline:**
    `src/feature_pipeline.py`
2.  **Clean engineered feature dataset:** stored locally and in
    Snowflake.
3.  **Feature engineering documentation:** this document.

The completed feature dataset contains **13,307,953 rows and 21
columns**, with no duplicate customer-product pairs, missing values, or
infinite values.

## 12. Next Step

The next roadmap stage is **Step 3 --- Prepare Training Data**.

According to the Task 2 specification, this involves: - splitting the
data; - handling class imbalance; - creating reproducible train/test
sets.

The training-data stage must also avoid data leakage between training
and test data.
