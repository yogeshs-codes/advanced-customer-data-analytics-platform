# Task 2 — Feature Engineering Pipeline

## 1. Objective

The objective of Task 2 is to build a reproducible feature engineering pipeline that extracts, transforms, and engineers useful features from raw customer transaction data.

The resulting feature dataset is designed for downstream customer-demand analysis and machine-learning tasks.

The pipeline produces customer-product level features while also incorporating customer-level, product-level, department-level, and aisle-level behavioral information.

---

## 2. Source Data

The pipeline uses the Instacart Market Basket Analysis dataset.

The main source tables are:

| Dataset | Purpose |
|---|---|
| `orders.csv` | Customer order history and order-level attributes |
| `order_products__prior.csv` | Historical customer-product transactions |
| `products.csv` | Product metadata and category relationships |
| `aisles.csv` | Aisle information |
| `departments.csv` | Department information |

Only historical orders where:

```text
eval_set == "prior"
```

are used for feature generation.

This prevents future/evaluation orders from being included in the historical feature calculations.

---

## 3. Pipeline Architecture

The feature engineering workflow is:

```text
Raw Instacart Data
        │
        ▼
Load Orders + Products
        │
        ▼
Filter Historical Prior Orders
        │
        ▼
Customer-Level Features
        │
        ├─────────────────────┐
        ▼                     ▼
Customer-Product        Product Features
Features                     │
        │                     │
        └──────────┬──────────┘
                   ▼
          Category Affinity
        Department + Aisle
                   │
                   ▼
        Feature Integration
                   │
                   ▼
          Quality Validation
                   │
                   ▼
     customer_product_features
                   │
                   ▼
              CSV.GZ
                   │
                   ▼
              Snowflake
```

---

## 4. Data Preparation

The pipeline first loads the required source tables using Pandas.

For the order table, only the columns required for feature generation are selected:

- `order_id`
- `user_id`
- `eval_set`
- `order_number`
- `order_dow`
- `order_hour_of_day`
- `days_since_prior_order`

The order data is then filtered to historical `prior` orders.

This creates the historical basis used for customer and customer-product feature calculations.

---

## 5. Customer-Level Features

Customer-level behavioral features are calculated by grouping historical orders by `user_id`.

The following features are generated:

### Total Orders

`user_total_orders`

Number of historical orders associated with the customer.

### Average Days Between Orders

`user_avg_days_between_orders`

Average number of days between the customer's historical orders.

### Average Order Hour

`user_avg_order_hour`

Average hour of the day at which the customer places orders.

### Average Order Day

`user_avg_order_dow`

Average day-of-week value across the customer's historical orders.

These features provide a high-level representation of customer purchasing behavior.

---

## 6. Customer-Product Features

The largest source table, `order_products__prior.csv`, contains historical product purchases.

Because this file is large, it is processed using Pandas chunking.

The pipeline uses:

```text
chunk_size = 500,000
```

This avoids loading the entire transaction table into memory at once.

Each chunk is joined with historical order information and aggregated by:

```text
(user_id, product_id)
```

### Purchase Count

`user_product_purchase_count`

Number of historical purchases of a product by a customer.

### Reorder Count

`user_product_reorder_count`

Number of times the customer reordered the product.

### Reorder Rate

`user_product_reorder_rate`

Calculated as:

```text
reorder_count / purchase_count
```

This provides an indication of how consistently a customer repurchases a product.

### Last Product Order

`user_product_last_order_number`

The customer's most recent historical order number containing the product.

### Average Cart Position

`user_product_avg_cart_position`

Average position of the product in the customer's cart.

This can provide information about product ordering behavior and shopping-list patterns.

---

## 7. Customer-Product Recency

A customer-product recency feature is calculated using the customer's most recent historical order number and the last order number in which the product was purchased.

Feature:

```text
user_product_recency_orders
```

Formula:

```text
customer_last_order_number
-
product_last_order_number
```

A smaller value indicates that the product was purchased more recently relative to the customer's latest historical order.

This is an important behavioral signal for future demand prediction.

---

## 8. Product-Level Features

Product-level statistics are calculated across customers.

The pipeline creates:

### Total Product Purchases

`product_total_purchases`

Total number of historical purchases represented by the engineered customer-product records.

### Unique Customers

`product_unique_users`

Number of unique customers who purchased the product.

### Product Reorder Rate

`product_reorder_rate`

Calculated from the total reorder count divided by total product purchases.

These features provide a measure of overall product popularity and repeat-purchase behavior.

---

## 9. Department Affinity

Customer purchasing behavior is aggregated by department.

The pipeline creates:

`user_department_purchase_count`

and:

`user_department_purchase_share`

Purchase share is calculated as the customer's purchases within a department divided by the customer's total purchases represented in the department-level feature table.

This provides a measure of the customer's preference for particular product departments.

---

## 10. Aisle Affinity

A similar approach is used for aisles.

The pipeline creates:

`user_aisle_purchase_count`

and:

`user_aisle_purchase_share`

These features capture more granular customer product-category preferences.

---

## 11. Feature Integration

The different feature groups are combined into a final customer-product dataset.

The final dataset contains:

### Customer-product behavior

- `user_product_purchase_count`
- `user_product_reorder_count`
- `user_product_last_order_number`
- `user_product_reorder_rate`
- `user_product_avg_cart_position`
- `user_product_recency_orders`

### Department affinity

- `department_id`
- `user_department_purchase_count`
- `user_department_purchase_share`

### Aisle affinity

- `aisle_id`
- `user_aisle_purchase_count`
- `user_aisle_purchase_share`

### Customer behavior

- `user_total_orders`
- `user_avg_days_between_orders`
- `user_avg_order_hour`
- `user_avg_order_dow`

### Product behavior

- `product_total_purchases`
- `product_unique_users`
- `product_reorder_rate`

---

## 12. Feature Rationale

The selected features were chosen because they represent different dimensions of customer demand behavior.

| Feature Group | Reason |
|---|---|
| Purchase count | Measures historical customer-product demand |
| Reorder count | Captures repeat purchasing behavior |
| Reorder rate | Measures product loyalty/repeat tendency |
| Recency | Captures how recently a customer purchased a product |
| Cart position | Represents ordering/cart behavior |
| Customer order count | Represents overall customer activity |
| Average days between orders | Represents purchase cadence |
| Product popularity | Identifies broadly popular products |
| Department share | Captures broad category preferences |
| Aisle share | Captures more granular product preferences |

Together, these features provide a richer representation of customer-product demand than using raw transaction records alone.

---

## 13. RFM Consideration

The Task 2 recommendations mention Recency, Frequency, and Monetary Value (RFM).

The implemented pipeline includes strong recency and frequency-related behavioral features.

However, the supplied Instacart source tables used in this project do not contain product price or transaction revenue information.

Therefore, a true monetary-value feature is **not fabricated** in this implementation.

If transaction price or revenue data becomes available, monetary-value features can be added in a future version.

---

## 14. Data Quality Validation

Before writing the final dataset, the pipeline checks:

- Duplicate `(user_id, product_id)` pairs
- Missing values
- Infinite numeric values
- Negative values in count-based features
- Range of rate/share features

The pipeline prints these validation results so that feature quality can be reviewed before downstream use.

---

## 15. Output

The final engineered dataset is written to:

```text
output/customer_product_features.csv.gz
```

The compressed output format reduces storage requirements while preserving the complete feature dataset.

The dataset is intended to be loaded into Snowflake for downstream analytics and machine-learning workflows.

---

## 16. Snowflake Storage

The repository contains:

```text
sql/snowflake_setup.sql
```

This script:

1. Creates the `CUSTOMER_DEMAND` database.
2. Creates the `FEATURE_ENGINEERING` schema.
3. Creates an XSMALL warehouse.
4. Creates the `CUSTOMER_PRODUCT_FEATURES` table.
5. Provides commands for loading the generated feature dataset.
6. Provides validation queries for the loaded data.

The target table mirrors the feature columns produced by the Python pipeline.

---

## 17. Reproducibility

The main implementation is available in:

```text
src/feature_pipeline.py
```

The notebook version is available in:

```text
notebooks/task2_feature_engineering.ipynb
```

The pipeline can be reproduced by ensuring the required raw dataset files are available locally and running:

```bash
python src/feature_pipeline.py
```

The pipeline then generates:

```text
output/customer_product_features.csv.gz
```

---

## 18. Efficiency Considerations

The `order_products__prior.csv` file is processed in chunks rather than being loaded entirely into memory.

The pipeline also:

- selects only required columns from the large transaction file
- performs aggregation during chunk processing
- uses lookup information from the order table
- compresses the final CSV output

These decisions reduce memory pressure and make the workflow more suitable for a large transactional dataset.

---

## 19. Limitations and Future Improvements

Potential improvements include:

- Adding transaction price/revenue data to support monetary-value features.
- Moving large-scale aggregation to Snowflake for improved scalability.
- Loading the engineered dataset directly into Snowflake as part of an automated pipeline.
- Adding automated unit tests for feature calculations.
- Adding pipeline logging and execution metrics.
- Creating versioned feature datasets for reproducible experiments.
- Adding additional temporal and customer-product behavioral features.

---

## 20. Task 2 Deliverables

| Deliverable | Repository Location |
|---|---|
| Feature engineering Python pipeline | `src/feature_pipeline.py` |
| Notebook implementation | `notebooks/task2_feature_engineering.ipynb` |
| Snowflake setup/loading SQL | `sql/snowflake_setup.sql` |
| Feature engineering documentation | `docs/feature_engineering.md` |
| Dataset documentation | `data/README.md` |
| Generated engineered dataset | `output/customer_product_features.csv.gz` locally / Snowflake for storage |

The raw source dataset is intentionally excluded from GitHub because of its size.