# Dataset

## Source

This project uses the **Instacart Market Basket Analysis** dataset referenced by the Task 2 specification.

The dataset contains historical customer orders and product-level information used to build customer, customer-product, product, department, and aisle-level features.

## Local Dataset Files

The raw dataset is stored locally in this directory and is intentionally excluded from GitHub because of its size.

Expected source files include:

- `orders.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`
- `order_products__prior.csv`
- `order_products__train.csv`

The `order_products__prior.csv` file is particularly large, so the feature pipeline processes it in chunks rather than loading the complete file into memory.

## Data Usage

The feature engineering pipeline uses historical `prior` orders for feature generation.

The main relationships are:

```text
orders
   │
   ├── user_id
   └── order_id
          │
          ▼
order_products__prior
          │
          └── product_id
                    │
                    ▼
                products
                    │
             ┌──────┴──────┐
             ▼             ▼
        department       aisle
```

## Engineered Dataset

The feature pipeline generates:

```text
output/customer_product_features.csv.gz
```

The generated dataset is intended to be loaded into Snowflake for downstream analytics and machine-learning workflows.

Raw data files are not committed to this repository.