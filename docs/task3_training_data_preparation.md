# Task 3 — Training Data Preparation

## Overview

Task 3 prepares the engineered customer-product feature dataset from Task 2 for supervised machine learning.

The objective is to create a binary target indicating whether a customer purchased a product in a future order, followed by a reproducible train/test split and class-imbalance handling.

## Input Data

The Task 3 process uses:

- `output/customer_product_features.csv.gz`
- `data/orders.csv`
- `data/order_products__train.csv`

The Task 2 feature dataset contains 13,307,953 customer-product records and 21 engineered features.

## Target Creation

The future purchase target is created using the future training orders.

A `(user_id, product_id)` pair is assigned:

- `target = 1` when the customer-product pair appears in a future training order.
- `target = 0` when the customer-product pair does not appear in the future training orders.

The resulting dataset contains:

- 828,824 positive samples
- 12,479,129 negative samples

The positive class represents approximately 6.228% of the full dataset.

## Train/Test Split

The labeled dataset is divided into:

- 80% training data
- 20% testing data

A stratified split is used so that the original class distribution is preserved in both subsets.

The split uses:

```text
random_state = 42