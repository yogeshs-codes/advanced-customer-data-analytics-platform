"""
Task 2 — Customer Demand Analysis
Feature Engineering Pipeline

This script reproduces the feature-engineering workflow developed in
task2_feature_engineering.ipynb.

The pipeline uses historical ("prior") orders only and creates:
- customer-level order features
- customer-product purchase features
- customer-product recency
- product-level features
- department/aisle category affinity features
- final customer-product feature dataset
- quality validation

Output:
    output/customer_product_features.csv.gz
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load source data
# ============================================================

products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

# Only the historical/prior orders are used for feature generation.
order_cols = [
    "order_id",
    "user_id",
    "eval_set",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
]

orders_hist = orders[order_cols].copy()
orders_hist = orders_hist[
    orders_hist["eval_set"] == "prior"
].copy()


# ============================================================
# 3. Create customer-level order features
# ============================================================

customer_features = (
    orders_hist
    .groupby("user_id")
    .agg(
        user_total_orders=("order_id", "count"),
        user_avg_days_between_orders=("days_since_prior_order", "mean"),
        user_avg_order_hour=("order_hour_of_day", "mean"),
        user_avg_order_dow=("order_dow", "mean"),
    )
    .reset_index()
)


# ============================================================
# 4. Build customer-product historical aggregates
# ============================================================

PRIOR_FILE = DATA_DIR / "order_products__prior.csv"

prior_cols = [
    "order_id",
    "product_id",
    "add_to_cart_order",
    "reordered",
]

# Order lookup used to attach customer and order-number information.
order_lookup = (
    orders_hist[
        ["order_id", "user_id", "order_number"]
    ]
    .set_index("order_id")
)

customer_product_stats = defaultdict(
    lambda: {
        "purchase_count": 0,
        "reorder_count": 0,
        "last_order_number": 0,
        "cart_position_sum": 0,
    }
)

chunk_size = 500_000
processed_rows = 0

for chunk in pd.read_csv(
    PRIOR_FILE,
    usecols=prior_cols,
    chunksize=chunk_size,
):
    chunk = chunk.join(
        order_lookup,
        on="order_id",
        how="inner",
    )

    grouped = (
        chunk
        .groupby(["user_id", "product_id"])
        .agg(
            purchase_count=("order_id", "count"),
            reorder_count=("reordered", "sum"),
            last_order_number=("order_number", "max"),
            cart_position_sum=("add_to_cart_order", "sum"),
        )
    )

    for (user_id, product_id), row in grouped.iterrows():
        key = (user_id, product_id)

        stats = customer_product_stats[key]

        stats["purchase_count"] += int(row["purchase_count"])
        stats["reorder_count"] += int(row["reorder_count"])
        stats["last_order_number"] = max(
            stats["last_order_number"],
            int(row["last_order_number"]),
        )
        stats["cart_position_sum"] += int(
            row["cart_position_sum"]
        )

    processed_rows += len(chunk)

    if processed_rows % 5_000_000 < chunk_size:
        print(f"Processed {processed_rows:,} prior purchase rows")


# ============================================================
# 5. Create customer-product features
# ============================================================

customer_product_features = pd.DataFrame.from_dict(
    customer_product_stats,
    orient="index",
)

customer_product_features.index = pd.MultiIndex.from_tuples(
    customer_product_features.index,
    names=["user_id", "product_id"],
)

customer_product_features = (
    customer_product_features.reset_index()
)

customer_product_features = (
    customer_product_features.rename(
        columns={
            "purchase_count":
                "user_product_purchase_count",
            "reorder_count":
                "user_product_reorder_count",
            "last_order_number":
                "user_product_last_order_number",
            "cart_position_sum":
                "user_product_cart_position_sum",
        }
    )
)

customer_product_features[
    "user_product_reorder_rate"
] = (
    customer_product_features[
        "user_product_reorder_count"
    ]
    / customer_product_features[
        "user_product_purchase_count"
    ]
)

customer_product_features[
    "user_product_avg_cart_position"
] = (
    customer_product_features[
        "user_product_cart_position_sum"
    ]
    / customer_product_features[
        "user_product_purchase_count"
    ]
)

customer_product_features = (
    customer_product_features.drop(
        columns=["user_product_cart_position_sum"]
    )
)


# ============================================================
# 6. Add customer-product recency
# ============================================================

user_order_summary = (
    orders_hist
    .groupby("user_id")
    .agg(
        user_last_order_number=("order_number", "max")
    )
    .reset_index()
)

customer_product_features = (
    customer_product_features.merge(
        user_order_summary,
        on="user_id",
        how="left",
    )
)

customer_product_features[
    "user_product_recency_orders"
] = (
    customer_product_features[
        "user_last_order_number"
    ]
    - customer_product_features[
        "user_product_last_order_number"
    ]
)

customer_product_features = (
    customer_product_features.drop(
        columns=["user_last_order_number"]
    )
)


# ============================================================
# 7. Create product-level features
# ============================================================

product_features = (
    customer_product_features
    .groupby("product_id")
    .agg(
        product_total_purchases=(
            "user_product_purchase_count",
            "sum",
        ),
        product_unique_users=(
            "user_id",
            "nunique",
        ),
        product_total_reorders=(
            "user_product_reorder_count",
            "sum",
        ),
    )
    .reset_index()
)

product_features[
    "product_reorder_rate"
] = (
    product_features[
        "product_total_reorders"
    ]
    / product_features[
        "product_total_purchases"
    ]
)

product_features = product_features.drop(
    columns=["product_total_reorders"]
)

# Add product category information.
product_features = product_features.merge(
    products[
        [
            "product_id",
            "aisle_id",
            "department_id",
        ]
    ],
    on="product_id",
    how="left",
)


# ============================================================
# 8. Create customer category affinity features
# ============================================================

customer_product_categories = (
    customer_product_features[
        [
            "user_id",
            "product_id",
            "user_product_purchase_count",
        ]
    ]
    .merge(
        product_features[
            [
                "product_id",
                "aisle_id",
                "department_id",
            ]
        ],
        on="product_id",
        how="left",
    )
)

# Department affinity.
user_department_features = (
    customer_product_categories
    .groupby(["user_id", "department_id"])
    .agg(
        user_department_purchase_count=(
            "user_product_purchase_count",
            "sum",
        )
    )
    .reset_index()
)

user_total_purchase_counts = (
    user_department_features
    .groupby("user_id")[
        "user_department_purchase_count"
    ]
    .transform("sum")
)

user_department_features[
    "user_department_purchase_share"
] = (
    user_department_features[
        "user_department_purchase_count"
    ]
    / user_total_purchase_counts
)

# Aisle affinity.
user_aisle_features = (
    customer_product_categories
    .groupby(["user_id", "aisle_id"])
    .agg(
        user_aisle_purchase_count=(
            "user_product_purchase_count",
            "sum",
        )
    )
    .reset_index()
)

user_total_aisle_purchases = (
    user_aisle_features
    .groupby("user_id")[
        "user_aisle_purchase_count"
    ]
    .transform("sum")
)

user_aisle_features[
    "user_aisle_purchase_share"
] = (
    user_aisle_features[
        "user_aisle_purchase_count"
    ]
    / user_total_aisle_purchases
)


# ============================================================
# 9. Attach department and aisle affinity
# ============================================================

product_department_map = (
    product_features[
        ["product_id", "department_id"]
    ]
    .drop_duplicates("product_id")
)

customer_product_features = (
    customer_product_features.merge(
        product_department_map,
        on="product_id",
        how="left",
    )
)

customer_product_features = (
    customer_product_features.merge(
        user_department_features[
            [
                "user_id",
                "department_id",
                "user_department_purchase_count",
                "user_department_purchase_share",
            ]
        ],
        on=["user_id", "department_id"],
        how="left",
    )
)

customer_product_features[
    [
        "user_department_purchase_count",
        "user_department_purchase_share",
    ]
] = (
    customer_product_features[
        [
            "user_department_purchase_count",
            "user_department_purchase_share",
        ]
    ].fillna(0)
)

product_aisle_map = (
    product_features[
        ["product_id", "aisle_id"]
    ]
    .drop_duplicates("product_id")
)

customer_product_features = (
    customer_product_features.merge(
        product_aisle_map,
        on="product_id",
        how="left",
    )
)

customer_product_features = (
    customer_product_features.merge(
        user_aisle_features[
            [
                "user_id",
                "aisle_id",
                "user_aisle_purchase_count",
                "user_aisle_purchase_share",
            ]
        ],
        on=["user_id", "aisle_id"],
        how="left",
    )
)

customer_product_features[
    [
        "user_aisle_purchase_count",
        "user_aisle_purchase_share",
    ]
] = (
    customer_product_features[
        [
            "user_aisle_purchase_count",
            "user_aisle_purchase_share",
        ]
    ].fillna(0)
)


# ============================================================
# 10. Attach customer-level features
# ============================================================

customer_product_features = (
    customer_product_features.merge(
        customer_features,
        on="user_id",
        how="left",
    )
)


# ============================================================
# 11. Attach product-level features
# ============================================================

product_cols = [
    "product_id",
    "product_total_purchases",
    "product_unique_users",
    "product_reorder_rate",
]

customer_product_features = (
    customer_product_features.merge(
        product_features[product_cols],
        on="product_id",
        how="left",
    )
)


# ============================================================
# 12. Remove any accidental duplicate columns
# ============================================================

customer_feature_names = [
    "user_total_orders",
    "user_avg_days_between_orders",
    "user_avg_order_hour",
    "user_avg_order_dow",
]

for col in customer_feature_names:
    x_col = col + "_x"
    y_col = col + "_y"

    if x_col in customer_product_features.columns:
        if y_col in customer_product_features.columns:
            same_values = (
                customer_product_features[x_col]
                .equals(
                    customer_product_features[y_col]
                )
            )

            print(
                f"{col}: _x and _y identical = "
                f"{same_values}"
            )

            customer_product_features = (
                customer_product_features.drop(
                    columns=[y_col]
                )
            )

        customer_product_features = (
            customer_product_features.rename(
                columns={x_col: col}
            )
        )


# ============================================================
# 13. Final column order
# ============================================================

final_columns = [
    "user_id",
    "product_id",
    "user_product_purchase_count",
    "user_product_reorder_count",
    "user_product_last_order_number",
    "user_product_reorder_rate",
    "user_product_avg_cart_position",
    "user_product_recency_orders",
    "department_id",
    "user_department_purchase_count",
    "user_department_purchase_share",
    "aisle_id",
    "user_aisle_purchase_count",
    "user_aisle_purchase_share",
    "user_total_orders",
    "user_avg_days_between_orders",
    "user_avg_order_hour",
    "user_avg_order_dow",
    "product_total_purchases",
    "product_unique_users",
    "product_reorder_rate",
]

customer_product_features = (
    customer_product_features[final_columns]
)


# ============================================================
# 14. Feature quality validation
# ============================================================

duplicate_count = (
    customer_product_features
    .duplicated(
        subset=["user_id", "product_id"]
    )
    .sum()
)

missing_values = (
    customer_product_features.isna().sum()
)

numeric_columns = (
    customer_product_features
    .select_dtypes(include=["number"])
    .columns
)

infinite_counts = (
    customer_product_features[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
)

rate_features = [
    "user_product_reorder_rate",
    "user_department_purchase_share",
    "user_aisle_purchase_share",
    "product_reorder_rate",
]

count_features = [
    "user_product_purchase_count",
    "user_product_reorder_count",
    "user_product_last_order_number",
    "user_product_avg_cart_position",
    "user_product_recency_orders",
    "user_department_purchase_count",
    "user_aisle_purchase_count",
    "user_total_orders",
    "product_total_purchases",
    "product_unique_users",
]

print("\n=== FEATURE DATASET VALIDATION ===")
print("Shape:", customer_product_features.shape)
print("Duplicate customer-product pairs:", duplicate_count)
print("Total missing values:", missing_values.sum())
print("Total infinite values:", infinite_counts.sum())

print("\nRate/share range checks:")
for col in rate_features:
    print(
        f"{col}: "
        f"min={customer_product_features[col].min():.4f}, "
        f"max={customer_product_features[col].max():.4f}"
    )

print("\nNegative value checks:")
for col in count_features:
    negative_count = (
        customer_product_features[col] < 0
    ).sum()
    print(f"{col}: {negative_count}")


# ============================================================
# 15. Save final engineered dataset
# ============================================================

output_file = (
    OUTPUT_DIR / "customer_product_features.csv.gz"
)

customer_product_features.to_csv(
    output_file,
    index=False,
    compression="gzip",
)

print("\nFinal assembled feature shape:")
print(customer_product_features.shape)

print("\nSaved engineered dataset:")
print(output_file)

print("\nFeature pipeline completed successfully.")
