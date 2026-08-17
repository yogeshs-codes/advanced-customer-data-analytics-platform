"""
Task 3 — Prepare Training Data

Creates a supervised learning dataset by combining:
1. Task 2 engineered customer-product features
2. Future purchase information from Instacart training orders

Target:
    1 = customer purchased the product in the future order
    0 = customer did not purchase the product in the future order

Class imbalance:
    Random undersampling is applied only to the training set.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

FEATURE_FILE = OUTPUT_DIR / "customer_product_features.csv.gz"

ORDERS_FILE = DATA_DIR / "orders.csv"
TRAIN_PRODUCTS_FILE = DATA_DIR / "order_products__train.csv"


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 500_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Step 3.1 — Load future training-order information
# ============================================================

print("=" * 70)
print("TASK 3 — PREPARE TRAINING DATA")
print("=" * 70)

print("\nLoading future training orders...")

orders = pd.read_csv(
    ORDERS_FILE,
    usecols=[
        "order_id",
        "user_id",
        "eval_set"
    ]
)

train_orders = orders[
    orders["eval_set"] == "train"
][
    ["order_id", "user_id"]
].copy()

print(
    f"Future training orders: "
    f"{len(train_orders):,}"
)


# ============================================================
# Step 3.2 — Build positive customer-product pairs
# ============================================================

print("\nBuilding future purchase pairs...")

future_products = pd.read_csv(
    TRAIN_PRODUCTS_FILE,
    usecols=[
        "order_id",
        "product_id"
    ]
)

future_pairs = (
    future_products
    .merge(
        train_orders,
        on="order_id",
        how="inner"
    )
    [
        ["user_id", "product_id"]
    ]
    .drop_duplicates()
)

print(
    "Unique future customer-product purchases: "
    f"{len(future_pairs):,}"
)

print(
    "Unique customers with future purchases: "
    f"{future_pairs['user_id'].nunique():,}"
)


# ============================================================
# Step 3.3 — Create future-purchase lookup
# ============================================================

print("\nCreating future-purchase lookup...")

positive_pair_keys = set(
    zip(
        future_pairs["user_id"],
        future_pairs["product_id"]
    )
)

print(
    f"Future-purchase lookup size: "
    f"{len(positive_pair_keys):,}"
)


# ============================================================
# Step 3.4 — Load and label Task 2 feature data
# ============================================================

print("\nLoading and labeling Task 2 feature data...")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Task 2 feature file not found: {FEATURE_FILE}"
    )

labeled_chunks = []

total_rows = 0
positive_rows = 0

for chunk in pd.read_csv(
    FEATURE_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_keys = zip(
        chunk["user_id"],
        chunk["product_id"]
    )

    chunk["target"] = [
        int(key in positive_pair_keys)
        for key in chunk_keys
    ]

    total_rows += len(chunk)
    positive_rows += int(chunk["target"].sum())

    labeled_chunks.append(chunk)

    print(
        f"Processed: {total_rows:,} rows | "
        f"Positive: {positive_rows:,}"
    )


# ============================================================
# Combine labeled chunks
# ============================================================

labeled_data = pd.concat(
    labeled_chunks,
    ignore_index=True
)

print("\nLabeled dataset created.")

print(
    f"Total rows: "
    f"{len(labeled_data):,}"
)

print(
    f"Positive rows: "
    f"{labeled_data['target'].sum():,}"
)

print(
    f"Negative rows: "
    f"{(labeled_data['target'] == 0).sum():,}"
)


# ============================================================
# Step 3.5 — Separate features and target
# ============================================================

print("\nPreparing X and y...")

X = labeled_data.drop(
    columns=["target"]
)

y = labeled_data["target"]


# ============================================================
# Step 3.6 — Stratified train/test split
# ============================================================

print("\nCreating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# Step 3.7 — Check original train/test distributions
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST CLASS DISTRIBUTION")
print("=" * 70)


def print_distribution(name, target):
    counts = target.value_counts().sort_index()
    percentages = (
        target.value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(f"\n{name}")
    print("-" * 40)

    for class_value in counts.index:
        print(
            f"Class {class_value}: "
            f"{counts[class_value]:,} "
            f"({percentages[class_value]:.4f}%)"
        )


print_distribution(
    "Original dataset",
    y
)

print_distribution(
    "Training set before balancing",
    y_train
)

print_distribution(
    "Testing set",
    y_test
)


# ============================================================
# Step 3.8 — Handle Class Imbalance
# ============================================================

print("\n" + "=" * 70)
print("HANDLING CLASS IMBALANCE")
print("=" * 70)

train_data = X_train.copy()
train_data["target"] = y_train.values


# Separate majority and minority classes

positive_train = train_data[
    train_data["target"] == 1
]

negative_train = train_data[
    train_data["target"] == 0
]

print("\nBefore undersampling:")

print(
    f"Positive training samples: "
    f"{len(positive_train):,}"
)

print(
    f"Negative training samples: "
    f"{len(negative_train):,}"
)


# ============================================================
# Step 3.9 — Random undersampling
# ============================================================

negative_sampled = negative_train.sample(
    n=len(positive_train),
    random_state=RANDOM_STATE
)

balanced_train = pd.concat(
    [
        positive_train,
        negative_sampled
    ],
    ignore_index=True
)


# Shuffle the balanced training dataset

balanced_train = balanced_train.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# Separate balanced features and target

X_train_balanced = balanced_train.drop(
    columns=["target"]
)

y_train_balanced = balanced_train["target"]


# ============================================================
# Step 3.10 — Validate balanced training data
# ============================================================

print("\nAfter undersampling:")

print(
    f"Balanced training rows: "
    f"{len(X_train_balanced):,}"
)

print(
    f"Class 0: "
    f"{(y_train_balanced == 0).sum():,}"
)

print(
    f"Class 1: "
    f"{(y_train_balanced == 1).sum():,}"
)

print(
    f"Class 0 percentage: "
    f"{(y_train_balanced == 0).mean():.2%}"
)

print(
    f"Class 1 percentage: "
    f"{(y_train_balanced == 1).mean():.2%}"
)


# ============================================================
# Step 3.11 — Verify test set remains unchanged
# ============================================================

print("\nTest set remains unchanged:")

print(
    f"Test rows: "
    f"{len(X_test):,}"
)

print(
    f"Test Class 0: "
    f"{(y_test == 0).sum():,}"
)

print(
    f"Test Class 1: "
    f"{(y_test == 1).sum():,}"
)

print(
    f"Test Class 0 percentage: "
    f"{(y_test == 0).mean():.4%}"
)

print(
    f"Test Class 1 percentage: "
    f"{(y_test == 1).mean():.4%}"
)


# ============================================================
# Step 3.12 — Final summary
# ============================================================

print("\n" + "=" * 70)
print("TASK 3 PREPARATION SUMMARY")
print("=" * 70)

print(
    f"Original rows: "
    f"{len(labeled_data):,}"
)

print(
    f"Training rows before balancing: "
    f"{len(X_train):,}"
)

print(
    f"Balanced training rows: "
    f"{len(X_train_balanced):,}"
)

print(
    f"Testing rows: "
    f"{len(X_test):,}"
)

print(
    f"\nRandom state: "
    f"{RANDOM_STATE}"
)

print(
    f"Test size: "
    f"{TEST_SIZE:.0%}"
)

print("\nTask 3 split and imbalance handling completed.")


#---

# ============================================================
# Step 3.13 — Save Prepared Training and Testing Datasets
# ============================================================

print("\n" + "=" * 70)
print("SAVING PREPARED DATASETS")
print("=" * 70)

TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training_data"

# Create output directory if it does not exist
TRAINING_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = TRAINING_OUTPUT_DIR / "train.csv.gz"
TEST_FILE = TRAINING_OUTPUT_DIR / "test.csv.gz"


# ------------------------------------------------------------
# Prepare final training dataset
# ------------------------------------------------------------

train_output = X_train_balanced.copy()

train_output["target"] = y_train_balanced.values


# ------------------------------------------------------------
# Prepare final testing dataset
# ------------------------------------------------------------

test_output = X_test.copy()

test_output["target"] = y_test.values


# ------------------------------------------------------------
# Save training dataset
# ------------------------------------------------------------

print("\nSaving training dataset...")

train_output.to_csv(
    TRAIN_FILE,
    index=False,
    compression="gzip"
)

print(
    f"Training dataset saved to: "
    f"{TRAIN_FILE}"
)

print(
    f"Training shape: "
    f"{train_output.shape}"
)


# ------------------------------------------------------------
# Save testing dataset
# ------------------------------------------------------------

print("\nSaving testing dataset...")

test_output.to_csv(
    TEST_FILE,
    index=False,
    compression="gzip"
)

print(
    f"Testing dataset saved to: "
    f"{TEST_FILE}"
)

print(
    f"Testing shape: "
    f"{test_output.shape}"
)


# ============================================================
# Step 3.14 — Verify Saved Files
# ============================================================

print("\n" + "=" * 70)
print("SAVED DATASET VERIFICATION")
print("=" * 70)

print(
    f"\nTraining file exists: "
    f"{TRAIN_FILE.exists()}"
)

print(
    f"Testing file exists: "
    f"{TEST_FILE.exists()}"
)

if TRAIN_FILE.exists():
    print(
        f"Training file size: "
        f"{TRAIN_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )

if TEST_FILE.exists():
    print(
        f"Testing file size: "
        f"{TEST_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )


# ------------------------------------------------------------
# Final class distribution in saved datasets
# ------------------------------------------------------------

print("\nFinal training class distribution:")

print(
    train_output["target"]
    .value_counts()
    .sort_index()
)

print("\nFinal testing class distribution:")

print(
    test_output["target"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("TASK 3 DATASET CREATION COMPLETED")
print("=" * 70)