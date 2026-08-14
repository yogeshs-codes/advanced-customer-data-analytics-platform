-- ============================================================
-- Advanced Customer Data Analytics Platform
-- Task 2 - Feature Engineering / Snowflake Setup
-- ============================================================

-- ============================================================
-- 1. Create database, schema and warehouse
-- ============================================================

CREATE DATABASE IF NOT EXISTS CUSTOMER_DEMAND;

CREATE SCHEMA IF NOT EXISTS
CUSTOMER_DEMAND.FEATURE_ENGINEERING;

CREATE WAREHOUSE IF NOT EXISTS CUSTOMER_DEMAND_WH
WITH
WAREHOUSE_SIZE = 'XSMALL'
AUTO_SUSPEND = 60
AUTO_RESUME = TRUE
INITIALLY_SUSPENDED = TRUE;

USE WAREHOUSE CUSTOMER_DEMAND_WH;

USE DATABASE CUSTOMER_DEMAND;

USE SCHEMA FEATURE_ENGINEERING;

-- ============================================================
-- 2. Create engineered feature table
-- ============================================================

CREATE TABLE IF NOT EXISTS CUSTOMER_PRODUCT_FEATURES (
USER_ID INTEGER,
PRODUCT_ID INTEGER,

```
USER_PRODUCT_PURCHASE_COUNT INTEGER,
USER_PRODUCT_REORDER_COUNT INTEGER,
USER_PRODUCT_LAST_ORDER_NUMBER INTEGER,
USER_PRODUCT_REORDER_RATE FLOAT,
USER_PRODUCT_AVG_CART_POSITION FLOAT,
USER_PRODUCT_RECENCY_ORDERS INTEGER,

DEPARTMENT_ID INTEGER,
USER_DEPARTMENT_PURCHASE_COUNT INTEGER,
USER_DEPARTMENT_PURCHASE_SHARE FLOAT,

AISLE_ID INTEGER,
USER_AISLE_PURCHASE_COUNT INTEGER,
USER_AISLE_PURCHASE_SHARE FLOAT,

USER_TOTAL_ORDERS INTEGER,
USER_AVG_DAYS_BETWEEN_ORDERS FLOAT,
USER_AVG_ORDER_HOUR FLOAT,
USER_AVG_ORDER_DOW FLOAT,

PRODUCT_TOTAL_PURCHASES INTEGER,
PRODUCT_UNIQUE_USERS INTEGER,
PRODUCT_REORDER_RATE FLOAT
```

);

-- ============================================================
-- 3. Create an internal stage for the engineered dataset
-- ============================================================

CREATE STAGE IF NOT EXISTS
CUSTOMER_PRODUCT_FEATURES_STAGE;

-- ============================================================
-- 4. Upload the generated feature dataset
-- ============================================================

-- IMPORTANT:
-- The PUT command is executed from SnowSQL / Snowflake CLI,
-- not from the standard Snowsight SQL worksheet.
-------------------------------------------------

## -- Run the Python pipeline first:

## --     python src/feature_pipeline.py

## -- It creates:

## --     output/customer_product_features.csv.gz

## -- Then execute the following command from SnowSQL.

## -- Replace <PROJECT_PATH> with the absolute path to your project.

## -- Example:

-- PUT 'file:///D:/PROJECTS/customer_demand_analysis/output/customer_product_features.csv.gz'
--     @CUSTOMER_PRODUCT_FEATURES_STAGE
--     AUTO_COMPRESS=FALSE
--     OVERWRITE=TRUE;

-- ============================================================
-- 5. Load the engineered dataset into Snowflake
-- ============================================================

-- After the PUT command has uploaded the file, run:

COPY INTO CUSTOMER_PRODUCT_FEATURES
FROM @CUSTOMER_PRODUCT_FEATURES_STAGE
FILE_FORMAT = (
TYPE = CSV
SKIP_HEADER = 1
FIELD_OPTIONALLY_ENCLOSED_BY = '"'
COMPRESSION = GZIP
)
ON_ERROR = 'ABORT_STATEMENT';

-- ============================================================
-- 6. Verify Snowflake objects
-- ============================================================

SHOW TABLES IN SCHEMA
CUSTOMER_DEMAND.FEATURE_ENGINEERING;

-- ============================================================
-- 7. Validate loaded feature dataset
-- ============================================================

SELECT
COUNT(*) AS ROW_COUNT,
COUNT(DISTINCT USER_ID || '-' || PRODUCT_ID)
AS UNIQUE_CUSTOMER_PRODUCT_PAIRS,
COUNT_IF(USER_ID IS NULL) AS NULL_USER_IDS,
COUNT_IF(PRODUCT_ID IS NULL) AS NULL_PRODUCT_IDS
FROM CUSTOMER_DEMAND.FEATURE_ENGINEERING.CUSTOMER_PRODUCT_FEATURES;

-- ============================================================
-- 8. Feature quality checks
-- ============================================================

SELECT
COUNT(*) AS TOTAL_ROWS,
COUNT_IF(USER_PRODUCT_PURCHASE_COUNT < 0)
AS NEGATIVE_PURCHASE_COUNTS,
COUNT_IF(USER_PRODUCT_REORDER_COUNT < 0)
AS NEGATIVE_REORDER_COUNTS,
COUNT_IF(USER_PRODUCT_REORDER_RATE < 0
OR USER_PRODUCT_REORDER_RATE > 1)
AS INVALID_REORDER_RATES,
COUNT_IF(USER_DEPARTMENT_PURCHASE_SHARE < 0
OR USER_DEPARTMENT_PURCHASE_SHARE > 1)
AS INVALID_DEPARTMENT_SHARES,
COUNT_IF(USER_AISLE_PURCHASE_SHARE < 0
OR USER_AISLE_PURCHASE_SHARE > 1)
AS INVALID_AISLE_SHARES,
COUNT_IF(PRODUCT_REORDER_RATE < 0
OR PRODUCT_REORDER_RATE > 1)
AS INVALID_PRODUCT_REORDER_RATES
FROM CUSTOMER_DEMAND.FEATURE_ENGINEERING.CUSTOMER_PRODUCT_FEATURES;

-- ============================================================
-- 9. Inspect sample records
-- ============================================================

SELECT *
FROM CUSTOMER_DEMAND.FEATURE_ENGINEERING.CUSTOMER_PRODUCT_FEATURES
LIMIT 20;
