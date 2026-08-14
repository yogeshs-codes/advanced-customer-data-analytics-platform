-- ============================================================
-- Advanced Customer Data Analytics Platform
-- Task 2 - Snowflake Setup
-- ============================================================

-- 1. Create database
CREATE DATABASE IF NOT EXISTS CUSTOMER_DEMAND;

-- 2. Create feature engineering schema
CREATE SCHEMA IF NOT EXISTS
    CUSTOMER_DEMAND.FEATURE_ENGINEERING;

-- 3. Create warehouse
CREATE WAREHOUSE IF NOT EXISTS CUSTOMER_DEMAND_WH
WITH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- 4. Select warehouse
USE WAREHOUSE CUSTOMER_DEMAND_WH;

-- 5. Select database
USE DATABASE CUSTOMER_DEMAND;

-- 6. Select schema
USE SCHEMA FEATURE_ENGINEERING;


-- ============================================================
-- 7. Create engineered feature table
-- ============================================================

CREATE TABLE IF NOT EXISTS CUSTOMER_PRODUCT_FEATURES (
    USER_ID INTEGER,
    PRODUCT_ID INTEGER,

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
);


-- ============================================================
-- 8. Verify Snowflake objects
-- ============================================================

SHOW TABLES IN SCHEMA
    CUSTOMER_DEMAND.FEATURE_ENGINEERING;


-- ============================================================
-- 9. Validate loaded feature dataset
-- ============================================================

SELECT
    COUNT(*) AS ROW_COUNT,
    COUNT(DISTINCT USER_ID || '-' || PRODUCT_ID)
        AS UNIQUE_CUSTOMER_PRODUCT_PAIRS,
    COUNT_IF(USER_ID IS NULL) AS NULL_USER_IDS,
    COUNT_IF(PRODUCT_ID IS NULL) AS NULL_PRODUCT_IDS
FROM CUSTOMER_DEMAND.FEATURE_ENGINEERING.CUSTOMER_PRODUCT_FEATURES;