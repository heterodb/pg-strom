---
--- Regression test for data consistency between CSV, Parquet, and Arrow files
--- This test verifies that python_generated_data.csv, python_generated_data.parquet,
--- and python_generated_data.arrow contain identical data by importing all three 
--- and comparing their contents using EXCEPT.
---
\t on
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_python_data_temp CASCADE;
CREATE SCHEMA regtest_python_data_temp;
RESET client_min_messages;

SET search_path = regtest_python_data_temp,public;

--
-- Create table for CSV data import
--
CREATE TABLE python_csv_data (
    a_int32      INTEGER,
    b_int64      BIGINT,
    c_int96      TIMESTAMP,
    d_float      REAL,
    e_double     DOUBLE PRECISION,
    f_string     TEXT,
    g_enum       TEXT,
    h_decimal    NUMERIC(10,2),
    i_date       DATE,
    k_timestamp  TIMESTAMP
);

--
-- Set up data file paths using environment variables
--
\getenv python_data_dir ARROW_TEST_DATA_DIR
\set python_csv_file_path :python_data_dir '/python_generated_data.csv' 
\set python_parquet_file_path :python_data_dir '/python_generated_data.parquet'
\set python_arrow_file_path :python_data_dir '/python_generated_data.arrow'

--
-- Import CSV data using SQL COPY command
--
COPY python_csv_data FROM :'python_csv_file_path' WITH (FORMAT csv, HEADER true);

--
-- Create foreign table for Parquet data access via arrow_fdw
--
CREATE FOREIGN TABLE python_parquet_data (
    a_int32      INTEGER,
    b_int64      BIGINT,
    c_int96      TIMESTAMP,
    d_float      REAL,
    e_double     DOUBLE PRECISION,
    f_string     TEXT,
    g_enum       TEXT,
    h_decimal    NUMERIC(10,2),
    i_date       DATE,
    k_timestamp  TIMESTAMP
) SERVER arrow_fdw
OPTIONS (file :'python_parquet_file_path');

--
-- Create foreign table for Arrow data access via arrow_fdw
--
CREATE FOREIGN TABLE python_arrow_data (
    a_int32      INTEGER,
    b_int64      BIGINT,
    c_int96      TIMESTAMP,
    d_float      REAL,
    e_double     DOUBLE PRECISION,
    f_string     TEXT,
    g_enum       TEXT,
    h_decimal    NUMERIC(10,2),
    i_date       DATE,
    k_timestamp  TIMESTAMP
) SERVER arrow_fdw
OPTIONS (file :'python_arrow_file_path');

--
-- Main regression test: Data consistency check using EXCEPT
-- All queries should return empty results if datasets match the CSV baseline
--

-- CSV vs Parquet comparison
SELECT * FROM python_csv_data 
EXCEPT 
SELECT * FROM python_parquet_data;

SELECT * FROM python_parquet_data 
EXCEPT 
SELECT * FROM python_csv_data;

-- CSV vs Arrow comparison
SELECT * FROM python_csv_data 
EXCEPT 
SELECT * FROM python_arrow_data;

SELECT * FROM python_arrow_data 
EXCEPT 
SELECT * FROM python_csv_data;

-- Clean up
DROP SCHEMA regtest_python_data_temp CASCADE;