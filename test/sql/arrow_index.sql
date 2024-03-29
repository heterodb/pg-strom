---
--- Test for apache_arrow with index
---
-- \t on
-- SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_arrow_index_temp CASCADE;
CREATE SCHEMA regtest_arrow_index_temp;
RESET client_min_messages;

SET search_path = regtest_arrow_index_temp,public;

-- Prepare data for arrow index test
SELECT pgstrom.random_setseed(20210905);
CREATE TABLE arrow_index_data (
  id         int,
  int_num         int,
  float_num       float,
  half_num        float2,
  decimal_num     numeric,
  date_num        date,
  time_num        time,
  timestamp_num   timestamp
);

INSERT INTO arrow_index_data (
    SELECT x,   -- int_num
        pgstrom.random_int(0, -16777216, 16777216),   -- int_num
        pgstrom.random_float(0,-10000.0,10000.0),        -- float_num
        pgstrom.random_float(0,-10000.0,10000.0),        -- half_num
        pgstrom.random_float(0,-10000.0,10000.0),        -- decimal_num
        pgstrom.random_date(0),            -- date_num
        pgstrom.random_time(0),              -- time_num
        pgstrom.random_timestamp(0)                     -- date_num
    FROM generate_series(1,1000000) x);

-- ORDER BY date_num

-- Pick a record to search.
CREATE TABLE target_num AS SELECT * FROM arrow_index_data WHERE id=500000;

\getenv arrow_test_data_dir_path ARROW_TEST_DATA_DIR
\set test_arrow_index_path :arrow_test_data_dir_path '/test_arrow_index.data'
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY date_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=date_num
IMPORT FOREIGN SCHEMA regtest_arrow
  FROM SERVER arrow_fdw
  INTO regtest_arrow_index_temp
OPTIONS (file :'test_arrow_index_path');

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE date_num=(SELECT date_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE date_num between '2019-04-14' and '2023-05-23';

SELECT count(*)
  FROM regtest_arrow
 WHERE date_num between '2019-04-14' and '2023-05-23';

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE date_num between '2019-04-14' and '2023-05-23';
RESET pg_strom.enabled;

-- ORDER BY int_num

\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY int_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=int_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE int_num=(SELECT int_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE int_num between -50000 and 50000;

SELECT count(*)
  FROM regtest_arrow
 WHERE int_num between -50000 and 50000;

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE int_num between -50000 and 50000;
RESET pg_strom.enabled;

-- ORDER BY float
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY float_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=float_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE float_num=(SELECT float_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE float_num between -2000 and 2000;

SELECT count(*)
  FROM regtest_arrow
 WHERE float_num between -2000 and 2000;

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE float_num between -2000 and 2000;
RESET pg_strom.enabled;

-- ORDER BY half
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY half_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=half_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE half_num=(SELECT half_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE half_num between -2000 and 2000;

SELECT count(*)
  FROM regtest_arrow
 WHERE half_num between -2000 and 2000;

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE half_num between -2000 and 2000;
RESET pg_strom.enabled;

-- ORDER BY decimal
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY decimal_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=decimal_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE decimal_num=(SELECT decimal_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE decimal_num between -5000 and 5000;

SELECT count(*)
  FROM regtest_arrow
 WHERE decimal_num between -5000 and 5000;

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE decimal_num between -5000 and 5000;
RESET pg_strom.enabled;

-- ORDER BY time
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY time_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=time_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
 WHERE time_num=(SELECT time_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE time_num between '09:00:00' and '17:00:00';

SELECT count(*)
  FROM regtest_arrow
 WHERE time_num between '09:00:00' and '17:00:00';

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE time_num between '09:00:00' and '17:00:00';
RESET pg_strom.enabled;

-- ORDER BY timestamp
\! $PG2ARROW_CMD -s 16m --set=timezone:Asia/Tokyo -c 'SELECT * FROM regtest_arrow_index_temp.arrow_index_data ORDER BY timestamp_num' -o $ARROW_TEST_DATA_DIR/test_arrow_index.data --stat=timestamp_num
EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT * FROM regtest_arrow_index_temp.regtest_arrow
WHERE timestamp_num=(SELECT timestamp_num FROM regtest_arrow_index_temp.target_num);

EXPLAIN (ANALYZE,COSTS OFF,TIMING OFF,SUMMARY OFF)
SELECT count(*)
  FROM regtest_arrow
 WHERE timestamp_num between '2019-04-14 09:00:00' and '2023-05-23 17:00:00';

SELECT count(*)
  FROM regtest_arrow
 WHERE timestamp_num between '2019-04-14 09:00:00' and '2023-05-23 17:00:00';

SET pg_strom.enabled = off;
SELECT count(*)
  FROM arrow_index_data
 WHERE timestamp_num between '2019-04-14 09:00:00' and '2023-05-23 17:00:00';
RESET pg_strom.enabled;

DROP SCHEMA regtest_arrow_index_temp CASCADE;