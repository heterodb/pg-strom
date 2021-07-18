---
--- test cases for GPU Cache
---

SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS gpu_cache_temp_test CASCADE;
CREATE SCHEMA gpu_cache_temp_test;
RESET client_min_messages;

SET search_path = gpu_cache_temp_test,public;

---
--- Creating a table on GPU cache
---

CREATE TABLE cache_test_table (
  id   int,
  a    int1,
  b    int2,
  c    int4,
  d    int8,
  e    float2,
  f    float4,
  g    float8,
  h    text,
  i    varchar(32),
  j    char(32)
);

---
--- GPU Cache configuration
---

CREATE TRIGGER row_sync_test AFTER INSERT OR UPDATE OR DELETE ON cache_test_table FOR ROW 
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('gpu_device_id=0,max_num_rows=5000,redo_buffer_size=150m,gpu_sync_threshold=10m,gpu_sync_interval=4');
CREATE TRIGGER stmt_sync_test AFTER TRUNCATE ON cache_test_table FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER row_sync_test;
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER stmt_sync_test;

-- Make GPU cache 
INSERT INTO cache_test_table(id) values (1);

-- Check gpucache_info table.
SELECT config_options FROM pgstrom.gpucache_info WHERE table_name='cache_test_table'
ORDER BY redo_write_ts DESC LIMIT 1;
TRUNCATE TABLE cache_test_table;

-- Force to use GPU Cache
SET enable_seqscan=off;

---
--- INSERT 
---

EXPLAIN (costs off, verbose)
INSERT INTO cache_test_table(id) values (1);
INSERT INTO cache_test_table (
  SELECT x 
  ,pgstrom.random_int(1,-128,127)     -- a int1
  ,pgstrom.random_int(1,-32768,32767)  -- b int2
  ,pgstrom.random_int(1,-2147483648,2147483647)  -- c int4
  ,pgstrom.random_int(1,-9223372036854775808,9223372036854775807)   -- d int8
  ,pgstrom.random_float(0.1, -32000, 32000)   -- e float2
  ,pgstrom.random_float(0.1, -999999, 999999)   -- f float4
  ,pgstrom.random_float(0.1, -999999, 999999)   -- f float8
  ,LEFT(MD5((x%479)::TEXT),(x%32+1)::INTEGER)     -- h text
  ,LEFT(MD5((x%479+1)::TEXT),(x%32+1)::INTEGER)     -- i text
  ,LEFT(MD5((x%479+2)::TEXT),(x%32+1)::INTEGER)     -- j text
  FROM generate_series(1,4000) x
);

VACUUM ANALYZE;


-- Copy inserted records to normal table(without GPU Cache)
SELECT * INTO TEMPORARY normal_table FROM cache_test_table; 

---
--- UPDATE
---
EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET a = (a+1) % 127 WHERE a%97=0;
UPDATE cache_test_table SET a = (a+1) % 127 WHERE a%97=0;
UPDATE normal_table     SET a = (a+1) % 127 WHERE a%97=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET b = (b+1) % 32767 WHERE a%89=0;
UPDATE cache_test_table SET b = (b+1) % 32767 WHERE a%89=0;
UPDATE normal_table     SET b = (b+1) % 32767 WHERE a%89=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET c = (c+1) % 2147483647 WHERE a%83=0;
UPDATE cache_test_table SET c = (c+1) % 2147483647 WHERE a%83=0;
UPDATE normal_table     SET c = (c+1) % 2147483647 WHERE a%83=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET d = (d+1) % 9223372036854775807 WHERE a%79=0;
UPDATE cache_test_table SET d = (d+1) % 9223372036854775807 WHERE a%79=0;
UPDATE normal_table     SET d = (d+1) % 9223372036854775807 WHERE a%79=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET e = (e+0.1)::numeric % 999999 WHERE a%73=0;
UPDATE cache_test_table SET e = (e+0.1)::numeric % 999999 WHERE a%73=0;
UPDATE normal_table     SET e = (e+0.1)::numeric % 999999 WHERE a%73=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET f = (f+0.1)::numeric % 999999 WHERE a%71=0;
UPDATE cache_test_table SET f = (f+0.1)::numeric % 999999 WHERE a%71=0;
UPDATE normal_table     SET f = (f+0.1)::numeric % 999999 WHERE a%71=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET g = (g+0.1)::numeric % 999999 WHERE a%67=0;
UPDATE cache_test_table SET g = (g+0.1)::numeric % 999999 WHERE a%67=0;
UPDATE normal_table     SET g = (g+0.1)::numeric % 999999 WHERE a%67=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET h = 'delete' WHERE a%103=0;
UPDATE cache_test_table SET h = 'delete' WHERE a%103=0;
UPDATE normal_table     SET h = 'delete' WHERE a%103=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET i = 'delete' WHERE a%107=0;
UPDATE cache_test_table SET i = 'delete' WHERE a%107=0;
UPDATE normal_table     SET i = 'delete' WHERE a%107=0;

EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET j = 'delete' WHERE a%109=0;
UPDATE cache_test_table SET j = 'delete' WHERE a%109=0;
UPDATE normal_table     SET j = 'delete' WHERE a%109=0;

---
--- DETELE
---
EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE a%101=0 OR a IS NULL;
DELETE FROM cache_test_table WHERE a%101=0 OR a IS NULL;
DELETE FROM normal_table WHERE a%101=0 OR a IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE b%101=0 OR b IS NULL;
DELETE FROM cache_test_table WHERE b%101=0 OR b IS NULL;
DELETE FROM normal_table WHERE b%101=0 OR b IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE c%101=0 OR c IS NULL;
DELETE FROM cache_test_table WHERE c%101=0 OR c IS NULL;
DELETE FROM normal_table WHERE c%101=0 OR c IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE d%101=0 OR d IS NULL;
DELETE FROM cache_test_table WHERE d%101=0 OR d IS NULL;
DELETE FROM normal_table WHERE d%101=0 OR d IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE e::int8%101=0 OR e IS NULL;
DELETE FROM cache_test_table WHERE e::int8%101=0 OR e IS NULL;
DELETE FROM normal_table WHERE e::int8%101=0 OR e IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE f::int8%101=0 OR f IS NULL;
DELETE FROM cache_test_table WHERE f::int8%101=0 OR f IS NULL;
DELETE FROM normal_table WHERE f::int8%101=0 OR f IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE g::int8%101=0 OR g IS NULL;
DELETE FROM cache_test_table WHERE g::int8%101=0 OR g IS NULL;
DELETE FROM normal_table WHERE g::int8%101=0 OR g IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE h='delete' OR h IS NULL;
DELETE FROM cache_test_table WHERE h='delete' OR h IS NULL;
DELETE FROM normal_table WHERE h='delete' OR h IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE i='delete' OR i IS NULL;
DELETE FROM cache_test_table WHERE i='delete' OR i IS NULL;
DELETE FROM normal_table WHERE i='delete' OR i IS NULL;

EXPLAIN (costs off, verbose) 
DELETE FROM cache_test_table WHERE j='delete' OR j IS NULL;
DELETE FROM cache_test_table WHERE j='delete' OR j IS NULL;
DELETE FROM normal_table WHERE j='delete' OR j IS NULL;


---
--- ALTER TABLE 
---

EXPLAIN (costs off, verbose) 
ALTER TABLE cache_test_table ADD COLUMN k int2;
ALTER TABLE cache_test_table ADD COLUMN k int2;
ALTER TABLE normal_table ADD COLUMN k int2;

UPDATE cache_test_table SET k=b/2;
UPDATE normal_table SET k=b/2;

---
--- SELECT 
---

EXPLAIN (costs off, verbose)
SELECT * FROM cache_test_table WHERE a % 3 = 0;

SELECT COUNT(*) = (SELECT COUNT(*) FROM normal_table WHERE a % 3 = 0) AS ok FROM (
SELECT * FROM normal_table WHERE a % 3 = 0
UNION DISTINCT
SELECT * FROM cache_test_table WHERE a % 3 = 0) AS q;


--clms=("a" "b" "c" "d" "e" "f" "g") ; for a in ${clms[@]}; do echo "COUNT($a) AS ${a}_count,ROUND(SUM($a)::NUMERIC,3) AS ${a}_sum,ROUND(AVG($a)::NUMERIC,3) AS ${a}_avg,MAX($a) AS ${a}_max,MIN($a) AS ${a}_min," ; done

EXPLAIN (costs off, verbose)
SELECT 
COUNT(a) AS a_count,ROUND(SUM(a)::NUMERIC,3) AS a_sum,ROUND(AVG(a)::NUMERIC,3) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,ROUND(SUM(b)::NUMERIC,3) AS b_sum,ROUND(AVG(b)::NUMERIC,3) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,ROUND(SUM(c)::NUMERIC,3) AS c_sum,ROUND(AVG(c)::NUMERIC,3) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,ROUND(SUM(d)::NUMERIC,3) AS d_sum,ROUND(AVG(d)::NUMERIC,3) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,ROUND(SUM(e)::NUMERIC,3) AS e_sum,ROUND(AVG(e)::NUMERIC,3) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,ROUND(SUM(f)::NUMERIC,3) AS f_sum,ROUND(AVG(f)::NUMERIC,3) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,ROUND(SUM(g)::NUMERIC,3) AS g_sum,ROUND(AVG(g)::NUMERIC,3) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY cached_result FROM cache_test_table where id%3=0;

SELECT 
COUNT(a) AS a_count,ROUND(SUM(a)::NUMERIC,3) AS a_sum,ROUND(AVG(a)::NUMERIC,3) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,ROUND(SUM(b)::NUMERIC,3) AS b_sum,ROUND(AVG(b)::NUMERIC,3) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,ROUND(SUM(c)::NUMERIC,3) AS c_sum,ROUND(AVG(c)::NUMERIC,3) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,ROUND(SUM(d)::NUMERIC,3) AS d_sum,ROUND(AVG(d)::NUMERIC,3) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,ROUND(SUM(e)::NUMERIC,3) AS e_sum,ROUND(AVG(e)::NUMERIC,3) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,ROUND(SUM(f)::NUMERIC,3) AS f_sum,ROUND(AVG(f)::NUMERIC,3) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,ROUND(SUM(g)::NUMERIC,3) AS g_sum,ROUND(AVG(g)::NUMERIC,3) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY cached_result FROM cache_test_table where id%3=0;

SET enable_seqscan=on;

SELECT
COUNT(a) AS a_count,ROUND(SUM(a)::NUMERIC,3) AS a_sum,ROUND(AVG(a)::NUMERIC,3) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,ROUND(SUM(b)::NUMERIC,3) AS b_sum,ROUND(AVG(b)::NUMERIC,3) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,ROUND(SUM(c)::NUMERIC,3) AS c_sum,ROUND(AVG(c)::NUMERIC,3) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,ROUND(SUM(d)::NUMERIC,3) AS d_sum,ROUND(AVG(d)::NUMERIC,3) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,ROUND(SUM(e)::NUMERIC,3) AS e_sum,ROUND(AVG(e)::NUMERIC,3) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,ROUND(SUM(f)::NUMERIC,3) AS f_sum,ROUND(AVG(f)::NUMERIC,3) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,ROUND(SUM(g)::NUMERIC,3) AS g_sum,ROUND(AVG(g)::NUMERIC,3) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY normal_result FROM normal_table where id%3=0;

select COUNT(*)=1 AS ok FROM (
  SELECT * FROM cached_result
  UNION DISTINCT
  SELECT * FROM normal_result
) AS m;

---
--- TRUNCATE
---

TRUNCATE TABLE cache_test_table;
TRUNCATE TABLE normal_table;

SELECT COUNT(*) = 0 AS ok FROM cache_test_table;
SELECT COUNT(*) = 0 AS ok FROM normal_table;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA gpu_cache_temp_test CASCADE;
