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
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('gpu_device_id=0,max_num_rows=10000,redo_buffer_size=150m,gpu_sync_threshold=10m,gpu_sync_interval=4');
CREATE TRIGGER stmt_sync_test BEFORE TRUNCATE ON cache_test_table FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER row_sync_test;
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER stmt_sync_test;
-- Make GPU cache 
INSERT INTO cache_test_table(id) values (1);
-- Check gpucache_info table.
SELECT config_options FROM pgstrom.gpucache_info WHERE table_name='cache_test_table' AND database_name=current_database();

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
  ,pgstrom.random_float(0.5, -32000, 32000)   -- e float2
  ,pgstrom.random_float(0.5, -32000, 32000)   -- f float4
  ,pgstrom.random_float(0.5, -32000, 32000)   -- g float8
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
UPDATE cache_test_table SET e = (e+0.5)::numeric % 999999 WHERE a%73=0;


UPDATE cache_test_table SET e = (e+0.5)::numeric % 999999 WHERE a%73=0;
UPDATE normal_table     SET e = (e+0.5)::numeric % 999999 WHERE a%73=0;
EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET f = (f+0.5)::numeric % 999999 WHERE a%71=0;


UPDATE cache_test_table SET f = (f+0.5)::numeric % 999999 WHERE a%71=0;
UPDATE normal_table     SET f = (f+0.5)::numeric % 999999 WHERE a%71=0;
EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET g = (g+0.5)::numeric % 999999 WHERE a%67=0;


UPDATE cache_test_table SET g = (g+0.5)::numeric % 999999 WHERE a%67=0;
UPDATE normal_table     SET g = (g+0.5)::numeric % 999999 WHERE a%67=0;
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
ALTER TABLE cache_test_table ADD COLUMN k int2;
ALTER TABLE normal_table ADD COLUMN k int2;
UPDATE cache_test_table SET k=b/2;
UPDATE normal_table SET k=b/2;
---
--- SELECT 
---
EXPLAIN (costs off, verbose)
SELECT * FROM cache_test_table WHERE a % 3 = 0;

--clms=("a" "b" "c" "d" "e" "f" "g") ; for a in ${clms[@]}; do echo "COUNT($a) AS ${a}_count,ROUND(SUM($a)::NUMERIC/100,2) AS ${a}_sum,ROUND(AVG($a)::NUMERIC/100,2) AS ${a}_avg,MAX($a) AS ${a}_max,MIN($a) AS ${a}_min," ; done
EXPLAIN (costs off, verbose)
SELECT 
COUNT(a) AS a_count,SUM(a) AS a_sum,AVG(a) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,SUM(b) AS b_sum,AVG(b) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,SUM(c) AS c_sum,AVG(c) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,SUM(d) AS d_sum,AVG(d) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,SUM(e) AS e_sum,AVG(e) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,SUM(f) AS f_sum,AVG(f) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,SUM(g) AS g_sum,AVG(g) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY cached_result FROM cache_test_table WHERE id%3=0;

SELECT 
COUNT(a) AS a_count,SUM(a) AS a_sum,AVG(a) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,SUM(b) AS b_sum,AVG(b) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,SUM(c) AS c_sum,AVG(c) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,SUM(d) AS d_sum,AVG(d) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,SUM(e) AS e_sum,AVG(e) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,SUM(f) AS f_sum,AVG(f) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,SUM(g) AS g_sum,AVG(g) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY cached_result FROM cache_test_table WHERE id%3=0;
SELECT
COUNT(a) AS a_count,SUM(a) AS a_sum,AVG(a) AS a_avg,MAX(a) AS a_max,MIN(a) AS a_min,
COUNT(b) AS b_count,SUM(b) AS b_sum,AVG(b) AS b_avg,MAX(b) AS b_max,MIN(b) AS b_min,
COUNT(c) AS c_count,SUM(c) AS c_sum,AVG(c) AS c_avg,MAX(c) AS c_max,MIN(c) AS c_min,
COUNT(d) AS d_count,SUM(d) AS d_sum,AVG(d) AS d_avg,MAX(d) AS d_max,MIN(d) AS d_min,
COUNT(e) AS e_count,SUM(e) AS e_sum,AVG(e) AS e_avg,MAX(e) AS e_max,MIN(e) AS e_min,
COUNT(f) AS f_count,SUM(f) AS f_sum,AVG(f) AS f_avg,MAX(f) AS f_max,MIN(f) AS f_min,
COUNT(g) AS g_count,SUM(g) AS g_sum,AVG(g) AS g_avg,MAX(g) AS g_max,MIN(g) AS g_min
INTO TEMPORARY normal_result FROM normal_table WHERE id%3=0;
--clms=("a" "b" "c" "d" "e" "f" "g") ; mtds=("count" "sum" "avg" "max" "min"); for a in ${clms[@]}; do for m in ${mtds[@]}; do echo ",ABS(c.${a}_${m} - n.${a}_${m}) < 1 AS ${a}_${m}_ok" ; done ; done;
SELECT 
ABS(c.a_count - n.a_count) < 1 AS a_count_ok
,ABS(c.a_sum - n.a_sum) < 1 AS a_sum_ok
,ABS(c.a_avg - n.a_avg) < 1 AS a_avg_ok
,ABS(c.a_max - n.a_max) < 1 AS a_max_ok
,ABS(c.a_min - n.a_min) < 1 AS a_min_ok
,ABS(c.b_count - n.b_count) < 1 AS b_count_ok
,ABS(c.b_sum - n.b_sum) < 1 AS b_sum_ok
,ABS(c.b_avg - n.b_avg) < 1 AS b_avg_ok
,ABS(c.b_max - n.b_max) < 1 AS b_max_ok
,ABS(c.b_min - n.b_min) < 1 AS b_min_ok
,ABS(c.c_count - n.c_count) < 1 AS c_count_ok
,ABS(c.c_sum - n.c_sum) < 1 AS c_sum_ok
,ABS(c.c_avg - n.c_avg) < 1 AS c_avg_ok
,ABS(c.c_max - n.c_max) < 1 AS c_max_ok
,ABS(c.c_min - n.c_min) < 1 AS c_min_ok
,ABS(c.d_count - n.d_count) < 1 AS d_count_ok
,ABS(c.d_sum - n.d_sum) < 1 AS d_sum_ok
,ABS(c.d_avg - n.d_avg) < 1 AS d_avg_ok
,ABS(c.d_max - n.d_max) < 1 AS d_max_ok
,ABS(c.d_min - n.d_min) < 1 AS d_min_ok
,ABS(c.e_count - n.e_count) < 1 AS e_count_ok
,ABS(c.e_sum - n.e_sum)::float8/ABS(n.e_sum) < 0.00001 AS e_sum_ok
,ABS(c.e_avg - n.e_avg)::float8/ABS(c.e_avg) < 0.00001 AS e_avg_ok
,ABS(c.e_max - n.e_max) < 1 AS e_max_ok
,ABS(c.e_min - n.e_min) < 1 AS e_min_ok
,ABS(c.f_count - n.f_count) < 1 AS f_count_ok
,ABS(c.f_sum - n.f_sum)::float8/ABS(n.f_sum) < 0.00001 AS f_sum_ok
,ABS(c.f_avg - n.f_avg)::float8/ABS(c.f_avg) < 0.00001 AS f_avg_ok
,ABS(c.f_max - n.f_max) < 1 AS f_max_ok
,ABS(c.f_min - n.f_min) < 1 AS f_min_ok
,ABS(c.g_count - n.g_count) < 1 AS g_count_ok
,ABS(c.g_sum - n.g_sum)::float8/ABS(n.g_sum) < 0.00001 AS g_sum_ok
,ABS(c.g_avg - n.g_avg)::float8/ABS(c.g_avg) < 0.00001 AS g_avg_ok
,ABS(c.g_max - n.g_max) < 1 AS g_max_ok
,ABS(c.g_min - n.g_min) < 1 AS g_min_ok
 FROM cached_result AS c,
normal_result AS n
WHERE c.a_count = n.a_count;

---
--- TRUNCATE
---
TRUNCATE TABLE cache_test_table;
TRUNCATE TABLE normal_table;
SELECT COUNT(*) = 0 AS ok FROM cache_test_table;
SELECT COUNT(*) = 0 AS ok FROM normal_table;
---
--- corruption_check
---

CREATE TABLE cache_corruption_test (
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

CREATE TRIGGER row_sync_corruption AFTER INSERT OR UPDATE OR DELETE ON cache_corruption_test FOR ROW 
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('gpu_device_id=0,max_num_rows=5000,redo_buffer_size=150m,gpu_sync_threshold=10m,gpu_sync_interval=4');
CREATE TRIGGER stmt_sync_corruption BEFORE TRUNCATE ON cache_corruption_test FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_corruption_test ENABLE ALWAYS TRIGGER row_sync_corruption;
ALTER TABLE cache_corruption_test ENABLE ALWAYS TRIGGER stmt_sync_corruption;

-- INSERT 4000 rows ( < max: 5000 rows )
INSERT INTO cache_corruption_test (
  SELECT x 
  ,pgstrom.random_int(1,-128,127)     -- a int1
  ,pgstrom.random_int(1,-32768,32767)  -- b int2
  ,pgstrom.random_int(1,-2147483648,2147483647)  -- c int4
  ,pgstrom.random_int(1,-9223372036854775808,9223372036854775807)   -- d int8
  ,pgstrom.random_float(0.5, -32000, 32000)   -- e float2
  ,pgstrom.random_float(0.5, -32000, 32000)   -- f float4
  ,pgstrom.random_float(0.5, -32000, 32000)   -- f float8
  ,LEFT(MD5((x%479)::TEXT),(x%32+1)::INTEGER)     -- h text
  ,LEFT(MD5((x%479+1)::TEXT),(x%32+1)::INTEGER)     -- i text
  ,LEFT(MD5((x%479+2)::TEXT),(x%32+1)::INTEGER)     -- j text
  FROM generate_series(1,4000) x
);

-- Apply to GPUCache
SELECT pgstrom.gpucache_apply_redo('cache_corruption_test') = 0 AS apply_redo_result;

-- corrupted must be false
SELECT corrupted IS false FROM pgstrom.gpucache_info WHERE table_name='cache_corruption_test';

-- check GPUCache is still usable.
EXPLAIN (costs off, verbose)
SELECT * FROM cache_corruption_test WHERE b%3=0;

-- update to make the table corrupted
UPDATE cache_corruption_test SET d=d/2 WHERE id in (SELECT id FROM cache_corruption_test LIMIT 1000);
-- Apply to GPUCache (returns error because the table is corrupted)
SELECT pgstrom.gpucache_apply_redo('cache_corruption_test') > 0 AS apply_redo_result;

-- corrupted must be true
SELECT corrupted IS true FROM pgstrom.gpucache_info WHERE table_name='cache_corruption_test';

-- check GPUCache is not usable.
EXPLAIN (costs off, verbose)
SELECT * FROM cache_corruption_test WHERE b%3=0;

-- Recover GPUCache
SELECT pgstrom.gpucache_recovery('cache_corruption_test') = 0;

-- corrupted must be false
SELECT corrupted IS false FROM pgstrom.gpucache_info WHERE table_name='cache_corruption_test';

EXPLAIN (costs off, verbose)
SELECT count(*) FROM cache_corruption_test;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA gpu_cache_temp_test CASCADE;

-- Checking GPUCache is removed correctly.
SELECT count(*) = 0 AS ok FROM pgstrom.gpucache_info WHERE table_name in ('cache_test_table','cache_corruption_test') AND database_name=current_database();
