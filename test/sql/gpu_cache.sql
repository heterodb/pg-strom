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
CREATE TRIGGER stmt_sync_test AFTER TRUNCATE ON cache_test_table FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER row_sync_test;
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER stmt_sync_test;

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
  ,pgstrom.random_float(0.5, -32000, 32000)   -- f float8
  ,LEFT(MD5((x%479)::TEXT),(x%32+1)::INTEGER)     -- h text
  ,LEFT(MD5((x%479+1)::TEXT),(x%32+1)::INTEGER)     -- i text
  ,LEFT(MD5((x%479+2)::TEXT),(x%32+1)::INTEGER)     -- j text
  FROM generate_series(1,4000) x
);

ALTER TABLE cache_test_table ADD COLUMN k int2;
UPDATE cache_test_table SET k=b/2;

---
--- UPDATE
UPDATE cache_test_table SET a = (a+1) % 127 WHERE a%97=0;

UPDATE cache_test_table SET b = (b+1) % 32767 WHERE a%89=0;

UPDATE cache_test_table SET c = (c+1) % 2147483647 WHERE a%83=0;

UPDATE cache_test_table SET d = (d+1) % 9223372036854775807 WHERE a%79=0;

UPDATE cache_test_table SET e = (e+0.5)::numeric % 999999 WHERE a%73=0;

UPDATE cache_test_table SET f = (f+0.5)::numeric % 999999 WHERE a%71=0;

UPDATE cache_test_table SET g = (g+0.5)::numeric % 999999 WHERE a%67=0;

UPDATE cache_test_table SET h = 'delete' WHERE a%103=0;

UPDATE cache_test_table SET i = 'delete' WHERE a%107=0;

UPDATE cache_test_table SET j = 'delete' WHERE a%109=0;

---
--- DETELE
---
DELETE FROM cache_test_table WHERE a%101=0 OR a IS NULL;

DELETE FROM cache_test_table WHERE b%101=0 OR b IS NULL;

DELETE FROM cache_test_table WHERE c%101=0 OR c IS NULL;
DELETE FROM cache_test_table WHERE d%101=0 OR d IS NULL;

DELETE FROM cache_test_table WHERE e::int8%101=0 OR e IS NULL;

DELETE FROM cache_test_table WHERE f::int8%101=0 OR f IS NULL;

DELETE FROM cache_test_table WHERE g::int8%101=0 OR g IS NULL;

DELETE FROM cache_test_table WHERE h='delete' OR h IS NULL;

DELETE FROM cache_test_table WHERE i='delete' OR i IS NULL;

DELETE FROM cache_test_table WHERE j='delete' OR j IS NULL;

---
--- corruption_check
---
SET enable_seqscan=off;

select * from pgstrom.gpucache_info;

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

---
--- GPU Cache configuration
---
CREATE TRIGGER row_sync_test AFTER INSERT OR UPDATE OR DELETE ON cache_corruption_test FOR ROW 
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('gpu_device_id=0,max_num_rows=5000,redo_buffer_size=150m,gpu_sync_threshold=10m,gpu_sync_interval=4');
CREATE TRIGGER stmt_sync_test AFTER TRUNCATE ON cache_corruption_test FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_corruption_test ENABLE ALWAYS TRIGGER row_sync_test;
ALTER TABLE cache_corruption_test ENABLE ALWAYS TRIGGER stmt_sync_test;

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
SELECT pgstrom.gpucache_apply_redo('cache_corruption_test') AS apply_redo;    --２回に１回999CUDA_ERROR_UNKNOWNで落ちる。

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA gpu_cache_temp_test CASCADE;
