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
  a    int1 --,
  /*b    int2,
  c    int4,
  d    int8,
  e    float2,
  f    float4,
  g    float8,
  h    bit(3),
  i    bit varying(10),
  j    boolean,
  k    char(3),
  l    varchar(10),
  m    date,
  n    timestamp*/
);

---
--- GPU Cache configuration
---

CREATE TRIGGER row_sync_test AFTER INSERT OR UPDATE OR DELETE ON cache_test_table FOR ROW 
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('gpu_device_id=0,max_num_rows=3000,redo_buffer_size=150m,gpu_sync_threshold=10m,gpu_sync_interval=4');
CREATE TRIGGER stmt_sync_test AFTER TRUNCATE ON cache_test_table FOR STATEMENT
    EXECUTE FUNCTION pgstrom.gpucache_sync_trigger();
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER row_sync_test;
ALTER TABLE cache_test_table ENABLE ALWAYS TRIGGER row_stmt_test;

-- Make GPU cache 
INSERT INTO cache_test_table(id) SELECT 1 AS id;

-- Check gpucache_info table.
SELECT config_options FROM pgstrom.gpucache_info WHERE table_name='cache_test_table'
ORDER BY redo_write_ts DESC LIMIT 1;
TRUNCATE TABLE cache_test_table;

-- Force to use GPU Cache
SET enable_seqscan=off;

---
--- INSERT 
---

INSERT INTO cache_test_table (
  SELECT x, pgstrom.random_int(1,-128,127)
  FROM generate_series(1,2000) x
);

VACUUM ANALYZE;

-- Preparation of normal table
SELECT * INTO normal_table FROM cache_test_table; 

---
--- UPDATE
---
EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET 
  a = (a+1) % 127
WHERE id%100=0;

UPDATE cache_test_table SET 
  a = (a+1) % 127
WHERE id%100=0;

UPDATE normal_table SET 
  a = (a+1) % 127
WHERE id%100=0;

---
--- DETELE
---
EXPLAIN (costs off, verbose)
UPDATE cache_test_table SET 
  a = (a+1) % 127
WHERE id%100=0;

UPDATE cache_test_table SET 
  a = (a+1) % 127
WHERE id%100=0;

UPDATE normal_table SET 
  a = (a+1) % 127
WHERE id%100=0;

---
--- SELECT 
---

SELECT count(*) FROM normal_table AS n, cache_test_table AS c 
WHERE n.id = c.id 
AND (n.a <> c.a);


---
--- TRUNCATE
---


-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA gpu_cache_temp_test CASCADE;


