---
--- Micro test cases for jsonb operators
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_jsonb_temp CASCADE;
CREATE SCHEMA regtest_dtype_jsonb_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_jsonb_temp, public;
CREATE TABLE rt_jsonb_a (
  id  int,
  v   jsonb
);
CREATE TABLE rt_jsonb_o (
  id  int,
  v   jsonb
);
CREATE TABLE rt_jsonb_c (
  id  int,
  v   jsonb
);
SELECT pgstrom.random_setseed(20190623);

INSERT INTO rt_jsonb_a (
  SELECT x, ('[ ' || case when i is null then 'null' else i::text end
          || ', ' || case when f is null then 'null' else f::text end
          || ', ' || case when b is null then 'null' else b end
          || ', ' || case when s_1 is null then 'null' else s_1 end
          || ', ' || case when s_2 is null then 'null' else s_2 end
          || ']')::jsonb
    FROM (SELECT x, pgstrom.random_int(2, -10000, 10000) i,
                    pgstrom.random_float(2,-10000.0, 10000.0)::numeric(9,3) f,
                    case when pgstrom.random_int(2,0,1000) < 500
                         then 'true'
                         else 'false' end b,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_1,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_2
            FROM generate_series(1,5000) x) AS foo);

INSERT INTO rt_jsonb_o (
  SELECT x, ('{ "ival" : '   ||
                 case when i   is null then 'null' else i::text end
          || ', "fval" : '   ||
                 case when f   is null then 'null' else f::text end
          || ', "bval" : '   ||
                 case when b   is null then 'null' else b end
          || ', "sval_1" : ' ||
                 case when s_1 is null then 'null' else s_1 end
          || ', "sval_2" : ' ||
                 case when s_2 is null then 'null' else s_2 end
          || '}')::jsonb
    FROM (SELECT x, pgstrom.random_int(2, -10000, 10000) i,
                    pgstrom.random_float(2,-10000.0, 10000.0)::numeric(9,3) f,
                    case when pgstrom.random_int(2,0,1000) < 500
                         then 'true'
                         else 'false' end b,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_1,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_2
            FROM generate_series(1,5000) x) AS foo);
-- "sval_1" is missing
INSERT INTO rt_jsonb_o (
  SELECT x, ('{ "ival" : '   ||
                 case when i   is null then 'null' else i::text end
          || ', "fval" : '   ||
                 case when f   is null then 'null' else f::text end
          || ', "bval" : '   ||
                 case when b   is null then 'null' else b end
          || ', "sval_2" : ' ||
                 case when s_2 is null then 'null' else s_2 end
          || '}')::jsonb
    FROM (SELECT x, pgstrom.random_int(2, -10000, 10000) i,
                    pgstrom.random_float(2,-10000.0, 10000.0)::numeric(9,3) f,
                    case when pgstrom.random_int(2,0,1000) < 500
                         then 'true'
                         else 'false' end b,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_2
            FROM generate_series(5001,6000) x) AS foo);

-- add nested jsonb
INSERT INTO rt_jsonb_c (
  SELECT x, ('{ "num1" : ' ||
                case when i is null then 'null' else i::text end
          || ', "num2" : ' ||
                case when f is null then 'null' else f::text end
          || ', "str1" : ' ||
                case when s_1 is null then 'null' else s_1 end
          || ', "str2" : ' ||
                case when s_2 is null then 'null' else s_2 end
          || ', "comp" : ' || rt_jsonb_o.v::text
          || '}')::jsonb
    FROM (SELECT x, pgstrom.random_int(2, -10000, 10000) i,
                    pgstrom.random_float(2,-10000.0, 10000.0)::numeric(9,3) f,
                    case when pgstrom.random_int(2,0,1000) < 500
                         then 'true'
                         else 'false' end b,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_1,
                    '"' || pgstrom.random_text_len(2,80) || '"' s_2
            FROM generate_series(1,6000) x) AS foo, rt_jsonb_o
   WHERE foo.x = rt_jsonb_o.id);

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- Fetch items from array-jsonb
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, v->0 ival, v->1 fval, v->2 bval, v->3 sval_1, v->4 sval_2
  INTO test01g
  FROM rt_jsonb_a
 WHERE id > 0;
SELECT id, v->0 ival, v->1 fval, v->2 bval, v->3 sval_1, v->4 sval_2
  INTO test01g
  FROM rt_jsonb_a
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, v->0 ival, v->1 fval, v->2 bval, v->3 sval_1, v->4 sval_2
  INTO test01p
  FROM rt_jsonb_a
 WHERE id > 0;
(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, v->>0 ival, v->>1 fval, v->>2 bval, v->>3 sval_1, v->>4 sval_2
  INTO test02g
  FROM rt_jsonb_a
 WHERE id > 0;
SELECT id, v->>0 ival, v->>1 fval, v->>2 bval, v->>3 sval_1, v->>4 sval_2
  INTO test02g
  FROM rt_jsonb_a
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, v->>0 ival, v->>1 fval, v->>2 bval, v->>3 sval_1, v->>4 sval_2
  INTO test02p
  FROM rt_jsonb_a
 WHERE id > 0;
(SELECT * FROM test02g EXCEPT SELECT * FROM test02p) ORDER BY id;
(SELECT * FROM test02p EXCEPT SELECT * FROM test02g) ORDER BY id;

-- Fetch items from key-value jsonb
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, (v)->'ival' ival, (v)->'fval' fval, (v)->'bval',
           (v)->'sval_1' sval_1, (v)->'sval_2' sval_2
  INTO test03g
  FROM rt_jsonb_o
 WHERE id > 0;
SELECT id, (v)->'ival' ival, (v)->'fval' fval, (v)->'bval',
           (v)->'sval_1' sval_1, (v)->'sval_2' sval_2
  INTO test03g
  FROM rt_jsonb_o
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, (v)->'ival' ival, (v)->'fval' fval, (v)->'bval',
           (v)->'sval_1' sval_1, (v)->'sval_2' sval_2
  INTO test03p
  FROM rt_jsonb_o
 WHERE id > 0;
(SELECT * FROM test03g EXCEPT SELECT * FROM test03p) ORDER BY id;
(SELECT * FROM test03p EXCEPT SELECT * FROM test03g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, (v)->>'ival' ival, (v)->>'fval' fval, (v)->>'bval' bval,
           (v)->>'sval_1' sval_1, (v)->>'sval_2' sval_2
  INTO test04g
  FROM rt_jsonb_o
 WHERE id > 0;
SELECT id, (v)->>'ival' ival, (v)->>'fval' fval, (v)->>'bval' bval,
           (v)->>'sval_1' sval_1, (v)->>'sval_2' sval_2
  INTO test04g
  FROM rt_jsonb_o
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, (v)->>'ival' ival, (v)->>'fval' fval, (v)->>'bval' bval,
           (v)->>'sval_1' sval_1, (v)->>'sval_2' sval_2
  INTO test04p
  FROM rt_jsonb_o
 WHERE id > 0;
(SELECT * FROM test04g EXCEPT SELECT * FROM test04p) ORDER BY id;
(SELECT * FROM test04p EXCEPT SELECT * FROM test04g) ORDER BY id;

-- Fetch items from nested-jsonb
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, (v)->'comp' comp
  INTO test05g
  FROM rt_jsonb_c
 WHERE id > 0;
SELECT id, (v)->'comp' comp
  INTO test05g
  FROM rt_jsonb_c
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, (v)->'comp' comp
  INTO test05p
  FROM rt_jsonb_c
 WHERE id > 0;
(SELECT * FROM test05g EXCEPT SELECT * FROM test05p) ORDER BY id;
(SELECT * FROM test05p EXCEPT SELECT * FROM test05g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ((v)->>'num1')::numeric + ((v)->'comp'->>'ival')::numeric v1,
           ((v)->>'num2')::numeric + ((v)->'comp'->>'fval')::numeric v2,
           substring((v)->>'str1', 2, 16) ||
           substring((v)->'comp'->>'sval_1', 2, 16) v3,
           substring((v)->'comp'->>'sval_2', 2, 16) v4
  INTO test06g
  FROM rt_jsonb_o
 WHERE id > 0;
SELECT id, ((v)->>'num1')::numeric + ((v)->'comp'->>'ival')::numeric v1,
           ((v)->>'num2')::numeric + ((v)->'comp'->>'fval')::numeric v2,
           substring((v)->>'str1', 2, 16) ||
           substring((v)->'comp'->>'sval_1', 2, 16) v3,
           substring((v)->'comp'->>'sval_2', 2, 16) v4
  INTO test06g
  FROM rt_jsonb_o
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ((v)->>'num1')::numeric + ((v)->'comp'->>'ival')::numeric v1,
           ((v)->>'num2')::numeric + ((v)->'comp'->>'fval')::numeric v2,
           substring((v)->>'str1', 2, 16) ||
           substring((v)->'comp'->>'sval_1', 2, 16) v3,
           substring((v)->'comp'->>'sval_2', 2, 16) v4
  INTO test06p
  FROM rt_jsonb_o
 WHERE id > 0;

(SELECT * FROM test06g EXCEPT SELECT * FROM test06p) ORDER BY id;
(SELECT * FROM test06p EXCEPT SELECT * FROM test06g) ORDER BY id;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dtype_jsonb_temp CASCADE;

