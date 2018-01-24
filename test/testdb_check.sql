\pset format unaligned
\pset tuples_only

SET pg_strom.enabled = off;

WITH check_extension AS (
  SELECT count(*) c_ext
    FROM pg_extension
   WHERE extname = 'pg_strom'
),
check_relations AS (
  SELECT count(*) c_rel
    FROM pg_class
   WHERE relnamespace = 'public'::regnamespace
     AND pg_relation_size(oid) > 100 * 8192	-- not empty
     AND relname in ('t0','t1','t2')
)
SELECT c_ext > 0 AND c_rel = 3
  FROM check_extension, check_relations;
