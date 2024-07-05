---
--- Test for GPU PostGIS Support
---
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_postgis_temp CASCADE;
CREATE SCHEMA regtest_postgis_temp;
SET search_path = regtest_postgis_temp,public;
\set test_giskanto_src_path `echo -n $ARROW_TEST_DATA_DIR/giskanto.sql`
\i :test_giskanto_src_path
RESET client_min_messages;

CREATE TABLE dpoints (
  did    int,
  x      float8,
  y      float8
);
SELECT pgstrom.random_setseed(20240527);
INSERT INTO dpoints (SELECT i, pgstrom.random_float(0.0, 138.787661, 140.434258),
                               pgstrom.random_float(0.0,  35.250012,  36.179917)
                       FROM generate_series(1,300000) i);
---
--- Run GPU Join with GiST index
---
RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT pref, city, count(*)
  FROM giskanto, dpoints
 WHERE (pref = '東京都' or city like '横浜市 %')
   AND st_contains(geom, st_makepoint(x, y))
 GROUP BY pref, city
 ORDER BY pref, city;

SELECT pref, city, count(*)
  FROM giskanto, dpoints
 WHERE (pref = '東京都' or city like '横浜市 %')
   AND st_contains(geom, st_makepoint(x, y))
 GROUP BY pref, city
 ORDER BY pref, city;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT pref, city, count(*)
  FROM giskanto, dpoints
 WHERE ((pref = '東京都' and city like '%区') OR
        (pref = '埼玉県' and city like '%市%'))
   AND st_dwithin(geom, st_makepoint(x, y), 0.002)
 GROUP BY pref, city
 ORDER BY pref, city;

SELECT pref, city, count(*)
  FROM giskanto, dpoints
 WHERE ((pref = '東京都' and city like '%区') OR
        (pref = '埼玉県' and city like '%市%'))
   AND st_dwithin(geom, st_makepoint(x, y), 0.002)
 GROUP BY pref, city
 ORDER BY pref, city;

---
--- PostGIS Functions
---
RESET pg_strom.enabled;

/*
--
-- st_distance POLY-POLY is wrong
--
EXPLAIN
SELECT a.pref, a.city, b.pref, b.city, st_distance(a.geom, b.geom)
  FROM giskanto a, giskanto b
 WHERE a.gid <> b.gid
   AND a.city in ('目黒区','所沢市', '杉並区','府中市')
   AND b.city in ('鴻巣市','蕨市','葛飾区','海老名市');

SELECT a.pref, a.city, b.pref, b.city, st_distance(a.geom, b.geom)
  FROM giskanto a, giskanto b
 WHERE a.gid <> b.gid
   AND a.city in ('目黒区','所沢市', '杉並区','府中市')
   AND b.city in ('鴻巣市','蕨市','葛飾区','海老名市');
*/

-- distance from POINT(皇居)
SET enable_seqscan = off;
RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT gid, pref, city,
       st_distance(geom, st_makepoint(139.7234394, 35.6851783)) dist
  INTO test01g
  FROM giskanto
 WHERE gid > 0;

SELECT gid, pref, city,
       st_distance(geom, st_makepoint(139.7234394, 35.6851783)) dist
  INTO test01g
  FROM giskanto
 WHERE gid > 0;

SET pg_strom.enabled = off;
SELECT gid, pref, city,
       st_distance(geom, st_makepoint(139.7234394, 35.6851783)) dist
  INTO test01c
  FROM giskanto
 WHERE gid > 0;

/*
SELECT *
  FROM test01g g, test01c c
 WHERE g.gid = c.gid
   AND abs(g.dist - c.dist) >= 0.00001;
*/

-- distance from POINT(筑波大学)
RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT gid, pref, city,
       st_distance(geom, st_makepoint(140.1070404, 36.094009)) dist
  INTO test02g
  FROM giskanto
 WHERE gid > 0;

SELECT gid, pref, city,
       st_distance(geom, st_makepoint(140.1070404, 36.094009)) dist
  INTO test02g
  FROM giskanto
 WHERE gid > 0;

SET pg_strom.enabled = off;
SELECT gid, pref, city,
       st_distance(geom, st_makepoint(140.1070404, 36.094009)) dist
  INTO test02c
  FROM giskanto
 WHERE gid > 0;

/*
SELECT *
  FROM test02g g, test02c c
 WHERE g.gid = c.gid
   AND abs(g.dist - c.dist) >= 0.00001;
*/
