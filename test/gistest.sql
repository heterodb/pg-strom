DROP TABLE IF EXISTS _gistest CASCADE;

CREATE TABLE _gistest
(
  id  serial,
  a   geometry,
  b   geometry
);

CREATE OR REPLACE FUNCTION _st_relate(geometry,geometry)
RETURNS text
AS '$libdir/postgis-3','relate_full'
LANGUAGE C
STRICT
IMMUTABLE
PARALLEL SAFE
COST 10000;

CREATE VIEW gistest AS
SELECT id, st_astext(a) a, st_astext(b) b, cpu, gpu,
       CASE WHEN cpu != gpu THEN '***' ELSE null END diff
  FROM (SELECT  id, a, b,
                _st_relate(a,b) cpu,
				st_relate(a,b) gpu
          FROM _gistest
		 WHERE id > 0) qry
 ORDER BY id;

CREATE VIEW gistest_r AS
SELECT id, st_astext(b) b, st_astext(a) a, cpu, gpu,
       CASE WHEN cpu != gpu THEN '***' ELSE null END diff
  FROM (SELECT  id, b, a,
                _st_relate(b,a) cpu,
				st_relate(b,a) gpu
          FROM _gistest
		 WHERE id > 0) qry
 ORDER BY id;

---
--- Test data
---

-- point - point
INSERT INTO _gistest(a,b) VALUES
('POINT EMPTY','POINT EMPTY'),
('POINT EMPTY','POINT (1 1)'),
('POINT EMPTY','MULTIPOINT (1 1,2 2,3 3)'),
('POINT (2 2)','POINT (3 3)'),
('POINT (4 4)','POINT (4 4)'),
('POINT (3 3)','MULTIPOINT (1 1,2 2,3 3)'),
('POINT (3 3)','MULTIPOINT (1 2,3 3,4 4)'),
('MULTIPOINT (2 2,3 3,4 4)','MULTIPOINT (3 3,4 4,5 5)'),
('MULTIPOINT (2 2,3 3)','MULTIPOINT (1 1,2 2,3 3,4 4)'),
('MULTIPOINT (1 1,2 2)','MULTIPOINT (3 3,4 4)');

-- point - line
INSERT INTO _gistest(a,b) VALUES
('POINT EMPTY','LINESTRING(1 1,2 2)'),
('POINT (1 1)','LINESTRING(1 1,2 2)'),
('POINT (2 2)','LINESTRING(1 1,2 2)'),
('POINT (3 3)','LINESTRING(1 1,2 2)'),
('POINT (3 3)','LINESTRING(0 0,5 5)'),
('POINT (4 4)','LINESTRING EMPTY'),
('MULTIPOINT (1 1,2 2)','LINESTRING EMPTY'),
('MULTIPOINT (1 1,2 2)','LINESTRING (1 1,1 2)'),
('MULTIPOINT (1 1,3 3)','LINESTRING (2 0,2 4)'),
('MULTIPOINT (1 1,3 3)','LINESTRING (3 0,3 3)'),
('MULTIPOINT (2 0,0 2)','LINESTRING (2 0,2 2,0 2)'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','LINESTRING (0 4,2 0,4 4)'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','LINESTRING (0 0,3 0,3 6,6 6)'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','LINESTRING (0 0,2 0,2 -2)'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((0 0,5 0,5 5),(0 5,5 0))'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((0 0,5 0,5 5),(0 5,5 5,5 0))'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((0 0,5 0,4 4),(0 5,5 5,5 0))'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((0 0,5 0,4 4),(0 5,3 3,5 0))'),
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((1 1,1 0,2 0,2 2),(3 3,3 0,4 0,4 4))');
