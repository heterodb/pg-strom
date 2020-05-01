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

-- point - triangle
INSERT INTO _gistest(a,b) VALUES
('POINT EMPTY','TRIANGLE EMPTY'),
('POINT (1 1)','TRIANGLE EMPTY'),
('MULTIPOINT (1 1,2 2,3 3)','TRIANGLE EMPTY'),
('POINT EMPTY','TRIANGLE ((0 0,2 0,0 2,0 0))'),
('POINT (2 2)','TRIANGLE ((0 0,5 0,0 5,0 0))'),
('POINT (2 2)','TRIANGLE ((0 0,4 0,0 4,0 0))'),
('POINT (2 2)','TRIANGLE ((0 0,2 2,4 0,0 0))'),
('POINT (8 8)','TRIANGLE ((-2 -2,2 -2,0 3,-2 -2))'),
('MULTIPOINT (2 2,6 6)','TRIANGLE ((0 0,6 0,0 6,0 0))'),
('MULTIPOINT (2 2,3 3,4 4)','TRIANGLE ((0 0,6 0,0 6,0 0))'),
('MULTIPOINT (4 2,2 4)','TRIANGLE ((0 0,6 0,0 6,0 0))'),
('MULTIPOINT (4 2,2 4,0 6)','TRIANGLE ((0 0,6 0,0 6,0 0))'),
('MULTIPOINT (1 3,2 2,3 1)','TRIANGLE ((0 0,6 0,0 6,0 0))');

-- point - polygon
INSERT INTO _gistest(a,b) VALUES
('POINT EMPTY','POLYGON EMPTY'),
('POINT (1 1)','POLYGON EMPTY'),
('POINT (1 1)','MULTIPOLYGON(EMPTY,EMPTY)'),
('POINT (2 2)','POLYGON ((0 0,4 0,4 4,0 4,0 0))'),
('POINT (2 2)','POLYGON ((2 0,3 1,2 2,1 1,2 0))'),
('POINT (2 2)','POLYGON ((0 0,3 0,3 2,0 2,0 0))'),
('POINT (2 2)','POLYGON ((0 0,1 0,1 1,0 1,0 0))'),
('POINT (3 3)','POLYGON ((0 0,5 0,5 5,0 5,0 0),(3 2,4 3,2 5,1 4,3 2))'),
('POINT (3 3)','POLYGON ((0 0,5 0,5 5,0 5,0 0),(2 2,3 2,3 3,2 3,2 2))'),
('POINT (3 3)','POLYGON ((0 0,5 0,5 5,0 5,0 0),(2 2,3 2,3 4,2 4,2 2))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,1 0,0 1,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,3 0,3 3,0 3,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,2 0,2 2,0 2,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,2 0,2 3,0 3,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,4 0,4 4,0 4,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((0 0,5 0,5 5,0 5,0 0))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((3 3,5 3,5 5,3 5,3 3))'),
('MULTIPOINT (2 2,4 4)','POLYGON ((1 3,3 1,5 3,3 5,1 3))'),
('MULTIPOINT (2 2,4 4,6 6)','POLYGON((0 0,8 0,8 8,0 8,0 0),(1 2,2 1,3 2,2 3,1 2),(3 4,4 3,5 4,4 5,3 4),(5 6,6 5,7 6,6 7,5 6))'),
('MULTIPOINT (2 2,4 4,6 6,8 8)','POLYGON((0 0,8 0,8 8,0 8,0 0),(1 2,2 1,3 2,2 3,1 2),(3 4,4 3,5 4,4 5,3 4),(5 6,6 5,7 6,6 7,5 6))'),
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((0 0,1 1,2 0,0 0)),((0 2,1 3,0 4,0 2)),((2 0,3 1,4 0,2 0)))'),
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((1 2,2 1,3 2,2 3,1 2)),((3 4,4 3,5 4,4 5,3 4)))'),
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((2 2,3 2,3 3,2 3,2 2)),((3 3,4 3,4 4,3 4,3 3)))'),
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((2 2,3 2,3 3,2 3,2 2)),((3 3,4 3,4 4,3 4,3 3)),((4 5,5 4,6 5,5 6,4 5)))'),
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((1 2,2 1,2 3,1 2)),((2 4,4 2,6 4,4 6,2 4),(3 4,4 3,5 4,4 5,3 4)))');

-- line-line
INSERT INTO _gistest(a,b) VALUES
('LINESTRING EMPTY','LINESTRING EMPTY'),
('LINESTRING EMPTY','LINESTRING (1 1,2 2)'),
('LINESTRING EMPTY','LINESTRING (1 1,2 2,3 1)'),
('LINESTRING (0 0,2 2)','LINESTRING (0 2,2 0)'),
('LINESTRING (0 0,2 2,4 0)','LINESTRING (0 2,4 2)'),
-- issue #461
-- ('LINESTRING (1 1,2 0,3 0,4 1)','LINESTRING(1 1,4 1)'),
-- ('LINESTRING (0 0,2 0,2 2,3 2,3 0,5 0)','LINESTRING(1 0,4 0)'),
('LINESTRING (0 0,1 1,2 0)','LINESTRING (0 1,2 1)'),
('LINESTRING (0 0,2 2,4 4,6 6)','LINESTRING (1 1,5 5)'),
('LINESTRING (0 0,2 2,4 4,6 6)','LINESTRING (0 0,6 6)'),
('LINESTRING (0 0,2 2,4 4,6 6)','LINESTRING (3 3,6 6)'),
('MULTILINESTRING (EMPTY,EMPTY,EMPTY)','LINESTRING (1 1,2 2)'),
('MULTILINESTRING (EMPTY,EMPTY,EMPTY)',
 'MULTILINESTRING (EMPTY,EMPTY,EMPTY)'),
('MULTILINESTRING (EMPTY,EMPTY,EMPTY)',
 'MULTILINESTRING (EMPTY,(0 0,1 1,2 0),EMPTY)'),
('MULTILINESTRING ((0 1,3 1),(0 3,3 3))','LINESTRING(3 0,3 4)'),
('MULTILINESTRING ((0 1,3 1),(0 3,3 3))','LINESTRING(3 1,3 3)');


