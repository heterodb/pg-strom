DROP TABLE IF EXISTS _gistest CASCADE;

CREATE TABLE _gistest
(
  id  serial,
  a   geometry,
  b   geometry
);
CREATE VIEW gisview AS SELECT id,st_astext(a) a,st_astext(b) b FROM _gistest;

CREATE OR REPLACE FUNCTION __st_relate(geometry,geometry)
RETURNS text
AS '$libdir/postgis-3','relate_full'
LANGUAGE C
STRICT
IMMUTABLE
PARALLEL SAFE
COST 10000;

CREATE OR REPLACE FUNCTION __st_contains(geometry,geometry)
RETURNS bool
AS '$libdir/postgis-3','contains'
LANGUAGE C
STRICT
IMMUTABLE
PARALLEL SAFE
COST 10000;

CREATE OR REPLACE FUNCTION __st_crosses(geometry,geometry)
RETURNS bool
AS '$libdir/postgis-3','crosses'
LANGUAGE C
STRICT
IMMUTABLE
PARALLEL SAFE
COST 10000;

CREATE VIEW gistest AS
SELECT id, st_astext(a) a, st_astext(b) b, cpu, gpu,
       CASE WHEN cpu != gpu THEN '***' ELSE null END diff
  FROM (SELECT  id, a, b,
                __st_relate(a,b) cpu,
				st_relate(a,b) gpu
          FROM _gistest
		 WHERE id > 0) qry
 ORDER BY id;

CREATE VIEW gistest_r AS
SELECT id, st_astext(b) b, st_astext(a) a, cpu, gpu,
       CASE WHEN cpu != gpu THEN '***' ELSE null END diff
  FROM (SELECT  id, b, a,
                __st_relate(b,a) cpu,
				st_relate(b,a) gpu
          FROM _gistest
		 WHERE id > 0) qry
 ORDER BY id;

CREATE VIEW test_st_contains AS
SELECT id, st_astext(a) a, st_astext(b) b,
       cpu1, gpu1, CASE WHEN cpu1 != gpu1 THEN '***' ELSE null END diff1,
       cpu2, gpu2, CASE WHEN cpu2 != gpu2 THEN '***' ELSE null END diff2
  FROM (SELECT id, a, b,
               __st_contains(a,b) cpu1,
			   st_contains(a,b) gpu1,
			   __st_contains(b,a) cpu2,
			   st_contains(b,a) gpu2
		  FROM _gistest
		 WHERE id > 0) qry
 ORDER BY id;

CREATE VIEW test_st_crosses AS
SELECT id, st_astext(a) a, st_astext(b) b,
       cpu1, gpu1, CASE WHEN cpu1 != gpu1 THEN '***' ELSE null END diff1,
       cpu2, gpu2, CASE WHEN cpu2 != gpu2 THEN '***' ELSE null END diff2
  FROM (SELECT id, a, b,
               __st_crosses(a,b) cpu1,
			   st_crosses(a,b) gpu1,
			   __st_crosses(b,a) cpu2,
			   st_crosses(b,a) gpu2
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
('MULTIPOINT (1 1,2 2)','MULTIPOINT (3 3,4 4)'),
('POINT (1 1)','MULTIPOINT (2 2,3 3,4 4)'),-- sumiここから13
('MULTIPOINT (1 2,3 3,4 4)','POINT (1 1)'),-- MULTIPOINTとPOINT
('MULTIPOINT (1 1,2 2,3 3)','POINT EMPTY'),--複数の点とEMPTY
('MULTIPOINT (2 2,3 3)','MULTIPOINT (2 2,3 3)'),-- 全部一致
('MULTIPOINT (1 1,2 2)','MULTIPOINT (1 1,2 2,3 3)'),-- 一部一致
('MULTIPOINT (1 1,2 2)','MULTIPOINT (3 3,4 4,5 5)'),-- 全部一致せず
('MULTIPOINT (1 1,3 1,1 3,3 3)','POINT (2 2)'),-- 4つの点の中央に1つの点
('MULTIPOINT (1 1,2 1,3 1,4 1)','MULTIPOINT (1 2,2 2,3 2,4 2)'),-- 4点が横に平行に並ぶ
('MULTIPOINT (1 1,1 2,1 3,1 4)','MULTIPOINT (2 1,2 2,2 3,2 4)'),-- 4点が縦に平行に並ぶ
('MULTIPOINT (1 1,2 2,3 1,4 2)','MULTIPOINT (1 2,2 1,3 2,4 1)'),-- W状の点とM状の点
('POINT (1 1)','MULTIPOINT (2 1,3 1,2 2,3 3,3 2)'),-- 三角形状
('POINT (1 1)','MULTIPOINT (2 1,1 2,2 2)'),-- 四角形状
('MULTIPOINT (1 1,2 1,1 2,2 2)','MULTIPOINT (1 2,2 2,1 3,2 3)');-- 口が縦に２つ並び「日」に

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
('MULTIPOINT (1 1,2 2,3 3,4 4)','MULTILINESTRING ((1 1,1 0,2 0,2 2),(3 3,3 0,4 0,4 4))'),
('POINT EMPTY','LINESTRING EMPTY'),-- sumiここから21
('POINT (0 0)','LINESTRING (0 0,3 0,3 3,0 0)'),-- 三角形状の線の一部が点と重なる
('POINT (2 1)','LINESTRING (0 0,3 0,3 3,0 0)'),-- 三角形状の線の中に点がある（重ならない）
('POINT (3 1)','MULTILINESTRING ((0 0,4 0),(1 1,3 3))'),-- 2つの線と1つの点。いずれも重ならない。
('POINT (0 0)','MULTILINESTRING ((0 0,4 0),(0 0,2 2))'),-- 2つの線と1つの点。いずれも一点で重なる
('POINT (3 1)','MULTILINESTRING ((0 0,4 0),(0 0,2 2))'),-- 2つの線と1つの点。（線は重なる・点は重ならない）
('POINT (2 2)','MULTILINESTRING ((0 0,4 0,0 4),(0 0,4 4))'),-- 又状に交差する線の中に点（線は重なる・点も重なる）
('POINT (2 1)','MULTILINESTRING ((0 0,4 0,0 4),(0 0,4 4))'),-- 又状に交差する線の中に点（線は重なる・点は重ならない）
('POINT (2 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と点（いずれも重ならない）
('POINT (2 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(1 0,3 4))'),-- 折れた線と直線と点（いずれも重なる）
('POINT (3 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(1 0,3 4))'),-- 折れた線と直線と点（線は重なる・点は重ならない）
('POINT (3 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と点（線は重ならない・点は直線と重なる）
('POINT (1 1)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と点（線は重ならない・点は折れた線と重なる）
('MULTIPOINT (1 2,2 2,4 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と3つの点（いずれも重ならない）
('MULTIPOINT (1 1,2 1,3 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と3つの点（3つの点は全て線に重なる）
('MULTIPOINT (1 1,2 3,3 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と3つの点（3つの点のうち2つが線に重なる）
('MULTIPOINT (1 2,2 3,3 2)','MULTILINESTRING ((0 0,1 1,2 1,1 4),(2 2,4 4))'),-- 折れた線と直線と3つの点（3つの点のうち1つが線に重なる）
('MULTIPOINT (0 0,1 0,2 1,2 2,2 3,3 4)','MULTILINESTRING ((0 0,1 0),(1 0,2 1),(2 1,2 2),(2 2,2 3),(2 3,3 4))'),--細切れで繋がる線と各線をつなぐ点（西武多摩川線と6駅）
('MULTIPOINT (0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 0)','MULTILINESTRING ((0 0,7 0))'),--直線とその上に8つの点（中央線快速・三鷹-新宿）
('MULTIPOINT (0 0,1 0,3 0,6 0,7 0)','MULTILINESTRING ((0 0,7 0))'),--直線とその上に5つの点（中央線通勤特快・三鷹-新宿）
('MULTIPOINT (0 0,6 0,7 0)','MULTILINESTRING ((0 0,7 0))');--直線とその上に3つの点（中央特快・三鷹-新宿）

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
('MULTIPOINT (1 3,2 2,3 1)','TRIANGLE ((0 0,6 0,0 6,0 0))'),
-- sumiここから16
('MULTIPOINT (2 0,1 2,3 2)','TRIANGLE ((2 0,1 2,3 2,2 0))'),	--三角形に3つの点（3点は全て三角形に重なる）
('MULTIPOINT (2 2,1 2,2 1)','TRIANGLE ((2 0,1 2,3 2,2 0))'),	--三角形に3つの点（2点は全て三角形に重なり、残り1点は三角形の中央にあり重ならない）
('MULTIPOINT (2 2,1 2,1 1)','TRIANGLE ((2 0,1 2,3 2,2 0))'),	--三角形に3つの点（2点は全て三角形に重なり、残り1点は三角形の外側にあり重ならない）
('MULTIPOINT (2 2,1 4,2 4)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に3つの点（1点は三角形に重なり、残り2点は全て三角形の外側にあり重ならない）
('MULTIPOINT (2 2,1 4,2 2)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に3つの点（1点は三角形に重なり、1点は三角形の外側にあり、1点は三角形の内側にある）
('MULTIPOINT (2 2,1 4,2 3)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に3つの点（1点は三角形に重なり、1点は三角形の外側にあり、1点は三角形に接する）
('MULTIPOINT (2 2,1 3,2 2)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に3つの点（1点は三角形に重なり、1点は三角形の内側にあり、1点は三角形に接する）
('MULTIPOINT (2 0,1 2,3 2,2 1)','TRIANGLE ((2 0,1 2,3 2,2 0))'),	--三角形に4つの点（3点は全て三角形に重なり、残り1点は三角形の中央にあり重ならない）
('MULTIPOINT (2 0,1 2,3 2,1 1)','TRIANGLE ((2 0,1 2,3 2,2 0))'),	--三角形に4つの点（3点は全て三角形に重なり、残り1点は三角形の外側にあり重ならない）
('MULTIPOINT (2 0,1 3,2 4,3 4)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（2点は全て三角形に重なり、残り2点は三角形の外側にあり重ならない）
('MULTIPOINT (2 0,1 3,2 2,3 2)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（2点は全て三角形に重なり、残り2点は三角形の内側にあり重ならない）
('MULTIPOINT (2 0,1 3,2 2,3 4)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（2点は全て三角形に重なり、1点は三角形の内側にあり、1点は外側にある）
('MULTIPOINT (2 0,1 2,2 2,3 4)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（1点は三角形に重なり、2点は三角形の内側にあり、1点は外側にある）
('MULTIPOINT (2 0,1 2,2 4,3 4)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（1点は三角形に重なり、1点は三角形の内側にあり、2点は外側にある）
('MULTIPOINT (2 0,1 2,2 2,3 2)','TRIANGLE ((2 0,0 3,4 3,2 0))'),	--三角形に4つの点（1点は三角形に重なり、3点は三角形の内側にある）
('MULTIPOINT (2 0,1 4,2 4,3 4)','TRIANGLE ((2 0,0 3,4 3,2 0))');	--三角形に4つの点（1点は三角形に重なり、3点は三角形の外側にある）

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
('MULTIPOINT (2 2,4 4)','MULTIPOLYGON(((1 2,2 1,2 3,1 2)),((2 4,4 2,6 4,4 6,2 4),(3 4,4 3,5 4,4 5,3 4)))'),
-- sumiここから16
('POINT (2 2)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形の中に点１
('POINT (1 2)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形の外に点１
('POINT (1 3)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に１つの点が接する
('MULTIPOINT (1 3,2 3)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（1つは六角形に接し、もう1つは中に）
('MULTIPOINT (1 3,1 2)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（1つは六角形に接し、もう1つは外に）
('MULTIPOINT (1 3,2 1)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（2つとも六角形に接する）
('MULTIPOINT (2 3,3 3)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（2つとも六角形に接せず内側に）
('MULTIPOINT (1 1,1 2)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（2つとも六角形に接せず外側に）
('MULTIPOINT (1 1,2 2)','POLYGON ((2 1,1 3,2 5,3 5,4 3,3 1,2 1))'),--六角形に2つの点（それぞれ六角形の内側と外側に）
('MULTIPOINT (1 2,3 4,5 2)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 2,2 4,3 6,4 6,5 4,4 2,3 2)),((5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つジオヘックス状に配置され、それぞれの六角形の内側に点が1つ存在
('MULTIPOINT (1 5,2 5,5 5)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 2,2 4,3 6,4 6,5 4,4 2,3 2)),((5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つジオヘックス状に配置され、３つの点がどの六角形からも外側に存在
('MULTIPOINT (2 4,3 2,4 2,5 4)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 2,2 4,3 6,4 6,5 4,4 2,3 2)),((5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つジオヘックス状に配置され、六角形がそれぞれ接する点の４箇所にそれぞれ4つの点が存在
('MULTIPOINT (1 2,3 4,5 2)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 3,2 5,3 7,4 7,5 5,4 3,3 3),(5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つ、それぞれ接せずにジオヘックス状に配置され、それぞれの六角形の内側に点が1つ存在
('MULTIPOINT (1 5,2 6,5 6)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 3,2 5,3 7,4 7,5 5,4 3,3 3),(5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つ、それぞれ接せずにジオヘックス状に配置され、３つの点がどの六角形からも外側に存在
('MULTIPOINT (1 4,2 5,5 4)','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 3,2 5,3 7,4 7,5 5,4 3,3 3),(5 0,4 2,5 4,6 4,7 2,6 0,5 0)))'),--六角形が3つ、それぞれ接せずにジオヘックス状に配置され、3つの点がそれぞれの六角形に接する。
('POINT EMPTY','MULTIPOLYGON(((1 0,0 2,1 4,2 4,3 2,2 0,1 0)),((3 2,2 4,3 6,4 6,5 4,4 2,3 2)),((5 0,4 2,5 4,6 4,7 2,6 0,5 0)))');--六角形が3つジオヘックス状に配置され、点はEMPTYである。

-- line-line
INSERT INTO _gistest(a,b) VALUES
('LINESTRING EMPTY','LINESTRING EMPTY'),
('LINESTRING EMPTY','LINESTRING (1 1,2 2)'),
('LINESTRING EMPTY','LINESTRING (1 1,2 2,3 1)'),
('LINESTRING (0 0,2 2)','LINESTRING (0 2,2 0)'),
('LINESTRING (0 0,2 2,4 0)','LINESTRING (0 2,4 2)'),
('LINESTRING (1 1,2 0,3 0,4 1)','LINESTRING(1 1,4 1)'),
('LINESTRING (0 0,2 0,2 2,3 2,3 0,5 0)','LINESTRING(1 0,4 0)'),
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
('MULTILINESTRING ((0 1,3 1),(0 3,3 3))','LINESTRING(3 1,3 3)'),
-- sumiここから19
('LINESTRING (0 0,1 2,2 0)','LINESTRING (0 0,2 0)'),--三角形状
('LINESTRING (0 0,1 2,2 0)','LINESTRING (0 0,3 0)'),--三角形状で底辺が伸びたもの
('LINESTRING (0 0,0 1,1 1,1 0)','LINESTRING (0 1,0 2,1 2,1 1)'),--日
('LINESTRING (0 0,0 1)','MULTILINESTRING ((0 1,1 1),(0 0,1 0))'),--C
('LINESTRING (0 0,0 1)','MULTILINESTRING ((0 1,1 1),(0 0,1 0),(1 0,1 1))'),--口
('LINESTRING (0 0,0 2)','MULTILINESTRING ((0 2,1 2),(0 1,1 1),(0 0,1 0))'),--E
('LINESTRING (0 0,0 2)','MULTILINESTRING ((0 2,1 2),(0 1,1 1))'),--F
('LINESTRING (0 0,0 2)','MULTILINESTRING ((0 1,1 1),(1 0,1 2))'),--H
('LINESTRING (0 0,0 1)','LINESTRING (0 1,0 2)'),--I
('LINESTRING (0 2,2 2)','MULTILINESTRING ((1 2,1 0),(0 0,1 0))'),--J（丁）
('LINESTRING (0 0,0 2)','MULTILINESTRING ((0 1,1 2),(0 1,1 0))'),--K
('LINESTRING (0 0,0 1)','LINESTRING (0 0,1 0)'),--L
('LINESTRING (0 0,0 1,1 0)','LINESTRING (1 0,2 1,2 0)'),--M
('LINESTRING (0 0,0 1,1 0)','LINESTRING (1 0,1 1)'),--N
('LINESTRING (0 0,0 2)','LINESTRING (0 2,1 2,1 1,0 1)'),--P
('LINESTRING (0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 0)','LINESTRING (0 0,1 0,2 0,4 0,6 0,7 0)'),--重なった直線（中央線快速・通勤快速）
('LINESTRING (0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 0)','LINESTRING (0 0,1 0,6 0,7 0)'),--重なった直線（中央線快速・特快）
('LINESTRING (0 2,7 2)','LINESTRING (4 2,3 0,2 0)'),--重なった直線（中央線快速・西武多摩川線）
('LINESTRING (3 0,4 0,4 1)','LINESTRING (0 3,7 3)');--接しない２つの線（西武多摩川線・西武新宿線）

-- triangle-line
INSERT INTO _gistest(a,b) VALUES
('TRIANGLE EMPTY','LINESTRING EMPTY'),
('TRIANGLE EMPTY','LINESTRING (0 0,2 2)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING EMPTY'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 0,1 1)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 0,2 2)'),  -- B-B via inside
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 0,2 0)'),  -- B-B on boundary
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (1 1,2 2)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (1 1,3 3)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 0,3 3)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (-2 2,4 2)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (-2 4,4 4)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 4,4 4)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 4,4 4,4 0)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (1 2,4 4,4 0)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (1 2,4 4,2 1)'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (5 5,8 8)'),
-- sumiここから15
('TRIANGLE ((0 0,4 0,0 4,0 0))','LINESTRING (0 4,4 4,4 0)'), --三角形＋線2本＝逆〼
('TRIANGLE ((0 0,4 0,0 4,0 0))','MULTILINESTRING ((0 4,4 4),(4 0,4 4))'), --三角形＋線2本＝逆〼
('TRIANGLE ((0 0,4 0,0 4,0 0))','MULTILINESTRING ((0 0,4 0),(4 0,0 4),(0 4,0 0))'),--三角形とその上に線3本で同じ形の三角形を乗せる
('TRIANGLE ((0 0,5 0,0 5,0 0))','LINESTRING (1 1,2 2)'),-- 三角形の内部に線（三角形には接さない）
('TRIANGLE ((0 0,5 0,0 5,0 0))','LINESTRING (0 0,2 2)'),-- 三角形の内部に線（線の片側だけ三角形に接する）
('TRIANGLE ((0 0,5 0,0 5,0 0))','LINESTRING (1 1,4 4)'),-- 三角形の内部（三角形には接さない）から外部に突き出る線
('TRIANGLE ((0 0,5 0,0 5,0 0))','LINESTRING (0 0,4 4)'),-- 三角形の内部（三角形には接する）から外部に突き出る線
('TRIANGLE ((0 0,5 0,0 5,0 0))','MULTILINESTRING ((1 1,2 2),(1 2,2 1))'),-- 三角形の内部にX状の2本線（三角形には接さない）
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (1 4,3 4)'), --正三角形の頂点だけ線が接している
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (2 4,4 4)'), --正三角形の頂点だけ線が接している（線は線の端のみ接している）
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (1 3,3 3)'), --正三角形を突き抜ける1本の線
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (0 0,4 0)'), --正三角形の一辺に完全に重なる1本の線
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (0 0,5 0)'), --正三角形の一辺に重なる1本の線（三角形の外に延長している）
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (4 0,5 0)'), --正三角形と、その一辺の延長線を成す直線
('TRIANGLE ((0 0,4 0,2 2,0 0))','LINESTRING (3 0,5 0)'); --正三角形と、その一辺の延長線を成す直線（直線の一部は三角形の一辺に含まれる）

-- triangle-triangle
INSERT INTO _gistest(a,b) VALUES
('TRIANGLE EMPTY','TRIANGLE EMPTY'),
('TRIANGLE ((0 0,1 0,0 1,0 0))','TRIANGLE EMPTY'),
('TRIANGLE EMPTY','TRIANGLE ((0 0,1 0,0 1,0 0))'),
('TRIANGLE ((0 0,1 0,0 1,0 0))','TRIANGLE ((2 1,3 0,3 1,2 1))'),
('TRIANGLE ((0 0,1 0,0 1,0 0))','TRIANGLE ((1 0,2 0,2 1,1 0))'),
('TRIANGLE ((0 0,1 0,1 1,0 0))','TRIANGLE ((1 0,2 0,1 1,1 0))'),
('TRIANGLE ((0 0,4 0,2 2,0 0))','TRIANGLE ((0 3,4 3,2 1,0 3))'),
('TRIANGLE ((0 0,4 0,2 3,0 0))','TRIANGLE ((1 1,3 1,2 2,1 1))'),
-- sumiここから15
('TRIANGLE ((0 0,4 0,2 2,0 0))','TRIANGLE ((0 5,2 3,4 5,0 5))'), --上下で向かい合った三角形（接していない）
('TRIANGLE ((0 0,4 0,2 2,0 0))','TRIANGLE ((0 4,2 2,4 4,0 4))'), --上下で向かい合った三角形（頂点が接している）
('TRIANGLE ((0 0,4 0,2 2,0 0))','TRIANGLE ((0 3,2 1,4 3,0 3))'), --上下で向かい合った三角形（交差し、頂点がそれぞれの三角形の中にある）
('TRIANGLE ((0 0,4 0,2 2,0 0))','TRIANGLE ((0 2,2 0,4 2,0 2))'), --上下で向かい合った三角形（交差し、頂点がそれぞれの三角形の底辺に接している）
('TRIANGLE ((0 1,4 1,2 3,0 1))','TRIANGLE ((0 2,2 0,4 2,0 2))'), --上下で向かい合った三角形（交差し、頂点がそれぞれの三角形の底辺を突き抜けている）
('TRIANGLE ((0 2,4 2,2 4,0 2))','TRIANGLE ((0 2,2 0,4 2,0 2))'), --上下で底辺同士が接している同じ形の三角形

('TRIANGLE ((1 0,3 2,1 2,1 0))','TRIANGLE ((1 0,3 0,3 2,1 0))'), --直角三角形が左右に並んで底辺が接して四角形状になっている。
('TRIANGLE ((2 0,4 2,2 2,2 0))','TRIANGLE ((1 0,3 0,3 2,1 0))'), --上記の直角三角形のうち左側が右移動して交差している
('TRIANGLE ((3 0,5 2,3 2,3 0))','TRIANGLE ((1 0,3 0,3 2,1 0))'), --上記がさらに右移動して底辺が接して全体でひし形状となる。
('TRIANGLE ((2 1,4 3,2 3,2 1))','TRIANGLE ((1 0,3 0,3 2,1 0))'), --直角三角形が左右に並んで底辺が接して四角形状になっていた左側の三角形が右斜め上に移動して底辺同士の一部が接している
('TRIANGLE ((1 3,3 3,2 5,1 3))','TRIANGLE ((1 0,3 2,1 2,1 0))'), --上に正三角形、下に直角三角形（接していない）
('TRIANGLE ((1 2,3 2,2 4,1 2))','TRIANGLE ((1 0,3 2,1 2,1 0))'), --上記の三角形が下に移動し、それぞれの一辺が完全に接している
('TRIANGLE ((1 1,3 1,2 3,1 1))','TRIANGLE ((1 0,3 2,1 2,1 0))'), --上に正三角形がさらに下に移動し、お互い交差している。
('TRIANGLE ((1 0,3 0,2 2,1 0))','TRIANGLE ((1 0,3 2,1 2,1 0))'), --上に正三角形がさらに下に移動し、頂点と頂点、線と線、頂点と線が接している。
('TRIANGLE ((1 -1,3 -1,2 1,1 -1))','TRIANGLE ((1 0,3 2,1 2,1 0))'); --上に正三角形がさらに下に移動し、頂点と線が接している。

-- triangle-polygon
INSERT INTO _gistest(a,b) VALUES
('TRIANGLE EMPTY','POLYGON EMPTY'),
('TRIANGLE ((0 0,1 0,0 1,0 0))','POLYGON EMPTY'),
('TRIANGLE EMPTY','POLYGON ((0 0,1 0,1 1,0 1,0 0))'),
('TRIANGLE ((0 0,2 0,1 1,0 0))','POLYGON ((3 0,3 2,5 2,5 0,3 0))'),
('TRIANGLE ((0 0,2 0,1 1,0 0))','POLYGON ((0 1,2 1,2 2,0 2,0 1))'),
('TRIANGLE ((0 0,2 0,1 1,0 0))','POLYGON ((0 2,1 1,2 2,1 3,0 2))'),
('TRIANGLE ((0 0,2 3,4 0,0 0))','POLYGON ((0 2,4 2,4 4,0 4,0 2))'),
('TRIANGLE ((0 0,2 0,0 2,0 0))','POLYGON ((0 2,2 4,4 2,2 0,0 2))'),
('TRIANGLE ((0 0,2 0,0 2,0 0))','POLYGON ((0 0,2 0,0 2,0 0))'),
('TRIANGLE ((0 0,2 0,0 2,0 0))','POLYGON ((0 2,2 0,0 -2,0 2))'),
('TRIANGLE ((0 0,2 0,0 2,0 0))','POLYGON ((-3 0,0 3,3 0,0 -1,-3 0))'),
('TRIANGLE ((0 0,4 0,0 4,0 0))','POLYGON ((1 1,2 1,2 2,1 2,1 1))'),
('TRIANGLE ((0 0,5 0,0 5,0 0))','POLYGON ((1 1,2 1,2 2,1 2,1 1))'),
-- sumiここから16
('TRIANGLE ((4 2,6 2,4 4,4 2))','POLYGON ((1 2,3 2,3 4,1 4,1 2))'), --左に正方形、右に直角三角形（どちらも接していない）
('TRIANGLE ((4 2,6 2,4 4,4 2))','POLYGON ((2 2,4 2,4 4,2 4,2 2))'), --上記の正方形が右に移動して三角形と一辺が全て接する
('TRIANGLE ((4 2,6 2,4 4,4 2))','POLYGON ((3 2,5 2,5 4,3 4,3 2))'), --上記の正方形がさらに右に移動して三角形と一部が重なる
('TRIANGLE ((4 2,6 2,4 4,4 2))','POLYGON ((4 2,6 2,6 4,4 4,4 2))'), --上記の正方形がさらに右に移動して三角形と重なる
('TRIANGLE ((0 0,1 0,0 2,0 0))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --左に三角形、右に六角形（どちらも接していない）
('TRIANGLE ((1 0,2 0,1 2,1 0))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形が右に移動して六角形と一辺が全て接する
('TRIANGLE ((2 0,3 0,2 2,2 0))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形がさらに右に移動して六角形の内部に入り、一辺が全て接する
('TRIANGLE ((3 0,4 0,3 2,3 0))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形がさらに右に移動して六角形と交差する（線が交差し、一点は接する）
('TRIANGLE ((2 1,3 1,2 3,2 1))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形が六角形の中に入り、どこも接していない。
('TRIANGLE ((3 2,4 2,3 4,3 2))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形が右斜め上に移動。六角形の中にあり、一辺が全て接する
('TRIANGLE ((3 3,4 3,3 5,3 3))','POLYGON ((2 0,3 0,4 2,3 4,2 4,1 2,2 0))'), --上記の三角形が真上に移動して六角形と交差する（線同士交差、線が平行、線と点が接する）
('TRIANGLE ((0 0,7 0,0 7,0 0))','POLYGON ((0 0,2 0,2 2,0 2,0 0))'), --三角形の中に正方形があり、角の１点と２つの線が接している
('TRIANGLE ((0 0,7 0,0 7,0 0))','POLYGON ((1 1,3 1,3 3,1 3,1 1))'), --上記の正方形が右上に移動し、三角形の内部にあるが点も線も接していない
('TRIANGLE ((0 0,7 0,0 7,0 0))','POLYGON ((1 2,3 2,3 4,1 4,1 2))'), --上記の正方形が真上に移動し、角と線が接している
('TRIANGLE ((0 0,7 0,0 7,0 0))','POLYGON ((1 3,3 3,3 5,1 5,1 3))'), --上記の正方形がさらに真上に移動し、２つの線が交差している。
('TRIANGLE ((0 0,7 0,0 7,0 0))','POLYGON ((3 5,5 5,5 7,3 7,3 5))'); --上記の正方形は三角形の外にあり、どこも接していない。

-- polygon-line
INSERT INTO _gistest(a,b) VALUES
('POLYGON EMPTY','LINESTRING EMPTY'),
('POLYGON EMPTY','LINESTRING (0 0,1 1)'),
('POLYGON ((0 0,2 0,0 2,0 0))','LINESTRING EMPTY'),
('POLYGON ((0 0,2 0,2 2,0 2,0 0))','LINESTRING (-1 1,3 1)'),
('POLYGON ((0 0,2 0,2 2,0 2,0 0))','LINESTRING (0 1,2 1)'),
('POLYGON ((0 0,2 0,2 2,0 2,0 0))','LINESTRING (-2 1,2 -3)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 2,3 2)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 2,3 1,3 3)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 2,4 2)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 2,4 2,2 3)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 3,3 3,3 0)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 4,4 6)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 4,2 6,4 4)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (-2 2,2 6)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (-2 0,6 0)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (4 0,4 0,0 4)'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (4 0,4 0,-1 4)'),
-- sumiここから20
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 0,4 0,4 4,0 4,0 0)'),--四角形と、それと同じ大きさの形を成すLINE
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 0,2 0,2 2,0 2,0 0)'),--四角形と、その中に小さな四角形を成すLINEがあり、２辺が接している
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 0,3 0,3 2,1 2,1 0)'),--四角形と、その中に小さな四角形を成すLINEがあり、1辺が接している
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 1,3 1,3 3,1 3,1 1)'),--四角形と、その中に小さな四角形を成すLINEがあり、どこにも接していない
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 0,4 4)'),--四角形と、その中に対角線を結ぶ斜線が一本
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 1,3 3)'),--四角形と、その中にどこにも接しない斜線が一本
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (0 0,5 5)'),--四角形と、その中に対角線を結ぶ斜線が四角形の外まで伸びている
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','LINESTRING (1 1,5 5)'),--四角形の中から伸びる斜線が四角形の角１箇所に重なって外まで伸びている。
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','MULTILINESTRING ((0 0,4 4),(0 4,4 0))'),--四角形と、その中に対角線を結ぶ斜線が2本
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','MULTILINESTRING ((1 1,3 3),(1 3,3 1))'),--上記の斜線が四角形には接していない
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','MULTILINESTRING ((0 0,5 5),(0 4,4 -1))'),--四角形と、その中に対角線を結ぶ斜線が2本あり、それぞれ四角形の外まで伸びている。
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (2 2,3 3)'),--複雑なポリゴンの中に、接しない直線が1本
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (2 2,4 3)'),--複雑なポリゴンの中に直線が1本あり、ポリゴンの角に接する
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (2 2,5 4)'),--複雑なポリゴンの中に直線が1本あり、ポリゴンの角に接して外まで伸びる
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (2 2,4 4)'),--複雑なポリゴンの中に直線が1本、ポリゴンの線に接して外まで伸びる
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (1 4,3 0)'),--複雑なポリゴンを突き抜ける直線1本（ポリゴンの線2本と交差）
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (1 2,5 1)'),--複雑なポリゴンを突き抜ける直線1本（ポリゴンの線4本と交差）
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (1 2,4 2)'),--複雑なポリゴンの外部から内部に刺さる直線（ポリゴンの線1本と角1箇所に接する）
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (1 2,5 2)'),--複雑なポリゴンの外部から内部に刺さる直線（ポリゴンの線1本と角2箇所に接する）
('POLYGON ((0 0,3 2,4 1,5 2,4 3,3 5,1 1,0 2,0 0))','LINESTRING (1 2,6 2)');--複雑なポリゴンを突き抜ける直線1本（ポリゴンの線1本と角2箇所に接する）

-- polygon-polygon
INSERT INTO _gistest(a,b) VALUES
('POLYGON EMPTY','POLYGON EMPTY'),
('POLYGON ((0 0,2 0,0 2,0 0))','POLYGON EMPTY'),
('POLYGON EMPTY','POLYGON ((0 0,2 0,0 2,0 0))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','POLYGON((4 2,8 2,8 6,4 6,4 2))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','POLYGON((5 0,9 0,9 4,5 4,5 0))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','POLYGON((1 1,3 1,3 3,1 3,1 1))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','POLYGON((2 2,6 2,6 6,2 6,2 2))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))','POLYGON((4 8,4 4,8 4,8 8,4 8))'),

('POLYGON ((0 0,6 0,6 6,0 6,0 0),(1 1,5 1,5 5,1 5,1 1))',
 'POLYGON ((2 2,4 2,4 4,2 4,2 2))'),	-- fully contained in hole
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(1 2,5 2,5 4,1 4,1 2))',
 'POLYGON ((2 3,3 2,4 3,3 4,2 3))'),	-- touched hole at points
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(1 2,5 2,5 4,1 4,1 2))',
 'POLYGON ((2 2,4 2,4 4,2 4,2 2))'),	-- touched hole at line
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(1 2,5 2,5 4,1 4,1 2))',
 'POLYGON ((2 1,4 1,4 5,2 5,2 1))'), 	-- intersection to hole
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(1 2,5 2,5 4,1 4,1 2))',
 'POLYGON ((0 8,0 3,3 3,3 8,0 8))'),	-- intersection to ring/hole
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(3 3,5 3,5 5,3 5,3 3))',
 'POLYGON ((1 1,3 1,3 3,1 3,1 1))'),	-- touched hole at point
('POLYGON ((0 0,6 0,6 6,0 6,0 0),(3 3,4 2,5 3,4 4,3 3))',
 'POLYGON ((2 3,3 2,2 1,1 2,2 3))'),	-- disjoint to hole
-- sumiここから16
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((2 3,3 3,3 4,2 4,2 3))'),-- コの型の間に正方形（接していない）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((2 2,3 2,3 3,2 3,2 2))'),-- コの型の間に正方形（一辺が接している）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((2 1,3 1,3 2,2 2,2 1))'),-- コの型の中に正方形（2辺が接している）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((0 2,1 2,1 3,0 3,0 2))'),-- コの型の角に正方形が一点で接している
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((2 2,5 1,5 5,2 5,2 2))'),-- コの型の間に正方形（2辺が接している）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((3 2,6 1,6 5,3 5,3 2))'),-- コの型の間に正方形（3辺が接している）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((3 3,4 3,4 6,3 6,3 3))'),-- コの型に重なる長方形（コの外側には出ていない）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((3 3,4 3,4 7,3 7,3 3))'),-- コの型に重なる長方形（コの外側には出ている）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((3 1,4 1,4 7,3 7,3 1))'),-- コの型に重なる長方形（長方形の片側はコの内側、もう片方は外に出ている）
('POLYGON ((1 1,7 1,7 6,1 6,1 5,6 5,6 2,1 2,1 1))','POLYGON((3 0,4 0,4 7,3 7,3 0))'),-- コの型に重なる長方形（長方形の両側ともコの外に出ている）
('POLYGON ((1 3,2 3,2 4,1 4,1 3),(1 5,2 5,2 6,1 6,1 5))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))'),	--台形と２つの正方形がそれぞれどこも接していない
('POLYGON ((2 3,3 3,3 4,2 4,2 3),(1 5,2 5,2 6,1 6,1 5))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))'),	--台形に1つの正方形が重なり、もうひとつの正方形はどこにも接していない。
('POLYGON ((2 3,3 3,3 4,2 4,2 3),(3 5,4 5,4 6,3 6,3 5))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))'),	--台形に1つの正方形が重なり、もうひとつの正方形は台形の角に接している
('POLYGON ((2 3,3 3,3 4,2 4,2 3),(4 5,5 5,5 6,4 6,4 5))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))'),	--台形に2つの正方形が重なる
('POLYGON ((4 3,5 3,5 4,4 4,4 3),(4 5,5 5,5 6,4 6,4 5))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))'),	--台形に1つの正方形が重なり、もうひとつの正方形は台形の内側にあってどこにも接していない
('POLYGON ((3 2,4 2,4 3,3 3,3 2),(5 3,6 3,6 4,5 4,5 3))',
 'POLYGON ((1 1,7 1,7 6,4 5,1 1))');	--台形の中に２つの正方形があり、それぞれどこも接していない
