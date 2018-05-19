--
-- Definition of test dataset
--
DROP TABLE IF EXISTS t0;
DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS t2;
DROP TABLE IF EXISTS t3;
DROP TABLE IF EXISTS t4;
DROP TABLE IF EXISTS t5;
DROP TABLE IF EXISTS t6;
DROP TABLE IF EXISTS t7;
DROP TABLE IF EXISTS t8;
DROP TABLE IF EXISTS pt;
DROP TABLE IF EXISTS ph;

CREATE TABLE t0 (id int primary key,
                 cat text,
                 aid int,
                 bid int,
                 cid int,
                 did int,
                 eid int,
                 fid int,
                 gid int,
                 hid int);
CREATE TABLE t1 (aid int, atext text, ax float, ay float);
CREATE TABLE t2 (bid int, atext text, bx float, by float);
CREATE TABLE t3 (cid int, atext text, cx float, cy float);
CREATE TABLE t4 (did int, atext text, dx float, dy float);
CREATE TABLE t5 (eid int, atext text, ex float, ey float);
CREATE TABLE t6 (fid int, atext text, fx float, fy float);
CREATE TABLE t7 (gid int, atext text, gx float, gy float);
CREATE TABLE t8 (hid int, atext text, hx float, hy float);

CREATE TABLE pt (id int,
                 ymd date not null,  -- partition key
                 cat text,
                 aid int,
                 bid int,
                 cid int,
                 did int,
                 eid int,
                 fid int,
                 gid int,
                 hid int)
  PARTITION BY RANGE (ymd);
CREATE TABLE pt_2010 PARTITION OF pt
  FOR VALUES FROM (MINVALUE)     TO ('2011-01-01');
CREATE TABLE pt_2011 PARTITION OF pt
  FOR VALUES FROM ('2011-01-01') TO ('2012-01-01');
CREATE TABLE pt_2012 PARTITION OF pt
  FOR VALUES FROM ('2012-01-01') TO ('2013-01-01');
CREATE TABLE pt_2013 PARTITION OF pt
  FOR VALUES FROM ('2013-01-01') TO ('2014-01-01');
CREATE TABLE pt_2014 PARTITION OF pt
  FOR VALUES FROM ('2014-01-01') TO ('2015-01-01');
CREATE TABLE pt_2015 PARTITION OF pt
  FOR VALUES FROM ('2015-01-01') TO ('2016-01-01');
CREATE TABLE pt_2016 PARTITION OF pt
  FOR VALUES FROM ('2016-01-01') TO ('2017-01-01');
CREATE TABLE pt_2017 PARTITION OF pt
  FOR VALUES FROM ('2017-01-01') TO ('2018-01-01');
CREATE TABLE pt_2018 PARTITION OF pt
  FOR VALUES FROM ('2018-01-01') TO ('2019-01-01');
CREATE TABLE pt_2019 PARTITION OF pt
  FOR VALUES FROM ('2019-01-01') TO (MAXVALUE);


INSERT INTO t1 (SELECT x, md5((x+1)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t2 (SELECT x, md5((x+2)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t3 (SELECT x, md5((x+3)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t4 (SELECT x, md5((x+4)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t5 (SELECT x, md5((x+5)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t6 (SELECT x, md5((x+6)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t7 (SELECT x, md5((x+7)::text), random()*100.0
                  FROM generate_series(1,400000) x);
INSERT INTO t8 (SELECT x, md5((x+8)::text), random()*100.0
                  FROM generate_series(1,400000) x);

INSERT INTO t0 (SELECT x, CASE floor(random()*26)
                          WHEN  0 THEN 'aaa'
                          WHEN  1 THEN 'bbb'
                          WHEN  2 THEN 'ccc'
                          WHEN  3 THEN 'ddd'
                          WHEN  4 THEN 'eee'
                          WHEN  5 THEN 'fff'
                          WHEN  6 THEN 'ggg'
                          WHEN  7 THEN 'hhh'
                          WHEN  8 THEN 'iii'
                          WHEN  9 THEN 'jjj'
                          WHEN 10 THEN 'kkk'
                          WHEN 11 THEN 'lll'
                          WHEN 12 THEN 'mmm'
                          WHEN 13 THEN 'nnn'
                          WHEN 14 THEN 'ooo'
                          WHEN 15 THEN 'ppp'
                          WHEN 16 THEN 'qqq'
                          WHEN 17 THEN 'rrr'
                          WHEN 18 THEN 'sss'
                          WHEN 19 THEN 'ttt'
                          WHEN 20 THEN 'uuu'
                          WHEN 21 THEN 'vvv'
                          WHEN 22 THEN 'www'
                          WHEN 23 THEN 'xxx'
                          WHEN 24 THEN 'yyy'
                          ELSE 'zzz'
                          END,
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000),
                       pgstrom.random_int(0.5, 1, 400000)
                  FROM generate_series(1,200000000) x);

INSERT INTO pt (SELECT id, pgstrom.random_date(0.0, '2010-01-01'::date,
                                                    '2020-01-01'::date),
                       cat, aid, bid, cid, did, eid, fid, gid, hid
                  FROM t0);
