DROP TABLE IF EXISTS t0;
DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS t2;
DROP TABLE IF EXISTS t3;
DROP TABLE IF EXISTS t4;
DROP TABLE IF EXISTS t5;
DROP TABLE IF EXISTS t6;
DROP TABLE IF EXISTS t7;
DROP TABLE IF EXISTS t8;

CREATE TABLE t0 (id int primary key, cat text, aid int, bid int, cid int, did int, eid int, fid int, gid int, hid int);
CREATE TABLE t1 (aid int, atext text, ax float);
CREATE TABLE t2 (bid int, btext text, bx float);
CREATE TABLE t3 (cid int, ctext text, cx float);
CREATE TABLE t4 (did int, dtext text, dx float);
CREATE TABLE t5 (eid int, etext text, ex float);
CREATE TABLE t6 (fid int, ftext text, fx float);
CREATE TABLE t7 (gid int, gtext text, gx float);
CREATE TABLE t8 (hid int, htext text, hx float);

INSERT INTO t1 (SELECT x, md5((x+1)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t2 (SELECT x, md5((x+2)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t3 (SELECT x, md5((x+3)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t4 (SELECT x, md5((x+4)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t5 (SELECT x, md5((x+5)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t6 (SELECT x, md5((x+6)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t7 (SELECT x, md5((x+7)::text), random()*100.0 FROM generate_series(1,100000) x);
INSERT INTO t8 (SELECT x, md5((x+8)::text), random()*100.0 FROM generate_series(1,100000) x);
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
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1),
                       floor(random() * 100000 + 1)
					   FROM generate_series(1,100000000) x);
