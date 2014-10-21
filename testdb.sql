DROP TABLE IF EXISTS t0;
DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS t2;
DROP TABLE IF EXISTS t3;
DROP TABLE IF EXISTS t4;


CREATE TABLE t0 (id int primary key, cat text, aid int, bid int, cid int, did int, x float, y float, z text);
CREATE TABLE t1 (aid int, atext text);
CREATE TABLE t2 (bid int, btext text);
CREATE TABLE t3 (cid int, ctext text);
CREATE TABLE t4 (did int, dtext text);

INSERT INTO t1 (SELECT x, md5((x+1)::text) FROM generate_series(1,40000) x);
INSERT INTO t2 (SELECT x, md5((x+2)::text) FROM generate_series(1,40000) x);
INSERT INTO t3 (SELECT x, md5((x+3)::text) FROM generate_series(1,40000) x);
INSERT INTO t4 (SELECT x, md5((x+4)::text) FROM generate_series(1,40000) x);
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
                       floor(random() * 40000 + 1),
                       floor(random() * 40000 + 1),
                       floor(random() * 40000 + 1),
                       floor(random() * 40000 + 1),
                       random() * 100,
                       random() * 100,
                       md5(x::text) FROM generate_series(1,20000000) x);
