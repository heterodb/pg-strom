--
-- Definition of test dataset
--
DROP TABLE IF EXISTS t0;

CREATE TABLE t0 (id int primary key,
		 ymd date,
                 cat text,
		 x float,
		 y float,
		 z text);

INSERT INTO t0 (SELECT x, now()::date - 1000 + (1000 * random())::int,
                       CASE floor(random()*26)
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
		       1000.0 * random(),
		       1000.0 * random(),
		       md5(random()::text)
                  FROM generate_series(1,200000) x);
