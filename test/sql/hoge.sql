\t on
SET client_min_messages = error;
DROP SCHEMA IF EXISTS hoge CASCADE;
CREATE SCHEMA hoge;
RESET client_min_messages;
SET search_path = hoge,public;

CREATE TABLE tbl (
  id int,
  a  int,
  b  int,
  c  float,
  d  date,
  ts timestamp,
  tx text);

SELECT pgstrom.random_setseed(20230910);

INSERT INTO tbl (SELECT x, pgstrom.random_int(1, -3200000,  3200000),
	                   pgstrom.random_int(1, -3200000,  3200000),
			   pgstrom.random_float(1,  -100000.0,   100000.0),
			   pgstrom.random_date(1),
			   pgstrom.random_timestamp(1),
			   md5(random()::text)
		   FROM generate_series(1,40000) x);
\! $PG2ARROW_CMD -t hoge.tbl -o $MY_DATA_DIR/hoge.arrow

\set hoge_arrow `echo -n $MY_DATA_DIR/hoge.arrow`

IMPORT FOREIGN SCHEMA f_tbl FROM SERVER arrow_fdw INTO hoge
  OPTIONS (file :'hoge_arrow');

EXPLAIN (VERBOSE,COSTS OFF) SELECT a % 20, count(*), sum(b) from tbl group by 1;
SELECT a % 20, count(*), sum(b) from tbl group by 1 order by 1;
