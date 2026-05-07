---
--- initialization of regression test database
---
SELECT current_database();
\gset

\getenv regress_dataset_revision REGRESS_DATASET_REVISION
SELECT pgstrom.regression_testdb_revision() = cast(:regress_dataset_revision as text) revision_check_result
\gset

SET client_min_messages TO WARNING;

\if :revision_check_result
---
--- OK, regression test database is the latest revision
---
SELECT pgstrom.regression_testdb_revision();

\else
BEGIN;
---
--- special GUC configuration
---
ALTER DATABASE :current_database SET pg_strom.regression_test_mode = on;

---
--- load initial data of SSBM
---
SET search_path = public;

DROP TABLE IF EXISTS customer;
CREATE TABLE customer (
    c_custkey   integer NOT NULL,
    c_name      character varying(25),
    c_address   character varying(25),
    c_city      character(10),
    c_nation    character(15),
    c_region    character(12),
    c_phone     character(15),
    c_mktsegment character(10)
);

DROP TABLE IF EXISTS date1;
CREATE TABLE date1 (
    d_datekey   integer NOT NULL,
    d_date      character(18),
    d_dayofweek character(12),
    d_month     character(9),
    d_year      integer,
    d_yearmonthnum numeric,
    d_yearmonth character(7),
    d_daynuminweek numeric,
    d_daynuminmonth numeric,
    d_daynuminyear numeric,
    d_monthnuminyear numeric,
    d_weeknuminyear numeric,
    d_sellingseason character(12),
    d_lastdayinweekfl character(1),
    d_lastdayinmonthfl character(1),
    d_holidayfl character(1),
    d_weekdayfl character(1)
);

DROP TABLE IF EXISTS lineorder;
CREATE TABLE lineorder (
    lo_orderkey     bigint,
    lo_linenumber   integer,
    lo_custkey      integer,
    lo_partkey      integer,
    lo_suppkey      integer,
    lo_orderdate    integer,
    lo_orderpriority character(15),
    lo_shippriority character(1),
    lo_quantity     numeric,
    lo_extendedprice numeric,
    lo_ordertotalprice numeric,
    lo_discount     numeric,
    lo_revenue      numeric,
    lo_supplycost   numeric,
    lo_tax          numeric,
    lo_commit_date  character(8),
    lo_shipmode     character(10)
);

DROP TABLE IF EXISTS part;
CREATE TABLE part (
    p_partkey       integer NOT NULL,
    p_name          character varying(22),
    p_mfgr          character(6),
    p_category      character(7),
    p_brand1        character(9),
    p_color         character varying(11),
    p_type          character varying(25),
    p_size          numeric,
    p_container     character(10)
);

DROP TABLE IF EXISTS supplier;
CREATE TABLE supplier (
    s_suppkey       integer NOT NULL,
    s_name          character(25),
    s_address       character varying(25),
    s_city          character(10),
    s_nation        character(15),
    s_region        character(12),
    s_phone         character(15)
);
\copy customer  FROM PROGRAM '$DBGEN_SSBM_CMD -s 1 -X -Tc' DELIMITER '|'
\copy date1     FROM PROGRAM '$DBGEN_SSBM_CMD -s 1 -X -Td' DELIMITER '|'
\copy lineorder FROM PROGRAM '$DBGEN_SSBM_CMD -s 1 -X -Tl' DELIMITER '|'
\copy part      FROM PROGRAM '$DBGEN_SSBM_CMD -s 1 -X -Tp' DELIMITER '|'
\copy supplier  FROM PROGRAM '$DBGEN_SSBM_CMD -s 1 -X -Ts' DELIMITER '|'

---
--- Partitioned lineorder
---
\set plineorder_p1992_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1992_rem1.parquet`
\set plineorder_p1993_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1993_rem1.parquet`
\set plineorder_p1994_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1994_rem1.parquet`
\set plineorder_p1995_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1995_rem1.parquet`
\set plineorder_p1996_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1996_rem1.parquet`
\set plineorder_p1997_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1997_rem1.parquet`
\set plineorder_p1998_rem1_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1998_rem1.parquet`

\set plineorder_p1992_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1992_rem2.arrow`
\set plineorder_p1993_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1993_rem2.arrow`
\set plineorder_p1994_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1994_rem2.arrow`
\set plineorder_p1995_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1995_rem2.arrow`
\set plineorder_p1996_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1996_rem2.arrow`
\set plineorder_p1997_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1997_rem2.arrow`
\set plineorder_p1998_rem2_path `echo -n $ARROW_TEST_DATA_DIR/plineorder_p1998_rem2.arrow`

DROP TABLE IF EXISTS plineorder CASCADE;
CREATE TABLE plineorder (
  LIKE lineorder
) PARTITION BY RANGE (lo_orderdate);

CREATE TABLE plineorder_p1992
PARTITION OF plineorder FOR VALUES FROM (19920000) TO (19930000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1993
PARTITION OF plineorder FOR VALUES FROM (19930000) TO (19940000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1994
PARTITION OF plineorder FOR VALUES FROM (19940000) TO (19950000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1995
PARTITION OF plineorder FOR VALUES FROM (19950000) TO (19960000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1996
PARTITION OF plineorder FOR VALUES FROM (19960000) TO (19970000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1997
PARTITION OF plineorder FOR VALUES FROM (19970000) TO (19980000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1998
PARTITION OF plineorder FOR VALUES FROM (19980000) TO (19990000)
PARTITION BY HASH (lo_shipmode);

CREATE TABLE plineorder_p1992_heap
PARTITION OF plineorder_p1992 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1992_parquet
PARTITION OF plineorder_p1992 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1992_rem1_path');
CREATE FOREIGN TABLE plineorder_p1992_arrow
PARTITION OF plineorder_p1992 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1992_rem2_path');

CREATE TABLE plineorder_p1993_heap
PARTITION OF plineorder_p1993 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1993_parquet
PARTITION OF plineorder_p1993 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1993_rem1_path');
CREATE FOREIGN TABLE plineorder_p1993_arrow
PARTITION OF plineorder_p1993 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1993_rem2_path');

CREATE TABLE plineorder_p1994_heap
PARTITION OF plineorder_p1994 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1994_parquet
PARTITION OF plineorder_p1994 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1994_rem1_path');
CREATE FOREIGN TABLE plineorder_p1994_arrow
PARTITION OF plineorder_p1994 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1994_rem2_path');

CREATE TABLE plineorder_p1995_heap
PARTITION OF plineorder_p1995 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1995_parquet
PARTITION OF plineorder_p1995 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1995_rem1_path');
CREATE FOREIGN TABLE plineorder_p1995_arrow
PARTITION OF plineorder_p1995 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1995_rem2_path');

CREATE TABLE plineorder_p1996_heap
PARTITION OF plineorder_p1996 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1996_parquet
PARTITION OF plineorder_p1996 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1996_rem1_path');
CREATE FOREIGN TABLE plineorder_p1996_arrow
PARTITION OF plineorder_p1996 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1996_rem2_path');

CREATE TABLE plineorder_p1997_heap
PARTITION OF plineorder_p1997 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1997_parquet
PARTITION OF plineorder_p1997 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1997_rem1_path');
CREATE FOREIGN TABLE plineorder_p1997_arrow
PARTITION OF plineorder_p1997 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1997_rem2_path');

CREATE TABLE plineorder_p1998_heap
PARTITION OF plineorder_p1998 FOR VALUES WITH (modulus 3, remainder 0);
CREATE FOREIGN TABLE plineorder_p1998_parquet
PARTITION OF plineorder_p1998 FOR VALUES WITH (modulus 3, remainder 1)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1998_rem1_path');
CREATE FOREIGN TABLE plineorder_p1998_arrow
PARTITION OF plineorder_p1998 FOR VALUES WITH (modulus 3, remainder 2)
SERVER arrow_fdw OPTIONS (file :'plineorder_p1998_rem2_path');

INSERT INTO plineorder (SELECT * FROM lineorder
                         WHERE lo_shipmode IN ('TRUCK','RAIL','MAIN')
						 ORDER BY lo_orderdate);
CREATE INDEX ON plineorder USING brin (lo_orderdate);







\set revision_checker_body 'SELECT CAST(':regress_dataset_revision' as text)'

CREATE OR REPLACE FUNCTION
pgstrom.regression_testdb_revision()
RETURNS text
AS :'revision_checker_body'
LANGUAGE 'sql';

COMMIT;
-- vacuum tables
VACUUM ANALYZE;
--- revision confirmation
SELECT pgstrom.regression_testdb_revision();

\endif
