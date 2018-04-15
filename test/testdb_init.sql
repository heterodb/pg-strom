--
-- Initialization of Test DB
--
SET search_path = public,pgstrom;

SET client_min_messages = error;
DROP EXTENSION IF EXISTS pg_strom CASCADE;
DROP EXTENSION IF EXISTS pgcrypto CASCADE;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS t0;
DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS t2;
DROP TABLE IF EXISTS t3;
DROP TABLE IF EXISTS t4;
DROP TABLE IF EXISTS t5;
DROP TABLE IF EXISTS t6;
DROP TABLE IF EXISTS t7;
DROP TABLE IF EXISTS t8;
DROP TABLE IF EXISTS t9;
DROP TABLE IF EXISTS t_int1;
DROP TABLE IF EXISTS t_int2;
DROP TABLE IF EXISTS t_float1;
DROP TABLE IF EXISTS t_float2;
DROP TABLE IF EXISTS t_numeric1;
DROP TABLE IF EXISTS t_numeric2;
DROP TABLE IF EXISTS t_date1;
DROP TABLE IF EXISTS t_date2;
DROP TABLE IF EXISTS t_date1;
DROP TABLE IF EXISTS t_time1;
DROP TABLE IF EXISTS t_time2;
DROP TABLE IF EXISTS t_timetz1;
DROP TABLE IF EXISTS t_timetz2;
DROP TABLE IF EXISTS t_timestamp1;
DROP TABLE IF EXISTS t_timestamp2;
DROP TABLE IF EXISTS t_timestamptz1;
DROP TABLE IF EXISTS t_timestamptz2;
DROP TABLE IF EXISTS t_network1;
DROP TABLE IF EXISTS t_network2;
DROP TABLE IF EXISTS t_uuid1;
DROP TABLE IF EXISTS t_uuid2;
DROP FUNCTION IF EXISTS pgstrom_regression_test_revision();
RESET client_min_messages;

-- all the setup shall be atomic
BEGIN;

-- initialize random seed
SELECT setseed(0.20180124);

-- create extension
CREATE EXTENSION pg_strom;
CREATE EXTENSION pgcrypto;

-- create tables of DBT-3
CREATE TABLE supplier (
    s_suppkey  INTEGER,
    s_name CHAR(25),
    s_address VARCHAR(40),
    s_nationkey INTEGER,
    s_phone CHAR(15),
    s_acctbal REAL,
    s_comment VARCHAR(101));

CREATE TABLE part (
    p_partkey INTEGER,
    p_name VARCHAR(55),
    p_mfgr CHAR(25),
    p_brand CHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container CHAR(10),
    p_retailprice REAL,
    p_comment VARCHAR(23));

CREATE TABLE partsupp (
    ps_partkey INTEGER,
    ps_suppkey INTEGER,
    ps_availqty INTEGER,
    ps_supplycost REAL,
    ps_comment VARCHAR(199));

CREATE TABLE customer (
    c_custkey INTEGER,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER,
    c_phone CHAR(15),
    c_acctbal REAL,
    c_mktsegment CHAR(10),
    c_comment VARCHAR(117));

CREATE TABLE orders (
    o_orderkey INTEGER,
    o_custkey INTEGER,
    o_orderstatus CHAR(1),
    o_totalprice REAL,
    o_orderdate DATE,
    o_orderpriority CHAR(15),
    o_clerk CHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79));

CREATE TABLE lineitem (
    l_orderkey INTEGER,
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity REAL,
    l_extendedprice REAL,
    l_discount REAL,
    l_tax REAL,
    l_returnflag CHAR(1),
    l_linestatus CHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct CHAR(25),
    l_shipmode CHAR(10),
    l_comment VARCHAR(44));

CREATE TABLE nation (
    n_nationkey INTEGER,
    n_name CHAR(25),
    n_regionkey INTEGER,
    n_comment VARCHAR(152));

CREATE TABLE region (
    r_regionkey INTEGER,
    r_name CHAR(25),
    r_comment VARCHAR(152));

\copy supplier FROM PROGRAM './dbt3/dbgen -X -T s -s 24' delimiter '|';
\copy part     FROM PROGRAM './dbt3/dbgen -X -T P -s 24' delimiter '|';
\copy partsupp FROM PROGRAM './dbt3/dbgen -X -T S -s 24' delimiter '|';
\copy customer FROM PROGRAM './dbt3/dbgen -X -T c -s 24' delimiter '|';
\copy orders   FROM PROGRAM './dbt3/dbgen -X -T O -s 24' delimiter '|';
\copy lineitem FROM PROGRAM './dbt3/dbgen -X -T L -s 24' delimiter '|';
\copy nation   FROM PROGRAM './dbt3/dbgen -X -T n -s 24' delimiter '|';
\copy region   FROM PROGRAM './dbt3/dbgen -X -T r -s 24' delimiter '|';

-- general putpose large table
CREATE TABLE t0 (id int primary key,
                 cat text,
                 aid int,
                 bid int,
                 cid int,
                 did int,
                 eid int,
                 fid int,
                 gid int,
                 hid int,
                 iid int,
                 ymd date);
CREATE TABLE t1 (aid int primary key, atext text, ax float, ay float);
CREATE TABLE t2 (bid int, btext text, bx float, by float);
CREATE TABLE t3 (cid int primary key, ctext text, cx float, cy float);
CREATE TABLE t4 (did int, dtext text, dx float, dy float);
CREATE TABLE t5 (eid int primary key, etext text, ex float, ey float);
CREATE TABLE t6 (fid int, ftext text, fx float, fy float);
CREATE TABLE t7 (gid int primary key, gtext text, gx float, gy float);
CREATE TABLE t8 (hid int, htext text, hx float, hy float);
CREATE TABLE t9 (iid int primary key, itext text, ix float, iy float);

INSERT INTO t0 (SELECT x, random_text(0.5),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_int(0.5, 1, 100000),
                       random_date(0.5)
                  FROM generate_series(1,1000000) x);
INSERT INTO t1 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t2 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t3 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t4 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t5 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t6 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t7 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t8 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
INSERT INTO t9 (SELECT x,md5((x+1)::text),
                       random_float(0.5, 1.0, 1000.0),
                       random_float(0.5, 1.0, 1000.0)
                  FROM generate_series(1, 100000) x);
--
-- Tables for misc data types
--

/* integer data types */
CREATE TABLE t_int1(id int primary key, dummy text,
                    a smallint, b smallint,
                    c int, d int,
                    e bigint, f bigint);
CREATE TABLE t_int2(id int primary key, dummy text,
                    a smallint,
                    c int,
                    e bigint);
INSERT INTO t_int1 (SELECT x, random_text_len(0.5, 20),
                           random_int(0.5, -32767, 32767),
                           random_int(0.5, -32767, 32767),
                           random_int(0.5, -2147483647, 2147483647),
                           random_int(0.5, -2147483647, 2147483647),
                           random_int(0.5, -2147483647, 2147483647),
                           random_int(0.5, -2147483647, 2147483647)
                      FROM generate_series(1,1000000) X);
INSERT INTO t_int2 (SELECT x, random_text_len(0.5, 20),
                           random_int(0.5, -32767, 32767),
                           random_int(0.5, -2147483647, 2147483647),
                           random_int(0.5, -2147483647, 2147483647)
                      FROM generate_series(1,800000) X);
ALTER TABLE t_int1 DROP COLUMN dummy;
ALTER TABLE t_int2 DROP COLUMN dummy;
ALTER TABLE t_int2 ADD COLUMN b smallint;
ALTER TABLE t_int2 ADD COLUMN d int;
ALTER TABLE t_int2 ADD COLUMN f bigint;
INSERT INTO t_int2 (SELECT x, random_int(0.5, -32767, 32767),
                              random_int(0.5, -2147483647, 2147483647),
                              random_int(0.5, -2147483647, 2147483647),
                              random_int(0.5, -32767, 32767),
                              random_int(0.5, -2147483647, 2147483647),
                              random_int(0.5, -2147483647, 2147483647)
                      FROM generate_series(800001,1000000) X);

/* floating point data types */
CREATE TABLE t_float1(id int primary key, dummy text,
                      a float2, b float2,
                      c float4, d float4,
                      e float8, f float8);
CREATE TABLE t_float2(id int primary key, dummy text,
                      a float2,
                      c float4,
                      e float8);
INSERT INTO t_float1 (SELECT x, random_text_len(0.5, 20),
                                random_float(0.5,  -20000.0,  20000.0),
                                random_float(0.5,  -20000.0,  20000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0)
                        FROM generate_series(1,1000000) X);
INSERT INTO t_float2 (SELECT x, random_text_len(0.5, 20),
                                random_float(0.5,  -20000.0,  20000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0)
                        FROM generate_series(1,800000) X);
ALTER TABLE t_float1 DROP COLUMN dummy;
ALTER TABLE t_float2 DROP COLUMN dummy;
ALTER TABLE t_float2 ADD COLUMN b float2;
ALTER TABLE t_float2 ADD COLUMN d float4;
ALTER TABLE t_float2 ADD COLUMN f float8;
INSERT INTO t_float2 (SELECT x, random_float(0.5,  -20000.0,  20000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5,  -20000.0,  20000.0),
                                random_float(0.5, -100000.0, 100000.0),
                                random_float(0.5, -100000.0, 100000.0)
                        FROM generate_series(800001,1000000) X);

/* numeric data types */
CREATE TABLE t_numeric1(id int primary key, dummy text,
                        a numeric, b numeric, c numeric);
CREATE TABLE t_numeric2(id int primary key, dummy text,
                        a numeric, b numeric);
INSERT INTO t_numeric1
      (SELECT x, random_text_len(0.5, 20),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4)
         FROM generate_series(1,1000000) X);
INSERT INTO t_numeric2
      (SELECT x, random_text_len(0.5, 20),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4)
         FROM generate_series(1,800000) X);
ALTER TABLE t_numeric1 DROP COLUMN dummy;
ALTER TABLE t_numeric2 DROP COLUMN dummy;
ALTER TABLE t_numeric2 ADD COLUMN c numeric;
INSERT INTO t_numeric2
      (SELECT x, random_float(0.5, -100000.0, 100000.0)::numeric(10,4),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4),
                 random_float(0.5, -100000.0, 100000.0)::numeric(10,4)
         FROM generate_series(800001,1000000) X);

/* date types */
CREATE TABLE t_date1 (id int primary key, dummy text,
                      a date, b date, c date);
CREATE TABLE t_date2 (id int primary key, dummy text,
                      a date, b date);
INSERT INTO t_date1 (SELECT x, random_text_len(0.5, 20),
                            random_date(0.5),
                            random_date(0.5),
                            random_date(0.5)
                       FROM generate_series(1,1000000) X);
INSERT INTO t_date2 (SELECT x, random_text_len(0.5, 20),
                            random_date(0.5),
                            random_date(0.5)
                       FROM generate_series(1,800000) X);
ALTER TABLE t_date1 DROP COLUMN dummy;
ALTER TABLE t_date2 DROP COLUMN dummy;
ALTER TABLE t_date2 ADD COLUMN c date;
INSERT INTO t_date2 (SELECT x, random_date(0.5),
                               random_date(0.5),
                               random_date(0.5)
                       FROM generate_series(800001,1000000) X);

/* time types */
CREATE TABLE t_time1 (id int primary key, dummy text,
                      a time, b time, c time);
CREATE TABLE t_time2 (id int primary key, dummy text,
                      a time, b time);
INSERT INTO t_time1 (SELECT x, random_text_len(0.5, 20),
                            random_time(0.5),
                            random_time(0.5),
                            random_time(0.5)
                       FROM generate_series(1,1000000) X);
INSERT INTO t_time2 (SELECT x, random_text_len(0.5, 20),
                            random_time(0.5),
                            random_time(0.5)
                       FROM generate_series(1,800000) X);
ALTER TABLE t_time1 DROP COLUMN dummy;
ALTER TABLE t_time2 DROP COLUMN dummy;
ALTER TABLE t_time2 ADD COLUMN c time;
INSERT INTO t_time2 (SELECT x, random_time(0.5),
                               random_time(0.5),
                               random_time(0.5)
                       FROM generate_series(800001,1000000) X);

/* timetz types */
CREATE TABLE t_timetz1 (id int primary key, dummy text,
                        a time, b time, c time);
CREATE TABLE t_timetz2 (id int primary key, dummy text,
                        a time, b time);
INSERT INTO t_timetz1 (SELECT x, random_text_len(0.5, 20),
                              random_timetz(0.5),
                              random_timetz(0.5),
                              random_timetz(0.5)
                         FROM generate_series(1,1000000) X);
INSERT INTO t_timetz2 (SELECT x, random_text_len(0.5, 20),
                              random_timetz(0.5),
                              random_timetz(0.5)
                         FROM generate_series(1,800000) X);
ALTER TABLE t_timetz1 DROP COLUMN dummy;
ALTER TABLE t_timetz2 DROP COLUMN dummy;
ALTER TABLE t_timetz2 ADD COLUMN c time;
INSERT INTO t_timetz2 (SELECT x, random_timetz(0.5),
                                 random_timetz(0.5),
                                 random_timetz(0.5)
                         FROM generate_series(800001,1000000) X);

/* timestamp types */
CREATE TABLE t_timestamp1 (id int primary key, dummy text,
                           a timestamp, b timestamp, c timestamp);
CREATE TABLE t_timestamp2 (id int primary key, dummy text,
                           a timestamp, b timestamp);
INSERT INTO t_timestamp1 (SELECT x, random_text_len(0.5, 20),
                                 random_timestamp(0.5),
                                 random_timestamp(0.5),
                                 random_timestamp(0.5)
                            FROM generate_series(1,1000000) X);
INSERT INTO t_timestamp2 (SELECT x, random_text_len(0.5, 20),
                                 random_timestamp(0.5),
                                 random_timestamp(0.5)
                            FROM generate_series(1,800000) X);
ALTER TABLE t_timestamp1 DROP COLUMN dummy;
ALTER TABLE t_timestamp2 DROP COLUMN dummy;
ALTER TABLE t_timestamp2 ADD COLUMN c timestamp;
INSERT INTO t_timestamp2 (SELECT x, random_timestamp(0.5),
                                    random_timestamp(0.5),
                                    random_timestamp(0.5)
                            FROM generate_series(800001,1000000) X);

/* timestamptz types */
CREATE TABLE t_timestamptz1 (id int primary key, dummy text,
                             a timestamptz, b timestamptz, c timestamptz);
CREATE TABLE t_timestamptz2 (id int primary key, dummy text,
                             a timestamptz, b timestamptz);
INSERT INTO t_timestamptz1 (SELECT x, random_text_len(0.5, 20),
                                   random_timestamp(0.5),
                                   random_timestamp(0.5),
                                   random_timestamp(0.5)
                              FROM generate_series(1,1000000) X);
INSERT INTO t_timestamptz2 (SELECT x, random_text_len(0.5, 20),
                                   random_timestamp(0.5),
                                   random_timestamp(0.5)
                              FROM generate_series(1,800000) X);
ALTER TABLE t_timestamptz1 DROP COLUMN dummy;
ALTER TABLE t_timestamptz2 DROP COLUMN dummy;
ALTER TABLE t_timestamptz2 ADD COLUMN c timestamptz;
INSERT INTO t_timestamptz2 (SELECT x, random_timestamp(0.5),
                                      random_timestamp(0.5),
                                      random_timestamp(0.5)
                              FROM generate_series(800001,1000000) X);
/* interval types (not yet) */

/* money (pay attention for locale) */

/* macaddr */
CREATE TABLE t_network1 (id int primary key, dummy text,
                         a macaddr, b macaddr,
                         c inet, d inet);
CREATE TABLE t_network2 (id int primary key, dummy text,
                         a macaddr,
                         c inet);
INSERT INTO t_network1 (SELECT x, random_text_len(0.5, 20),
                               random_macaddr(0.5),
                               random_macaddr(0.5),
                               random_inet(0.5),
                               random_inet(0.5)
                          FROM generate_series(1,1000000) X);
INSERT INTO t_network2 (SELECT x, random_text_len(0.5, 20),
                               random_macaddr(0.5),
                               random_inet(0.5)
                          FROM generate_series(1,800000) X);
ALTER TABLE t_network1 DROP COLUMN dummy;
ALTER TABLE t_network2 DROP COLUMN dummy;
ALTER TABLE t_network2 ADD COLUMN b macaddr;
ALTER TABLE t_network2 ADD COLUMN d inet;
INSERT INTO t_network2 (SELECT x, random_macaddr(0.5),
                                  random_inet(0.5),
                                  random_macaddr(0.5),
                                  random_inet(0.5)
                          FROM generate_series(800001,1000000) X);

/* uuid */
CREATE TABLE t_uuid1 (id int primary key, dummy text,
                      a uuid, b uuid);
CREATE TABLE t_uuid2 (id int primary key, dummy text,
                      a uuid);
INSERT INTO t_uuid1 (SELECT x, random_text_len(0.5, 20),
                            gen_random_uuid(),
                            gen_random_uuid()
                       FROM generate_series(1,1000000) X);
INSERT INTO t_uuid2 (SELECT x, random_text_len(0.5, 20),
                            gen_random_uuid()
                       FROM generate_series(1,700000) X);
ALTER TABLE t_uuid1 DROP COLUMN dummy;
ALTER TABLE t_uuid2 DROP COLUMN dummy;
ALTER TABLE t_uuid2 ADD COLUMN b uuid;
INSERT INTO t_uuid2 (SELECT row_number() over() + 700000, a, b
                       FROM t_uuid1
                      WHERE random() > 0.5
                      LIMIT 300000);

-- Mark TestDB construction completed
CREATE OR REPLACE FUNCTION
public.pgstrom_regression_test_revision()
RETURNS int
AS 'SELECT 20180124'
LANGUAGE 'sql';

COMMIT;
-- vacuum tables
VACUUM ANALYZE supplier;
VACUUM ANALYZE part;
VACUUM ANALYZE partsupp;
VACUUM ANALYZE customer;
VACUUM ANALYZE orders;
VACUUM ANALYZE lineitem;
VACUUM ANALYZE nation;
VACUUM ANALYZE region;
VACUUM ANALYZE t0;
VACUUM ANALYZE t1;
VACUUM ANALYZE t2;
VACUUM ANALYZE t3;
VACUUM ANALYZE t4;
VACUUM ANALYZE t5;
VACUUM ANALYZE t6;
VACUUM ANALYZE t7;
VACUUM ANALYZE t8;
VACUUM ANALYZE t9;
VACUUM ANALYZE t_int1;
VACUUM ANALYZE t_int2;
VACUUM ANALYZE t_float1;
VACUUM ANALYZE t_float2;
VACUUM ANALYZE t_numeric1;
VACUUM ANALYZE t_numeric2;
VACUUM ANALYZE t_date1;
VACUUM ANALYZE t_date2;
VACUUM ANALYZE t_date1;
VACUUM ANALYZE t_time1;
VACUUM ANALYZE t_time2;
VACUUM ANALYZE t_timetz1;
VACUUM ANALYZE t_timetz2;
VACUUM ANALYZE t_timestamp1;
VACUUM ANALYZE t_timestamp2;
VACUUM ANALYZE t_timestamptz1;
VACUUM ANALYZE t_timestamptz2;
VACUUM ANALYZE t_network1;
VACUUM ANALYZE t_network2;
VACUUM ANALYZE t_uuid1;
VACUUM ANALYZE t_uuid2;


