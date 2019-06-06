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
DROP TABLE IF EXISTS t_interval1;
DROP TABLE IF EXISTS t_interval2;
DROP TABLE IF EXISTS t_int4range1;
DROP TABLE IF EXISTS t_int4range2;
DROP TABLE IF EXISTS t_int8range1;
DROP TABLE IF EXISTS t_int8range2;
DROP TABLE IF EXISTS t_tsrange1;
DROP TABLE IF EXISTS t_tsrange2;
DROP TABLE IF EXISTS t_tstzrange1;
DROP TABLE IF EXISTS t_tstzrange2;
DROP TABLE IF EXISTS t_daterange1;
DROP TABLE IF EXISTS t_daterange2;
DROP TABLE IF EXISTS t_money1;
DROP TABLE IF EXISTS t_money2;
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
CREATE TABLE t_interval1 (id int primary key, dummy text,
                          a interval, b interval, c interval);
CREATE TABLE t_interval2 (id int primary key, dummy text,
                          a interval, b interval);
INSERT INTO t_interval1 (SELECT x, random_text_len(0.5, 20),
                                random_timestamp(0.5) - random_timestamp(0.5),
                                random_timestamp(0.5) - random_timestamp(0.5),
                                random_timestamp(0.5) - random_timestamp(0.5)
                           FROM generate_series(1,1000000) X);
INSERT INTO t_interval2 (SELECT x, random_text_len(0.5, 20),
                                random_timestamp(0.5) - random_timestamp(0.5),
                                random_timestamp(0.5) - random_timestamp(0.5)
                           FROM generate_series(1,800000) X);
ALTER TABLE t_interval1 DROP COLUMN dummy;
ALTER TABLE t_interval2 DROP COLUMN dummy;
ALTER TABLE t_interval2 ADD COLUMN c interval;
INSERT INTO t_interval2 (SELECT x, random_timestamp(0.5)-random_timestamp(0.5),
                                   random_timestamp(0.5)-random_timestamp(0.5),
                                   random_timestamp(0.5)-random_timestamp(0.5)
                           FROM generate_series(800001,1000000) X);

/* money (pay attention for 'lc_monetary') */
SET lc_monetary = 'C';
CREATE TABLE t_money1 (id int primary key, dummy text,
                       a money, b money, c money);
CREATE TABLE t_money2 (id int primary key, dummy text,
                       a money, b money);
INSERT INTO t_money1 (SELECT x, random_text_len(0.5, 20),
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money
                        FROM generate_series(1,1000000) X);
INSERT INTO t_money2 (SELECT x, random_text_len(0.5, 20),
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money
                        FROM generate_series(1,800000) X);
ALTER TABLE t_money1 DROP COLUMN dummy;
ALTER TABLE t_money2 DROP COLUMN dummy;
ALTER TABLE t_money2 ADD COLUMN c money;
INSERT INTO t_money2 (SELECT x,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money,
                             random_float(0.5, -9999999.99,
                                                9999999.99)::numeric::money
                        FROM generate_series(800001,1000000) X);
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

/* int4range */
CREATE TABLE t_int4range1 (id int primary key, dummy text,
                           a int4range, b int4range);
CREATE TABLE t_int4range2 (id int primary key, dummy text,
                           a int4range);
INSERT INTO t_int4range1 (SELECT x, random_text_len(0.5, 20),
                                 random_int4range(0.5),
                                 random_int4range(0.5)
                            FROM generate_series(1,1000000) X);
INSERT INTO t_int4range2 (SELECT x, random_text_len(0.5, 20),
                                 random_int4range(0.5)
                            FROM generate_series(1,800000) X);
ALTER TABLE t_int4range1 DROP COLUMN dummy;
ALTER TABLE t_int4range2 DROP COLUMN dummy;
ALTER TABLE t_int4range2 ADD COLUMN b int4range;
INSERT INTO t_int4range2 (SELECT x, random_int4range(0.5),
                                    random_int4range(0.5)
                            FROM generate_series(800001,1000000) X);

/* int8range */
CREATE TABLE t_int8range1 (id int primary key, dummy text,
                           a int8range, b int8range);
CREATE TABLE t_int8range2 (id int primary key, dummy text,
                           a int8range);
INSERT INTO t_int8range1 (SELECT x, random_text_len(0.5, 20),
                                 random_int8range(0.5),
                                 random_int8range(0.5)
                            FROM generate_series(1,1000000) X);
INSERT INTO t_int8range2 (SELECT x, random_text_len(0.5, 20),
                                 random_int8range(0.5)
                            FROM generate_series(1,800000) X);
ALTER TABLE t_int8range1 DROP COLUMN dummy;
ALTER TABLE t_int8range2 DROP COLUMN dummy;
ALTER TABLE t_int8range2 ADD COLUMN b int8range;
INSERT INTO t_int8range2 (SELECT x, random_int8range(0.5),
                                    random_int8range(0.5)
                            FROM generate_series(800001,1000000) X);

/* tsrange */
CREATE TABLE t_tsrange1 (id int primary key, dummy text,
                         a tsrange, b tsrange);
CREATE TABLE t_tsrange2 (id int primary key, dummy text,
                         a tsrange);
INSERT INTO t_tsrange1 (SELECT x, random_text_len(0.5, 20),
                                 random_tsrange(0.5),
                                 random_tsrange(0.5)
                          FROM generate_series(1,1000000) X);
INSERT INTO t_tsrange2 (SELECT x, random_text_len(0.5, 20),
                                 random_tsrange(0.5)
                          FROM generate_series(1,800000) X);
ALTER TABLE t_tsrange1 DROP COLUMN dummy;
ALTER TABLE t_tsrange2 DROP COLUMN dummy;
ALTER TABLE t_tsrange2 ADD COLUMN b tsrange;
INSERT INTO t_tsrange2 (SELECT x, random_tsrange(0.5),
                                  random_tsrange(0.5)
                          FROM generate_series(800001,1000000) X);

/* tstzrange */
CREATE TABLE t_tstzrange1 (id int primary key, dummy text,
                           a tstzrange, b tstzrange);
CREATE TABLE t_tstzrange2 (id int primary key, dummy text,
                           a tstzrange);
INSERT INTO t_tstzrange1 (SELECT x, random_text_len(0.5, 20),
                                 random_tstzrange(0.5),
                                 random_tstzrange(0.5)
                            FROM generate_series(1,1000000) X);
INSERT INTO t_tstzrange2 (SELECT x, random_text_len(0.5, 20),
                                    random_tstzrange(0.5)
                          FROM generate_series(1,800000) X);
ALTER TABLE t_tstzrange1 DROP COLUMN dummy;
ALTER TABLE t_tstzrange2 DROP COLUMN dummy;
ALTER TABLE t_tstzrange2 ADD COLUMN b tstzrange;
INSERT INTO t_tstzrange2 (SELECT x, random_tstzrange(0.5),
                                    random_tstzrange(0.5)
                            FROM generate_series(800001,1000000) X);

/* daterange */
CREATE TABLE t_daterange1 (id int primary key, dummy text,
                           a daterange, b daterange);
CREATE TABLE t_daterange2 (id int primary key, dummy text,
                           a daterange);
INSERT INTO t_daterange1 (SELECT x, random_text_len(0.5, 20),
                                 random_daterange(0.5),
                                 random_daterange(0.5)
                            FROM generate_series(1,1000000) X);
INSERT INTO t_daterange2 (SELECT x, random_text_len(0.5, 20),
                                    random_daterange(0.5)
                          FROM generate_series(1,800000) X);
ALTER TABLE t_daterange1 DROP COLUMN dummy;
ALTER TABLE t_daterange2 DROP COLUMN dummy;
ALTER TABLE t_daterange2 ADD COLUMN b daterange;
INSERT INTO t_daterange2 (SELECT x, random_daterange(0.5),
                                    random_daterange(0.5)
                            FROM generate_series(800001,1000000) X);

-- Mark TestDB construction completed
CREATE OR REPLACE FUNCTION
public.pgstrom_regression_test_revision()
RETURNS int
AS 'SELECT 20180124'
LANGUAGE 'sql';

COMMIT;
-- vacuum tables
VACUUM ANALYZE;
