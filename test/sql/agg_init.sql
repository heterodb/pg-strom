--#
--#       Initialization Script for Gpu PreAggregate TestCases
--#



select pg_sleep(5); -- wait until pg_strom background worker started.

--# ignore NOTICE or WARNING messages.
set client_min_messages to error;

--# PG-strom EXTENSION will load.
CREATE EXTENSION IF NOT EXISTS pg_strom;


--# CREATE testdata-table,and add comment every column.
DROP TABLE IF EXISTS strom_test CASCADE;
CREATE TABLE strom_test (
	id integer,
	key integer,
	smlint_x  smallint,
	integer_x integer,
	bigint_x  bigint,
	real_x    real,
	float_x   float,
	nume_x    numeric,
	smlsrl_x  smallserial,
	serial_x  serial,
    bigsrl_x  bigserial
);
COMMENT ON COLUMN strom_test.smlint_x  IS '-32768 to 32767';
COMMENT ON COLUMN strom_test.integer_x IS '-2147483648 to 2147483647';
COMMENT ON COLUMN strom_test.bigint_x  IS '-9223372036854775808 to 9223372036854775807';
COMMENT ON COLUMN strom_test.real_x    IS '6 decimal digits precision';
COMMENT ON COLUMN strom_test.float_x   IS '15 decimal digits precision';
COMMENT ON COLUMN strom_test.nume_x    IS 'up to 131072 digits before the decimal point; up to 16383 digits after the decimal point';
COMMENT ON COLUMN strom_test.smlsrl_x  IS '-32768 to 32767';
COMMENT ON COLUMN strom_test.serial_x  IS '-2147483648 to 2147483647';
COMMENT ON COLUMN strom_test.bigsrl_x  IS '-9223372036854775808 to 9223372036854775807';


--# Setting a seed for same random pattern.
SELECT setseed(0);

--#
--# [all +random]
--# 1 <= id <= 10000,all columns have positive(+) random numbers.
INSERT INTO strom_test SELECT
			generate_series(      1,10000),
			generate_series(  1,10),
			case when random() > 0.95 then null else random()*32767/1000 end,
			case when random() > 0.95 then null else random()*2147483647/1000 end,
			case when random() > 0.95 then null else random()*9223372036854775807/1000 end,
			case when random() > 0.95 then null else round(random()::numeric,4) end,
			case when random() > 0.95 then null else round(random()::numeric,13) end,
			case when random() > 0.95 then null else random()::numeric end,
			random()*32767/1000 ,
			random()*2147483647/1000 ,
			random()*9223372036854775807/1000 ; 

--#
--# [all -random]
--# 10001 <= id <= 20000,all columns have negative(-) random numbers.
INSERT INTO strom_test SELECT
			generate_series(10001,20000),
			generate_series(11,20),
			case when random() > 0.95 then null else random()*-32767/1000 end,
			case when random() > 0.95 then null else random()*-2147483647/1000 end,
			case when random() > 0.95 then null else random()*-9223372036854775807/1000 end,
			case when random() > 0.95 then null else round((random()*-1)::numeric,4) end,
			case when random() > 0.95 then null else round((random()*-1)::numeric,13) end,
			case when random() > 0.95 then null else (random()*-1)::numeric end,
			random()*-32767/1000 ,
			random()*-2147483647/1000 ,
			random()*-9223372036854775807/1000 ;


--#
--# [all +-random]
--# 20001 <= id <= 30000,all columns have positive(+) or negative(-) random numbers.
INSERT INTO strom_test SELECT
			generate_series(20001,30000),
			generate_series(21,30),
			case when random() > 0.95 then null else (random()*2-1)*32767/1000 end,
			case when random() > 0.95 then null else (random()*2-1)*2147483647/1000 end,
			case when random() > 0.95 then null else (random()*2-1)*9223372036854775807/1000 end,
			case when random() > 0.95 then null else round((random()*2-1)::numeric,4) end,
			case when random() > 0.95 then null else round((random()*2-1)::numeric,13) end,
			case when random() > 0.95 then null else (random()*2-1)::numeric end,
			(random()*2-1)*32767/1000 ,
			(random()*2-1)*2147483647/1000 ,
			(random()*2-1)*9223372036854775807/1000 ;
--#
--# [all null]
--# 30001 <= id <= 40000,all columns(but expect serial type) have NULL numbers.
--# serial columns have 0.
INSERT INTO strom_test SELECT
			generate_series(30001,40000),
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			0,
			0,
			0;



--# create no data table.
DROP TABLE IF EXISTS strom_zero_test;
CREATE TABLE strom_zero_test AS SELECT * FROM strom_test WITH NO DATA;

--# create overflowed table.
DROP TABLE IF EXISTS strom_overflow_test;
CREATE TABLE strom_overflow_test (
	id integer,
	key integer,
	smlint_x  smallint,
	integer_x integer,
	bigint_x  bigint,
	real_x    real,
	float_x   float,
	nume_x    numeric,
	smlsrl_x  smallserial,
	serial_x  serial,
    bigsrl_x  bigserial
);
COMMENT ON COLUMN strom_overflow_test.smlint_x  IS '-32768 to 32767';
COMMENT ON COLUMN strom_overflow_test.integer_x IS '-2147483648 to 2147483647';
COMMENT ON COLUMN strom_overflow_test.bigint_x  IS '-9223372036854775808 to 9223372036854775807';
COMMENT ON COLUMN strom_overflow_test.real_x    IS '6 decimal digits precision';
COMMENT ON COLUMN strom_overflow_test.float_x   IS '15 decimal digits precision';
COMMENT ON COLUMN strom_overflow_test.nume_x    IS 'up to 131072 digits before the decimal point; up to 16383 digits after the decimal point';
COMMENT ON COLUMN strom_overflow_test.smlsrl_x  IS '-32768 to 32767';
COMMENT ON COLUMN strom_overflow_test.serial_x  IS '-2147483648 to 2147483647';
COMMENT ON COLUMN strom_overflow_test.bigsrl_x  IS '-9223372036854775808 to 9223372036854775807';

SELECT setseed(0);

--#
--# [all +random]
--# 1 <= id <= 10000,all columns have positive(+) random numbers.
INSERT INTO strom_overflow_test SELECT
			generate_series(      1,10000),
			generate_series(  1,10),
			case when random() > 0.95 then null else 32767 end,
			case when random() > 0.95 then null else 2147483647 end,
			case when random() > 0.95 then null else 9223372036854775807 end,
			case when random() > 0.95 then null else (1.0e38)::real end,
			case when random() > 0.95 then null else (1.0e308)::float end,
			case when random() > 0.95 then null else floor(random()*1000000000000000000000) end,
			random()*32767 ,
			random()*2147483647 ,
			random()*9223372036854775807 ; 
--#
--# [all -random]
--# 10001 <= id <= 20000,all columns have negative(-) random numbers.
INSERT INTO strom_overflow_test SELECT
			generate_series(10001,20000),
			generate_series(11,20),
			case when random() > 0.95 then null else -32768 end,
			case when random() > 0.95 then null else -2147483648 end,
			case when random() > 0.95 then null else -9223372036854775808 end,
			case when random() > 0.95 then null else (-1.0e38)::real end,
			case when random() > 0.95 then null else (-1.0e308)::float end,
			case when random() > 0.95 then null else floor(random()*1000000000000000000000)*-1 end,
			random()*-32767 ,
			random()*-2147483647 ,
			random()*-9223372036854775807 ;

--#
--# [all +-random]
--# 20001 <= id <= 30000,all columns have positive(+) or negative(-) random numbers.
INSERT INTO strom_overflow_test SELECT
			generate_series(20001,30000),
			generate_series(21,30),
			case when random() > 0.95 then null else (random()*2-1)*32767 end,
			case when random() > 0.95 then null else (random()*2-1)*2147483647 end,
			case when random() > 0.95 then null else (random()*2-1)*9223372036854775807 end,
			case when random() > 0.95 then null else ((random()*2-1)*1.0e38)::real end,
			case when random() > 0.95 then null else ((random()*2-1)*1.0e308)::float end,
			case when random() > 0.95 then null else floor((random()*2-1)*1000000000000000000000) end,
			(random()*2-1)*32767 ,
			(random()*2-1)*2147483647 ,
			(random()*2-1)*9223372036854775807 ;

--#
--# [all null]
--# 30001 <= id <= 40000,all columns(but expect serial type) have NULL numbers.
--# serial columns have 0.
INSERT INTO strom_overflow_test SELECT
			generate_series(30001,40000),
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			0,
			0,
			0;







--# create unique index for materialized view.
CREATE UNIQUE INDEX strom_test_id_unq_idx on strom_test(id);
CREATE INDEX strom_test_key_idx on strom_test(key);

DROP MATERIALIZED VIEW IF EXISTS strom_mix;
CREATE MATERIALIZED VIEW strom_mix AS 
SELECT x.id,x.key,
x.smlint_x AS smlint_x,
y.smlint_x AS smlint_y,
z.smlint_x AS smlint_z,

x.integer_x AS integer_x,
y.integer_x AS integer_y,
z.integer_x AS integer_z,

x.bigint_x AS bigint_x,
y.bigint_x AS bigint_y,
z.bigint_x AS bigint_z,

x.real_x AS real_x,
y.real_x AS real_y,
z.real_x AS real_z,

x.float_x AS float_x,
y.float_x AS float_y,
z.float_x AS float_z,

x.nume_x AS nume_x,
y.nume_x AS nume_y,
z.nume_x AS nume_z,

x.smlsrl_x AS smlsrl_x,
y.smlsrl_x AS smlsrl_y,
z.smlsrl_x AS smlsrl_z,

x.serial_x AS serial_x,
y.serial_x AS serial_y,
z.serial_x AS serial_z,

x.bigsrl_x AS bigsrl_x,
y.bigsrl_x AS bigsrl_y,
z.bigsrl_x AS bigsrl_z

FROM
(SELECT * FROM strom_test WHERE id<=10000) AS x,
(SELECT id-10000 AS id,key-10 AS key,smlint_x,integer_x,bigint_x,real_x,float_x,nume_x,smlsrl_x,serial_x,bigsrl_x FROM strom_test WHERE key between 11 AND 20) AS y,
(SELECT id-20000 AS id,key-20 AS key,smlint_x,integer_x,bigint_x,real_x,float_x,nume_x,smlsrl_x,serial_x,bigsrl_x FROM strom_test WHERE key between 21 AND 30) AS z

WHERE x.id=y.id
AND y.id=z.id
AND z.id=x.id;
--# drop key index.
DROP INDEX strom_test_key_idx;

