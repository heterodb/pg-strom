--
-- DDL for Star Schema Benchmark Test
--

CREATE TABLE customer (
    c_custkey numeric PRIMARY KEY,
    c_name character varying(25),
    c_address character varying(25),
    c_city character(10),
    c_nation character(15),
    c_region character(12),
    c_phone character(15),
    c_mktsegment character(10)
);

CREATE TABLE date1 (
    d_datekey integer PRIMARY KEY,
    d_date character(18),
    d_dayofweek character(12),
    d_month character(9),
    d_year integer,
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

CREATE TABLE lineorder (
    lo_orderkey numeric,
    lo_linenumber integer,
    lo_custkey numeric,
    lo_partkey integer,
    lo_suppkey numeric,
    lo_orderdate integer,
    lo_orderpriority character(15),
    lo_shippriority character(1),
    lo_quantity numeric,
    lo_extendedprice numeric,
    lo_ordertotalprice numeric,
    lo_discount numeric,
    lo_revenue numeric,
    lo_supplycost numeric,
    lo_tax numeric,
    lo_commit_date character(8),
    lo_shipmode character(10)
);

CREATE TABLE part (
    p_partkey integer PRIMARY KEY,
    p_name character varying(22),
    p_mfgr character(6),
    p_category character(7),
    p_brand1 character(9),
    p_color character varying(11),
    p_type character varying(25),
    p_size numeric,
    p_container character(10)
);

CREATE TABLE supplier (
    s_suppkey numeric PRIMARY KEY,
    s_name character(25),
    s_address character varying(25),
    s_city character(10),
    s_nation character(15),
    s_region character(12),
    s_phone character(15)
);
