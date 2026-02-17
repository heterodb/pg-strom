--
-- PostgreSQL database dump
--

--
-- Name: customer; Type: TABLE; Schema: public; Owner: postgres
--
CREATE TABLE customer (
    c_custkey integer PRIMARY KEY,
    c_name character varying(25),
    c_address character varying(25),
    c_city character(10),
    c_nation character(15),
    c_region character(12),
    c_phone character(15),
    c_mktsegment character(10)
);

--
-- Name: date1; Type: TABLE; Schema: public; Owner: postgres
--
CREATE TABLE date1 (
    d_datekey integer PRIMARY KEY,
    d_date character(18),
    d_dayofweek character(12),
    d_month character(9),
    d_year integer,
    d_yearmonthnum integer,
    d_yearmonth character(7),
    d_daynuminweek integer,
    d_daynuminmonth integer,
    d_daynuminyear integer,
    d_monthnuminyear integer,
    d_weeknuminyear integer,
    d_sellingseason character(12),
    d_lastdayinweekfl character(1),
    d_lastdayinmonthfl character(1),
    d_holidayfl character(1),
    d_weekdayfl character(1)
);

--
-- Name: lineorder; Type: TABLE; Schema: public; Owner: postgres
--
CREATE TABLE lineorder (
    lo_orderkey bigint,
    lo_linenumber integer,
    lo_custkey integer,
    lo_partkey integer,
    lo_suppkey integer,
    lo_orderdate integer,
    lo_orderpriority character(15),
    lo_shippriority character(1),
    lo_quantity integer,
    lo_extendedprice integer,
    lo_ordertotalprice integer,
    lo_discount integer,
    lo_revenue integer,
    lo_supplycost integer,
    lo_tax integer,
    lo_commit_date character(8),
    lo_shipmode character(10)
);

--
-- Name: part; Type: TABLE; Schema: public; Owner: postgres
--
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

--
-- Name: supplier; Type: TABLE; Schema: public; Owner: postgres
--
CREATE TABLE supplier (
    s_suppkey integer PRIMARY KEY,
    s_name character(25),
    s_address character varying(25),
    s_city character(10),
    s_nation character(15),
    s_region character(12),
    s_phone character(15)
);
