--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.3
-- Dumped by pg_dump version 9.5.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: customer; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE customer (
    c_custkey numeric NOT NULL,
    c_name character varying(25),
    c_address character varying(25),
    c_city character(10),
    c_nation character(15),
    c_region character(12),
    c_phone character(15),
    c_mktsegment character(10)
);


ALTER TABLE customer OWNER TO postgres;

--
-- Name: date1; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE date1 (
    d_datekey integer NOT NULL,
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


ALTER TABLE date1 OWNER TO postgres;

--
-- Name: lineorder; Type: TABLE; Schema: public; Owner: postgres
--

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


ALTER TABLE lineorder OWNER TO postgres;

--
-- Name: part; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE part (
    p_partkey integer NOT NULL,
    p_name character varying(22),
    p_mfgr character(6),
    p_category character(7),
    p_brand1 character(9),
    p_color character varying(11),
    p_type character varying(25),
    p_size numeric,
    p_container character(10)
);


ALTER TABLE part OWNER TO postgres;

--
-- Name: supplier; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE supplier (
    s_suppkey numeric NOT NULL,
    s_name character(25),
    s_address character varying(25),
    s_city character(10),
    s_nation character(15),
    s_region character(12),
    s_phone character(15)
);


ALTER TABLE supplier OWNER TO postgres;

--
-- Name: customer_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY customer
    ADD CONSTRAINT customer_pkey PRIMARY KEY (c_custkey);


--
-- Name: date1_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY date1
    ADD CONSTRAINT date1_pkey PRIMARY KEY (d_datekey);


--
-- Name: part_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY part
    ADD CONSTRAINT part_pkey PRIMARY KEY (p_partkey);


--
-- Name: supplier_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY supplier
    ADD CONSTRAINT supplier_pkey PRIMARY KEY (s_suppkey);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

