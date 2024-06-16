--
-- PostgreSQL database dump
--

--
-- Name: customer; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE customer (
    c_custkey integer NOT NULL,
    c_name character varying(25) NOT NULL,
    c_address character varying(40) NOT NULL,
    c_nationkey integer NOT NULL,
    c_phone character(15) NOT NULL,
    c_acctbal numeric(15,2) NOT NULL,
    c_mktsegment character(10) NOT NULL,
    c_comment character varying(117) NOT NULL
);

--
-- Name: lineitem; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE lineitem (
    l_orderkey integer NOT NULL,
    l_partkey integer NOT NULL,
    l_suppkey integer NOT NULL,
    l_linenumber integer NOT NULL,
    l_quantity numeric(15,2) NOT NULL,
    l_extendedprice numeric(15,2) NOT NULL,
    l_discount numeric(15,2) NOT NULL,
    l_tax numeric(15,2) NOT NULL,
    l_returnflag character(1) NOT NULL,
    l_linestatus character(1) NOT NULL,
    l_shipdate date NOT NULL,
    l_commitdate date NOT NULL,
    l_receiptdate date NOT NULL,
    l_shipinstruct character(25) NOT NULL,
    l_shipmode character(10) NOT NULL,
    l_comment character varying(44) NOT NULL
);

--
-- Name: nation; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE nation (
    n_nationkey integer NOT NULL,
    n_name character(25) NOT NULL,
    n_regionkey integer NOT NULL,
    n_comment character varying(152)
);

--
-- Name: orders; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE orders (
    o_orderkey integer NOT NULL,
    o_custkey integer NOT NULL,
    o_orderstatus character(1) NOT NULL,
    o_totalprice numeric(15,2) NOT NULL,
    o_orderdate date NOT NULL,
    o_orderpriority character(15) NOT NULL,
    o_clerk character(15) NOT NULL,
    o_shippriority integer NOT NULL,
    o_comment character varying(79) NOT NULL
);

--
-- Name: part; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE part (
    p_partkey integer NOT NULL,
    p_name character varying(55) NOT NULL,
    p_mfgr character(25) NOT NULL,
    p_brand character(10) NOT NULL,
    p_type character varying(25) NOT NULL,
    p_size integer NOT NULL,
    p_container character(10) NOT NULL,
    p_retailprice numeric(15,2) NOT NULL,
    p_comment character varying(23) NOT NULL
);

--
-- Name: partsupp; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE partsupp (
    ps_partkey integer NOT NULL,
    ps_suppkey integer NOT NULL,
    ps_availqty integer NOT NULL,
    ps_supplycost numeric(15,2) NOT NULL,
    ps_comment character varying(199) NOT NULL
);

--
-- Name: region; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE region (
    r_regionkey integer NOT NULL,
    r_name character(25) NOT NULL,
    r_comment character varying(152)
);

--
-- Name: supplier; Type: TABLE; Schema: public; Owner: kaigai
--
CREATE TABLE supplier (
    s_suppkey integer NOT NULL,
    s_name character(25) NOT NULL,
    s_address character varying(40) NOT NULL,
    s_nationkey integer NOT NULL,
    s_phone character(15) NOT NULL,
    s_acctbal numeric(15,2) NOT NULL,
    s_comment character varying(101) NOT NULL
);
--
-- PostgreSQL database dump complete
--

