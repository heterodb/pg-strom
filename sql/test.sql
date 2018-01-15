--
-- SQL functions to support PG-Strom regression test
--
CREATE FUNCTION pgstrom.random_int(float=0.0,      -- NULL ratio (%)
                                   bigint=null,    -- lower bound
                                   bigint=null)    -- upper bound
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_random_int'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_float(float=0.0,
                                     float=null,
                                     float=null)
  RETURNS float
  AS 'MODULE_PATHNAME','pgstrom_random_float'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_date(float=0.0,
                                    date=null,
                                    date=null)
  RETURNS date
  AS 'MODULE_PATHNAME','pgstrom_random_date'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_time(float=0.0,
                                    time=null,
                                    time=null)
  RETURNS time
  AS 'MODULE_PATHNAME','pgstrom_random_time'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_timestamp(float=0.0,
                                         timestamp=null,
                                         timestamp=null)
  RETURNS timestamp
  AS 'MODULE_PATHNAME','pgstrom_random_timestamp'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_macaddr(float=0.0,
                                       macaddr=null,
                                       macaddr=null)
  RETURNS macaddr
  AS 'MODULE_PATHNAME','pgstrom_random_macaddr'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_inet4(float=0.0,
                                     inet=null)
  RETURNS inet
  AS 'MODULE_PATHNAME','pgstrom_random_inet4'
  LANGUAGE C CALLED ON NULL INPUT;
