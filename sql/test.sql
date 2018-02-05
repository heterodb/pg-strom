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

CREATE FUNCTION pgstrom.random_timetz(float=0.0,
                                      time=null,
                                      time=null)
  RETURNS timetz
  AS 'MODULE_PATHNAME','pgstrom_random_timetz'
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

CREATE FUNCTION pgstrom.random_inet(float=0.0,
                                    inet=null)
  RETURNS inet
  AS 'MODULE_PATHNAME','pgstrom_random_inet'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_text(float=0.0,
                                    text=null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_random_text'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_text_len(float=0.0,
                                        int=null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_random_text_length'
  LANGUAGE C CALLED ON NULL INPUT;
