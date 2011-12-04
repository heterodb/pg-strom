--
-- pg_strom installation queries
--
CREATE FUNCTION pgstrom_fdw_handler()
  RETURNS fdw_handler
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_fdw_validator(text[], oid)
  RETURNS void
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER pg_strom
  HANDLER pgstrom_fdw_handler
  VALIDATOR pgstrom_fdw_validator;

CREATE SERVER pg_strom FOREIGN DATA WRAPPER pg_strom;

CREATE FUNCTION pgstrom_data_load(regclass, regclass)
  RETURNS bool
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_data_clear(regclass)
  RETURNS bool
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_data_compaction(regclass)
  RETURNS bool
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;
