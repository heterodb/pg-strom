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

CREATE TYPE __pgstrom_device_info AS
  (devid int, name text, value text);

CREATE FUNCTION pgstrom_device_info(int)
  RETURNS SETOF __pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE VIEW pgstrom_shadow_relation AS
  SELECT oid, relname, relkind, pg_relation_size(oid) AS relsize
  FROM pg_class WHERE relnamespace IN
    (SELECT oid FROM pg_namespace WHERE nspname = 'pg_strom');
