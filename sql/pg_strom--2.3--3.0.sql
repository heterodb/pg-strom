--
-- Handlers of gstore_fdw extension (renew at v3.0)
--
CREATE FUNCTION pgstrom.gstore_fdw_handler()
  RETURNS fdw_handler
  AS  'MODULE_PATHNAME','pgstrom_gstore_fdw_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gstore_fdw_validator(text[],oid)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_validator'
  LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER gstore_fdw
  HANDLER   pgstrom.gstore_fdw_handler
  VALIDATOR pgstrom.gstore_fdw_validator;

CREATE SERVER gstore_fdw
  FOREIGN DATA WRAPPER gstore_fdw;

CREATE OR REPLACE FUNCTION pgstrom.gstore_fdw_precheck_schema()
  RETURNS event_trigger
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_precheck_schema'
  LANGUAGE C STRICT;
CREATE EVENT TRIGGER pgstrom_gstore_fdw_precheck_schema
  ON ddl_command_end
WHEN tag IN ('CREATE FOREIGN TABLE',
             'ALTER FOREIGN TABLE');

CREATE FUNCTION public.gstore_fdw_synchronize(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_synchronize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_compaction(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_compaction'
  LANGUAGE C STRICT;
