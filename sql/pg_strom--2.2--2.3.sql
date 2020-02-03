/* pg_strom--2.2--2.3.sql */
CREATE OR REPLACE FUNCTION pgstrom.random_setseed(int)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_random_setseed'
  LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pgstrom.arrow_fdw_precheck_schema()
  RETURNS event_trigger
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_precheck_schema'
  LANGUAGE C STRICT;

CREATE EVENT TRIGGER pgstrom_arrow_fdw_precheck_schema
    ON ddl_command_end
  WHEN tag IN ('CREATE FOREIGN TABLE',
               'ALTER FOREIGN TABLE')
EXECUTE FUNCTION pgstrom.arrow_fdw_precheck_schema();

CREATE OR REPLACE FUNCTION pgstrom.arrow_fdw_truncate(regclass)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_truncate'
  LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_export_cupy(regclass, text[] = null, int = null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_export_cupy'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_unload_gpu(text)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_unload_gpu'
  LANGUAGE C STRICT;
