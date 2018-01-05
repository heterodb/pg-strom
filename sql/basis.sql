--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- pg_strom installation queries
--
CREATE TYPE pgstrom.__pgstrom_device_info AS (
  id		int4,
  property	text,
  value		text
);
CREATE FUNCTION public.pgstrom_device_info()
  RETURNS SETOF pgstrom.__pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE pgstrom.__pgstrom_device_preserved_meminfo AS (
  device_id int4,
  handle    bytea,
  owner     regrole,
  length    int8
);
CREATE FUNCTION public.pgstrom_device_preserved_meminfo()
  RETURNS SETOF pgstrom.__pgstrom_device_preserved_meminfo
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

--
-- Functions/Languages to support PL/CUDA
--
CREATE FUNCTION pgstrom.plcuda_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','plcuda_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','plcuda_function_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_source(regproc)
  RETURNS text
  AS 'MODULE_PATHNAME','plcuda_function_source'
  LANGUAGE C STRICT;

CREATE LANGUAGE plcuda
  HANDLER pgstrom.plcuda_function_handler
  VALIDATOR pgstrom.plcuda_function_validator;
COMMENT ON LANGUAGE plcuda IS 'PL/CUDA procedural language';

CREATE FUNCTION pgstrom.pltext_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','pltext_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.pltext_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','pltext_function_handler'
  LANGUAGE C STRICT;

CREATE LANGUAGE pltext
  HANDLER pgstrom.pltext_function_handler
  VALIDATOR pgstrom.pltext_function_validator;
COMMENT ON LANGUAGE pltext IS 'PL/Text contents holder';

--
-- Functions related to columnar-cache
--
CREATE FUNCTION pgstrom.ccache_invalidator()
  RETURNS trigger
  AS 'MODULE_PATHNAME','pgstrom_ccache_invalidator'
  LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION public.pgstrom_ccache_enabled(regclass)
RETURNS text
AS $$
DECLARE
  qres RECORD;
BEGIN
  SELECT oid,relnamespace::regnamespace relnsp,relname
    INTO qres
    FROM pg_catalog.pg_class
   WHERE oid = $1;

  EXECUTE format('CREATE TRIGGER __ccache_%s_inval_r '
                 'AFTER INSERT OR UPDATE OR DELETE '
                 'ON %I.%I FOR ROW '
                 'EXECUTE PROCEDURE pgstrom.ccache_invalidator()',
                 qres.oid, qres.relnsp, qres.relname);
  EXECUTE format('CREATE TRIGGER __ccache_%s_inval_s '
                 'AFTER TRUNCATE '
                 'ON %I.%I FOR STATEMENT '
                 'EXECUTE PROCEDURE pgstrom.ccache_invalidator()',
                 qres.oid, qres.relnsp, qres.relname);
  EXECUTE format('ALTER TRIGGER __ccache_%s_inval_r ON %I.%I DEPENDS ON EXTENSION pg_strom',
                 qres.oid, qres.relnsp, qres.relname);
  EXECUTE format('ALTER TRIGGER __ccache_%s_inval_s ON %I.%I DEPENDS ON EXTENSION pg_strom',
                 qres.oid, qres.relnsp, qres.relname);
  EXECUTE format('ALTER TABLE %I.%I ENABLE ALWAYS TRIGGER __ccache_%s_inval_r',
                 qres.relnsp, qres.relname, qres.oid);
  EXECUTE format('ALTER TABLE %I.%I ENABLE ALWAYS TRIGGER __ccache_%s_inval_s',
                 qres.relnsp, qres.relname, qres.oid);
  RETURN 'enabled';
END
$$ LANGUAGE 'plpgsql';

CREATE OR REPLACE FUNCTION public.pgstrom_ccache_disabled(regclass)
RETURNS text
AS $$
DECLARE
  qres RECORD;
BEGIN
  SELECT oid,relnamespace::regnamespace relnsp,relname
    INTO qres
    FROM pg_catalog.pg_class
   WHERE oid = $1;

  EXECUTE format('DROP TRIGGER IF EXISTS __ccache_%s_inval_r ON %I.%I RESTRICT',
                 qres.oid, qres.relnsp, qres.relname);
  EXECUTE format('DROP TRIGGER IF EXISTS __ccache_%s_inval_s ON %I.%I RESTRICT',
                 qres.oid, qres.relnsp, qres.relname);
  RETURN 'disabled';
END
$$ LANGUAGE 'plpgsql';

CREATE TYPE pgstrom.__pgstrom_ccache_info AS (
    database_id  oid,
    table_id     regclass,
    block_nr     int,
    nitems       bigint,
    length       bigint,
    ctime        timestamp with time zone,
    atime        timestamp with time zone
);
CREATE FUNCTION public.pgstrom_ccache_info()
    RETURNS SETOF pgstrom.__pgstrom_ccache_info
    AS 'MODULE_PATHNAME'
    LANGUAGE C STRICT;

CREATE TYPE pgstrom.__pgstrom_ccache_builder_info AS (
    builder_id   int,
    state        text,
    database_id  oid,
    table_id     regclass,
    block_nr     int
);
CREATE FUNCTION public.pgstrom_ccache_builder_info()
    RETURNS SETOF pgstrom.__pgstrom_ccache_builder_info
    AS 'MODULE_PATHNAME'
    LANGUAGE C STRICT;

--
-- Handlers for gstore_fdw extension
--
CREATE FUNCTION pgstrom_gstore_fdw_handler()
  RETURNS fdw_handler
  AS  'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_gstore_fdw_validator(text[],oid)
  RETURNS void
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER gstore_fdw
  HANDLER pgstrom_gstore_fdw_handler
  VALIDATOR pgstrom_gstore_fdw_validator;

CREATE SERVER gstore_fdw
  FOREIGN DATA WRAPPER gstore_fdw;

CREATE TYPE reggstore;
CREATE FUNCTION pgstrom.reggstore_in(cstring)
  RETURNS reggstore
  AS 'MODULE_PATHNAME','pgstrom_reggstore_in'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_out(reggstore)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_reggstore_out'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_recv(internal)
  RETURNS reggstore
  AS 'MODULE_PATHNAME','pgstrom_reggstore_recv'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_send(reggstore)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_reggstore_send'
  LANGUAGE C STRICT IMMUTABLE;
CREATE TYPE public.reggstore
(
  input = pgstrom.reggstore_in,
  output = pgstrom.reggstore_out,
  receive = pgstrom.reggstore_recv,
  send = pgstrom.reggstore_send,
  like = pg_catalog.oid
);

CREATE CAST (reggstore AS oid)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (oid AS reggstore)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (reggstore AS integer)
  WITHOUT FUNCTION AS ASSIGNMENT;
CREATE CAST (reggstore AS bigint)
  WITH FUNCTION pg_catalog.int8(oid) AS ASSIGNMENT;
CREATE CAST (integer AS reggstore)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (smallint AS reggstore)
  WITH FUNCTION pg_catalog.int4(smallint) AS IMPLICIT;
CREATE CAST (bigint AS reggstore)
  WITH FUNCTION oid(bigint) AS IMPLICIT;

CREATE FUNCTION public.gstore_export_ipchandle(reggstore)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_gstore_export_ipchandle'
  LANGUAGE C STRICT;

CREATE FUNCTION public.lo_export_ipchandle(oid)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_lo_export_ipchandle'
  LANGUAGE C STRICT;

CREATE FUNCTION public.lo_import_ipchandle(bytea)
  RETURNS oid
  AS 'MODULE_PATHNAME','pgstrom_lo_import_ipchandle'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_format(reggstore)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_format'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_nitems(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_nitems'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_nattrs(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_nattrs'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_rawsize(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_rawsize'
  LANGUAGE C STRICT;

--
-- Type re-interpretation routines
--
CREATE FUNCTION pg_catalog.float4_as_int4(float4)
  RETURNS int4
  AS 'MODULE_PATHNAME','float4_as_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.int4_as_float4(int4)
  RETURNS float4
  AS 'MODULE_PATHNAME','int4_as_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.float8_as_int8(float8)
  RETURNS int8
  AS 'MODULE_PATHNAME','float8_as_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.int8_as_float8(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME','int8_as_float8'
  LANGUAGE C STRICT;
