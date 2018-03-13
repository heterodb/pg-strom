--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- Functions for device properties
--
CREATE TYPE pgstrom.__pgstrom_device_info AS (
  device_nr     int,
  aindex        int,
  attribute     text,
  value         text
);
CREATE FUNCTION pgstrom.pgstrom_device_info()
  RETURNS SETOF pgstrom.__pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.device_info AS
  SELECT * FROM pgstrom.pgstrom_device_info();

CREATE FUNCTION public.gpu_device_name(int = 0)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gpu_device_name'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_global_memsize(int = 0)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpu_global_memsize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_max_blocksize(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_max_blocksize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_warp_size(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_warp_size'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_max_shared_memory_perblock(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_max_shared_memory_perblock'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_registers_perblock(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_registers_perblock'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_multiptocessors(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_multiptocessors'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_cuda_cores(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_cuda_cores'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_cc_major(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_cc_major'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_cc_minor(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_cc_minor'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_pci_id(int = 0)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gpu_pci_id'
  LANGUAGE C STRICT;

--
-- Functions for system internal state
--
CREATE TYPE pgstrom.__pgstrom_device_preserved_meminfo AS (
  device_nr int4,
  handle    bytea,
  owner     regrole,
  length    int8,
  ctime     timestamp with time zone
);
CREATE FUNCTION pgstrom.pgstrom_device_preserved_meminfo()
  RETURNS SETOF pgstrom.__pgstrom_device_preserved_meminfo
  AS 'MODULE_PATHNAME'
  LANGUAGE C VOLATILE;
CREATE VIEW pgstrom.device_preserved_meminfo
  AS SELECT * FROM pgstrom.pgstrom_device_preserved_meminfo();

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

CREATE FUNCTION pg_catalog.plcuda_kernel_max_blocksz()
  RETURNS int
  AS 'MODULE_PATHNAME','plcuda_kernel_max_blocksz'
  LANGUAGE C VOLATILE;

CREATE FUNCTION pg_catalog.plcuda_kernel_static_shmsz()
  RETURNS int
  AS 'MODULE_PATHNAME','plcuda_kernel_static_shmsz'
  LANGUAGE C VOLATILE;

CREATE FUNCTION pg_catalog.plcuda_kernel_dynamic_shmsz()
  RETURNS int
  AS 'MODULE_PATHNAME','plcuda_kernel_dynamic_shmsz'
  LANGUAGE C VOLATILE;

CREATE FUNCTION pg_catalog.plcuda_kernel_const_memsz()
  RETURNS int
  AS 'MODULE_PATHNAME','plcuda_kernel_const_memsz'
  LANGUAGE C VOLATILE;

CREATE FUNCTION pg_catalog.plcuda_kernel_local_memsz()
  RETURNS int
  AS 'MODULE_PATHNAME','plcuda_kernel_local_memsz'
  LANGUAGE C VOLATILE;

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
CREATE FUNCTION pgstrom.pgstrom_ccache_info()
    RETURNS SETOF pgstrom.__pgstrom_ccache_info
    AS 'MODULE_PATHNAME'
    LANGUAGE C STRICT;
CREATE VIEW pgstrom.ccache_info AS
    SELECT * FROM pgstrom.pgstrom_ccache_info();

CREATE TYPE pgstrom.__pgstrom_ccache_builder_info AS (
    builder_id   int,
    state        text,
    database_id  oid,
    table_id     regclass,
    block_nr     int
);
CREATE FUNCTION pgstrom.pgstrom_ccache_builder_info()
    RETURNS SETOF pgstrom.__pgstrom_ccache_builder_info
    AS 'MODULE_PATHNAME'
    LANGUAGE C STRICT;
CREATE VIEW pgstrom.ccache_builder_info AS
    SELECT * FROM pgstrom.pgstrom_ccache_builder_info();

CREATE FUNCTION public.pgstrom_ccache_prewarm(regclass)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_ccache_prewarm'
  LANGUAGE C STRICT;

--
-- Handlers for gstore_fdw extension
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

CREATE TYPE public.reggstore;
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

CREATE FUNCTION public.gstore_export_ipchandle(reggstore)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_gstore_export_ipchandle'
  LANGUAGE C;

CREATE TYPE pgstrom.__gstore_fdw_chunk_info AS (
  database_oid	oid,
  table_oid		oid,
  revision		int,
  xmin			xid,
  xmax			xid,
  pinning		int,
  format		text,
  rawsize		bigint,
  nitems		bigint
);
CREATE FUNCTION pgstrom.gstore_fdw_chunk_info()
  RETURNS SETOF pgstrom.__gstore_fdw_chunk_info
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_chunk_info'
  LANGUAGE C VOLATILE;
CREATE VIEW pgstrom.gstore_fdw_chunk_info AS
  SELECT * FROM pgstrom.gstore_fdw_chunk_info();

CREATE FUNCTION public.lo_import_gpu(int, bytea, bigint, bigint, oid=0)
  RETURNS oid
  AS 'MODULE_PATHNAME','pgstrom_lo_import_gpu'
  LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION public.lo_export_gpu(oid, int, bytea, bigint, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_lo_export_gpu'
  LANGUAGE C STRICT VOLATILE;

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

--
-- Function to validate commercial license
--
CREATE FUNCTION pgstrom.license_validation()
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_license_validation'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.license_query()
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_license_query'
  LANGUAGE C STRICT;
