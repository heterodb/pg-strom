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

CREATE FUNCTION pgstrom_data_load(regclass, regclass,int4)
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

CREATE TYPE __pgstrom_debug_info AS (key text, value text);
CREATE FUNCTION pgstrom_debug_info()
  RETURNS SETOF __pgstrom_debug_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_device_properties AS
  (index int, name text, major int, minor int,
   proc_nums int, proc_warp_size int, proc_clock int,
   global_mem_size int8, global_mem_width int, global_mem_clock int,
   shared_mem_size int, L2_cache_size int, const_mem_size int,
   max_block_dim int[], max_grid_dim int[],
   integrated bool, unified_addr bool, concurrent_kernel bool);
CREATE FUNCTION pgstrom_device_properties()
  RETURNS SETOF __pgstrom_device_properties
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE VIEW pgstrom_device_summary AS
  SELECT index, name, major || ''.'' || minor AS capability,
         proc_nums, proc_warp_size, proc_clock,
         global_mem_size, global_mem_width, global_mem_clock,
         shared_mem_size, L2_cache_size, const_mem_size
         FROM pgstrom_device_properties();
