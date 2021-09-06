---
--- GPU Cache Functions
---
CREATE FUNCTION pgstrom.gpucache_recovery(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpucache_recovery'
  LANGUAGE C CALLED ON NULL INPUT;

---
--- Information Schema
---
CREATE TYPE pgstrom.__cuda_program_info AS (
  program_id    bigint,
  refcnt        int,
  crc32         int,
  target_cc     int,
  status        text,
  extra_flags   int,
  kern_define   text,
  kern_source   text,
  ptx_image     text,
  error_code    int,
  error_msg     text
);
CREATE FUNCTION pgstrom.cuda_program_info()
  RETURNS SETOF pgstrom.__cuda_program_info
  AS 'MODULE_PATHNAME','pgstrom_cuda_program_info'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.cuda_program_info AS
  SELECT * FROM pgstrom.cuda_program_info();


---
--- Deprecated Functions
---
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_unpin_gpu_buffer(text);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_put_gpu_buffer(text);
