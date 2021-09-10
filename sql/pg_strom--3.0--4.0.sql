---
--- GPU Cache Functions
---
CREATE FUNCTION pgstrom.gpucache_recovery(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpucache_recovery'
  LANGUAGE C CALLED ON NULL INPUT;

---
--- Portable Shared Memory
---
CREATE FUNCTION pgstrom.shared_buffer_info()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_shared_buffer_info'
  LANGUAGE C STRICT;

---
--- Deprecated Functions
---
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_unpin_gpu_buffer(text);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_put_gpu_buffer(text);
