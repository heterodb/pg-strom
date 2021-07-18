---
--- Deprecated Functions
---
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[], int);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_unpin_gpu_buffer(text);
DROP FUNCTION IF EXISTS pgstrom.arrow_fdw_put_gpu_buffer(text);
