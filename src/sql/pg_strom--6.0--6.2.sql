---
--- PG-Strom v6.0 -> v6.2 (minor changes)
---

---
--- A function to get parquet cache information
---
CREATE FUNCTION pgstrom.parquet_cache_info()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_parquet_cache_info'
  LANGUAGE C STRICT;

---
--- GPU Cache is removed at v6.2
---
DROP FUNCTION IF EXISTS pgstrom.gpucache_sync_trigger();
DROP FUNCTION IF EXISTS pgstrom.gpucache_apply_redo(regclass);
DROP FUNCTION IF EXISTS pgstrom.gpucache_compaction(regclass);
DROP FUNCTION IF EXISTS pgstrom.gpucache_recovery(regclass);
DROP TYPE IF EXISTS pgstrom.__pgstrom_gpucache_info_t CASCADE;

