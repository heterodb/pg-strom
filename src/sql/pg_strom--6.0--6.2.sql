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
--- GPU Cache is re-designed at v6.2
---
DROP FUNCTION IF EXISTS pgstrom.gpucache_apply_redo(regclass);
CREATE FUNCTION pgstrom.gpucache_apply_redo()
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gpucache_apply_redo'
  LANGUAGE C STRICT;

DROP TYPE IF EXISTS pgstrom.__pgstrom_gpucache_info_t CASCADE;
CREATE FUNCTION pgstrom.gpucache_info()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_gpucache_info'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.gpucache_info(regclass)
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_gpucache_info_one'
  LANGUAGE C STRICT;
