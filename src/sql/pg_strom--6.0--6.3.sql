---
--- PG-Strom v6.2 -> v6.3 (minor changes)
---

---
--- A function to get parquet cache information
---
CREATE FUNCTION pgstrom.parquet_cache_info()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_parquet_cache_info'
  LANGUAGE C STRICT;
