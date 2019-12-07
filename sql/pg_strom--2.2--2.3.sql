/* pg_strom--2.2--2.3.sql */
CREATE OR REPLACE FUNCTION pgstrom.random_setseed(int)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_random_setseed'
  LANGUAGE C STRICT;
