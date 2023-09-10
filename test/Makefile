include ../Makefile.common

# 'contrib_regression_pg_strom' as test database
USE_MODULE_DB := 1
MODULE_big := pg_strom

CREATEDB_CMD   := $(shell $(PG_CONFIG) --bindir)/createdb
DROPDB_CMD     := $(shell $(PG_CONFIG) --bindir)/dropdb
PSQL           := $(shell $(PG_CONFIG) --bindir)/psql
PG2ARROW_CMD   := $(shell cd ../arrow-tools; pwd)/pg2arrow
ARROW2CSV_CMD  := $(shell cd ../arrow-tools; pwd)/arrow2csv
DBGEN_SSBM_CMD := $(shell cd ./ssbm; pwd)/dbgen-ssbm
MY_DATA_DIR    := $(shell pwd)/data

REGRESS_DBNAME := contrib_regression_$(MODULE_big)
REGRESS_REVISION := 20230911
REGRESS := --schedule=./parallel_schedule
REGRESS_SETENV = 'PGDATABASE=$(REGRESS_DBNAME)'		\
                 'MY_DATA_DIR=$(MY_DATA_DIR)'		\
                 'PG2ARROW_CMD=$(PG2ARROW_CMD)'		\
                 'ARROW2CSV_CMD=$(ARROW2CSV_CMD)'	\
                 'DBGEN_SSBM_CMD=$(DBGEN_SSBM_CMD)'
REGRESS_OPTS := --inputdir=./$(PG_MAJORVERSION) \
                --outputdir=./$(PG_MAJORVERSION) \
                --encoding=UTF-8 \
                --load-extension=pg_strom \
                --load-extension=plpython3u \
                --use-existing \
                --launcher="env $(REGRESS_SETENV)"
__REGRESS_INIT_SQL := init_hoge.sql
REGRESS_INIT_SQL := $(addprefix init/,$(__REGRESS_INIT_SQL))
REGRESS_PREP := init_dataset pg2arrow arrow2csv dbgen-ssbm
REGRESS_DBCHECK_QUERY := SELECT current_database()
REGRESS_REVISION_QUERY := SELECT pgstrom.regression_testdb_revision()

all: installcheck

init_database:
	test "`$(PSQL) -At -c '$(REGRESS_DBCHECK_QUERY)' $(REGRESS_DBNAME)`" = '$(REGRESS_DBNAME)' || ( \
	  $(DROPDB_CMD) --if-exists $(REGRESS_DBNAME) &&		\
	  $(CREATEDB_CMD) -E UTF-8 -T template0 $(REGRESS_DBNAME) &&	\
	  $(PSQL) $(REGRESS_DBNAME) -c 'CREATE EXTENSION pg_strom' &&	\
	  $(PSQL) $(REGRESS_DBNAME) -c 'ALTER DATABASE $(REGRESS_DBNAME) SET pg_strom.regression_test_mode = on' \
	)

init_dataset: init_database
	test "`$(PSQL) -At -c '$(REGRESS_REVISION_QUERY)' $(REGRESS_DBNAME)`" = '$(REGRESS_REVISION)' || ( \
	  SQLFILE=`mktemp --suffix=.sql` &&				\
	  (cat $(REGRESS_INIT_SQL);					\
	   echo "CREATE OR REPLACE FUNCTION";				\
	   echo "pgstrom.regression_testdb_revision()";			\
	   echo "RETURNS text AS 'SELECT ''$(REGRESS_REVISION)'''";	\
	   echo "LANGUAGE sql;") > $${SQLFILE};				\
	   env $(REGRESS_SETENV) $(PSQL) $(REGRESS_DBNAME) --single-transaction -f $${SQLFILE} \
	  )

drop_database:
	$(DROPDB_CMD) -i $(REGRESS_DBNAME)


pg2arrow:
	make -C ../arrow-tools PG_CONFID=$(PG_CONFIG) pg2arrow

arrow2csv:
	make -C ../arrow-tools PG_CONFID=$(PG_CONFIG) arrow2csv

dbgen-ssbm:
	make -C ssbm

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)



