include ../Makefile.common

# 'contrib_regression_pg_strom' as test database
USE_MODULE_DB  := 1
MODULE_big     := pg_strom

CREATEDB_CMD   := $(shell $(PG_CONFIG) --bindir)/createdb
DROPDB_CMD     := $(shell $(PG_CONFIG) --bindir)/dropdb
PSQL           := $(shell $(PG_CONFIG) --bindir)/psql
PG2ARROW_CMD   := $(shell realpath ../arrow-tools/pg2arrow)
ARROW2CSV_CMD  := $(shell realpath ../arrow-tools/arrow2csv)
DBGEN_SSBM_CMD := $(shell realpath ./ssbm/dbgen-ssbm)
ARROW_TEST_DATA_DIR    := $(shell pwd)/data

REGRESS_DBNAME      := contrib_regression_$(MODULE_big)
REGRESS_REVISION    := 20240527
REGRESS_DB_REVISION := $(shell $(PSQL) $(REGRESS_DBNAME) -At -c 'SELECT pgstrom.regression_testdb_revision()')
REGRESS        := --schedule=./parallel_schedule
REGRESS_SETENV := 'PGDATABASE=$(REGRESS_DBNAME)'               \
                  'ARROW_TEST_DATA_DIR=$(ARROW_TEST_DATA_DIR)' \
                  'PG2ARROW_CMD=$(PG2ARROW_CMD)'               \
                  'ARROW2CSV_CMD=$(ARROW2CSV_CMD)'             \
                  'DBGEN_SSBM_CMD=$(DBGEN_SSBM_CMD)'           \
                  'REGRESS_DATASET_REVISION=$(REGRESS_REVISION)'
REGRESS_OPTS   := --inputdir=./$(PG_MAJORVERSION)              \
                  --outputdir=./$(PG_MAJORVERSION)             \
                  --encoding=UTF-8                             \
                  --load-extension=pg_strom                    \
                  --launcher="env $(REGRESS_SETENV)"

ifeq ($(REGRESS_DB_REVISION),$(REGRESS_REVISION))
REGRESS_OPTS   += --use-existing
endif

HAS_POSTGIS    := $(shell $(PSQL) template1 -At -c "SELECT count(*) FROM pg_catalog.pg_available_extensions WHERE name='postgis'")
ifeq ($(HAS_POSTGIS),1)
REGRESS_OPTS   += --load-extension=postgis
REGRESS        := --schedule=./parallel_schedule_postgis
endif

REGRESS_PREP   := pg2arrow arrow2csv dbgen-ssbm parallel_schedule_postgis
EXTRA_CLEAN    := parallel_schedule_postgis

all: installcheck

pg2arrow:
	make -C ../arrow-tools PG_CONFID=$(PG_CONFIG) pg2arrow

arrow2csv:
	make -C ../arrow-tools PG_CONFID=$(PG_CONFIG) arrow2csv

dbgen-ssbm:
	make -C ssbm

parallel_schedule_postgis: parallel_schedule
	cat $< | sed -e 's/#@@POSTGIS@@/test: postgis/g' > $@

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
