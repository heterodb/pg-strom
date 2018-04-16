ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = ..
endif

#
# Test support utilities
#
DBT3_DBGEN = $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, dbgen)
DBT3_DBGEN_DISTS_DSS = $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, dists.dss)
__DBT3_DBGEN_SOURCE = bcd2.c build.c load_stub.c print.c rng64.c text.c \
                   bm_utils.c driver.c permute.c rnd.c speed_seed.c
DBT3_DBGEN_SOURCE = $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, \
                                $(__DBT3_DBGEN_SOURCE))
DBT3_DBGEN_FLAGS = -Wno-unused-variable -Wno-unused-but-set-variable \
                   -Wno-parentheses -Wno-unused-result -Wall \
                   -g -I. -DLINUX=1 -DTPCH=1 -DEOL_HANDLING=1

TESTAPP_LARGEOBJECT = $(STROM_BUILD_ROOT)/test/testapp_largeobject
TESTAPP_LARGEOBJECT_SOURCE = $(addsuffix .cu,$(TESTAPP_LARGEOBJECT))

#
# Regression Test
#
USE_MODULE_DB = 1
REGRESS = --schedule=$(STROM_BUILD_ROOT)/test/parallel_schedule
REGRESS_DBNAME = contrib_regression_$(MODULE_big)
REGRESS_REVISION = 20180124
REGRESS_REVISION_QUERY = 'SELECT public.pgstrom_regression_test_revision()'
REGRESS_OPTS = --inputdir=$(STROM_BUILD_ROOT)/test --use-existing \
               --launcher="env PGDATABASE=$(REGRESS_DBNAME)"
REGRESS_PREP = init_regression_testdb $(TESTAPP_LARGEOBJECT)

include $(STROM_BUILD_ROOT)/Makefile

$(notdir $(DBT3_DBGEN)): $(DBT3_DBGEN)

$(DBT3_DBGEN): $(DBT3_DBGEN_SOURCE) $(addsuffix .h,$(DBT3_DBGEN_DISTS_DSS))
	cd $(STROM_BUILD_ROOT)/test/dbt3 \
        && $(CC) $(DBT3_DBGEN_FLAGS) $(notdir $(DBT3_DBGEN_SOURCE)) \
                                  -o $(notdir $(DBT3_DBGEN))

$(notdir $(TESTAPP_LARGEOBJECT)): $(TESTAPP_LARGEOBJECT)

$(TESTAPP_LARGEOBJECT): $(TESTAPP_LARGEOBJECT_SOURCE)
	$(NVCC) -I $(shell $(PG_CONFIG) --pkgincludedir) \
                -L $(shell $(PG_CONFIG) --pkglibdir)     \
                -Xcompiler \"-Wl,-rpath,$(shell $(PG_CONFIG) --pkglibdir)\"  \
                -lpq -o $@ $^

init_regression_testdb: $(DBT3_DBGEN)
	cd $(STROM_BUILD_ROOT)/test;
	REV=`$(PSQL) $(REGRESS_DBNAME) -At -c $(REGRESS_REVISION_QUERY)`; \
	if [ "$$REV" != "$(REGRESS_REVISION)" ]; then \
	  $(CREATEDB) -l C $(REGRESS_DBNAME); \
	  cd $(STROM_BUILD_ROOT)/test && \
	  $(PSQL) $(REGRESS_DBNAME) -f testdb_init.sql; \
	fi
