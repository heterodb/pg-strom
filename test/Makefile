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

SSBM_DBGEN = $(addprefix $(STROM_BUILD_ROOT)/test/ssbm/, dbgen)
SSBM_DBGEN_DISTS_DSS = $(addprefix $(STROM_BUILD_ROOT)/test/ssbm/, dists.dss)
__SSBM_DBGEN_SOURCE = bcd2.c  build.c load_stub.c print.c text.c \
                   bm_utils.c driver.c permute.c rnd.c speed_seed.c
SSBM_DBGEN_SOURCE = $(addprefix $(STROM_BUILD_ROOT)/test/ssbm/, \
                                $(__SSBM_DBGEN_SOURCE))
SSBM_DBGEN_FLAGS = -DDBNAME=\"dss\" -DLINUX -DDB2 -DSSBM -DTANDEM \
                   -O2 -g -I. -lm

TESTAPP_LARGEOBJECT = $(STROM_BUILD_ROOT)/test/testapp_largeobject
TESTAPP_LARGEOBJECT_SOURCE = $(addsuffix .cu,$(TESTAPP_LARGEOBJECT))

EXTRA_CLEAN_TEST = $(DBT3_DBGEN) $(DBT3_DBGEN_DISTS_DSS).h \
                   $(SSBM_DBGEN) $(SSBM_DBGEN_DISTS_DSS) \
                   $(TESTAPP_LARGEOBJECT)

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

$(notdir $(DBT3_DBGEN)): $(DBT3_DBGEN) $(SSBM_DBGEN)

$(DBT3_DBGEN): $(DBT3_DBGEN_SOURCE) $(addsuffix .h,$(DBT3_DBGEN_DISTS_DSS))
	cd $(STROM_BUILD_ROOT)/test/dbt3 \
        && $(CC) $(DBT3_DBGEN_FLAGS) $(notdir $(DBT3_DBGEN_SOURCE)) \
                                  -o $(notdir $(DBT3_DBGEN))

$(addsuffix .h,$(DBT3_DBGEN_DISTS_DSS)): $(DBT3_DBGEN_DISTS_DSS)
	@(echo "const char *static_dists_dss ="; \
          sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
              -e 's/^/  "/g' -e 's/$$/\\n"/g' < $^; \
          echo ";") > $@

$(SSBM_DBGEN): $(SSBM_DBGEN_SOURCE) $(addsuffix .h,$(SSBM_DBGEN_DISTS_DSS))
	cd $(STROM_BUILD_ROOT)/test/ssbm \
	&& $(CC) $(SSBM_DBGEN_FLAGS) $(notdir $(SSBM_DBGEN_SOURCE)) \
                                  -o $(notdir $(SSBM_DBGEN))

$(addsuffix .h,$(SSBM_DBGEN_DISTS_DSS)): $(SSBM_DBGEN_DISTS_DSS)
	@(echo "const char *static_dists_dss ="; \
          sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
              -e 's/^/  "/g' -e 's/$$/\\n"/g' < $^; \
          echo ";") > $@

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
