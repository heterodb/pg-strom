#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG := pg_config
PSQL := $(shell dirname $(shell which $(PG_CONFIG)))/psql
CREATEDB := $(shell dirname $(shell which $(PG_CONFIG)))/createdb
MKDOCS := mkdocs

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = .
endif

# Custom configurations if any
-include $(STROM_BUILD_ROOT)/Makefile.custom

#
# PostgreSQL versioning
#
PG_VERSION_NUM=$(shell $(PG_CONFIG) --version | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')

#
# Installation related
#
PGSTROM_SQL := $(STROM_BUILD_ROOT)/sql/pg_strom--2.0.sql

#
# Source file of CPU portion
#
__STROM_OBJS = main.o codegen.o datastore.o cuda_program.o \
		gpu_device.o gpu_context.o gpu_mmgr.o nvme_strom.o relscan.o \
		gpu_tasks.o gpuscan.o gpujoin.o gpupreagg.o pl_cuda.o \
		aggfuncs.o matrix.o float2.o ccache.o \
		largeobject.o gstore_fdw.o misc.o
__STROM_HEADERS = pg_strom.h nvme_strom.h device_attrs.h cuda_filelist
STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_OBJS))
__STROM_SOURCES = $(__STROM_OBJS:.o=.c)
STROM_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_SOURCES))

#
# Source file of GPU portion
#
__CUDA_SOURCES = $(shell cpp -D 'PGSTROM_CUDA(x)=cuda_\#\#x.h' \
                 $(STROM_BUILD_ROOT)/src/cuda_filelist | grep -v ^\#)
CUDA_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__CUDA_SOURCES))

#
# Source file of utilities
#
__STROM_UTILS = gpuinfo dbgen-ssbm dbgen-dbt3
STROM_UTILS = $(addprefix $(STROM_BUILD_ROOT)/utils/, $(__STROM_UTILS))

GPUINFO = $(STROM_BUILD_ROOT)/utils/gpuinfo
GPUINFO_SOURCE = $(STROM_BUILD_ROOT)/utils/gpuinfo.c
GPUINFO_CFLAGS = $(PGSTROM_FLAGS) -I $(IPATH) -L $(LPATH) -lcuda

SSBM_DBGEN = $(STROM_BUILD_ROOT)/utils/dbgen-ssbm
__SSBM_DBGEN_SOURCE = bcd2.c  build.c load_stub.c print.c text.c \
		bm_utils.c driver.c permute.c rnd.c speed_seed.c dists.dss.h
SSBM_DBGEN_SOURCE = $(addprefix $(STROM_BUILD_ROOT)/utils/ssbm/, \
                                $(__SSBM_DBGEN_SOURCE))
SSBM_DBGEN_DISTS_DSS = $(STROM_BUILD_ROOT)/utils/ssbm/dists.dss.h
SSBM_DBGEN_CFLAGS = -DDBNAME=\"dss\" -DLINUX -DDB2 -DSSBM -DTANDEM \
                    -DSTATIC_DISTS=1 \
                    -O2 -g -I. -I$(STROM_BUILD_ROOT)/utils/ssbm -lm
__SSBM_SQL_FILES = ssbm-11.sql ssbm-12.sql ssbm-13.sql \
                   ssbm-21.sql ssbm-22.sql ssbm-23.sql \
                   ssbm-31.sql ssbm-32.sql ssbm-33.sql ssbm-34.sql \
                   ssbm-41.sql ssbm-42.sql ssbm-43.sql

DBT3_DBGEN = $(STROM_BUILD_ROOT)/utils/dbgen-dbt3
__DBT3_DBGEN_SOURCE = bcd2.c build.c load_stub.c print.c rng64.c text.c \
                   bm_utils.c driver.c permute.c rnd.c speed_seed.c dists.dss.h
DBT3_DBGEN_SOURCE = $(addprefix $(STROM_BUILD_ROOT)/utils/dbt3/, \
                                $(__DBT3_DBGEN_SOURCE))
DBT3_DBGEN_DISTS_DSS = $(STROM_BUILD_ROOT)/utils/dbt3/dists.dss.h
DBT3_DBGEN_CFLAGS = -Wno-unused-variable -Wno-unused-but-set-variable \
                    -Wno-parentheses -Wno-unused-result -Wall \
                    -O2 -g -I. -I$(STROM_BUILD_ROOT)/utils/dbt3 \
                    -DLINUX=1 -DTPCH=1 -DEOL_HANDLING=1 -DSTATIC_DISTS=1
__DBT3_SQL_FILES = dbt3-01.sql dbt3-02.sql dbt3-03.sql dbt3-04.sql \
                   dbt3-05.sql dbt3-06.sql dbt3-07.sql dbt3-08.sql \
                   dbt3-09.sql dbt3-10.sql dbt3-11.sql dbt3-12.sql \
                   dbt3-13.sql dbt3-14.sql dbt3-15.sql dbt3-16.sql \
                   dbt3-17.sql dbt3-18.sql dbt3-19.sql dbt3-20.sql \
                   dbt3-21.sql dbt3-22.sql

TESTAPP_LARGEOBJECT = $(STROM_BUILD_ROOT)/test/testapp_largeobject
TESTAPP_LARGEOBJECT_SOURCE = $(TESTAPP_LARGEOBJECT).cu

#
# Header files
#
__STROM_HEADERS = pg_strom.h nvme_strom.h device_attrs.h
STROM_HEADERS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_HEADERS))

#
# Markdown (document) files
#
__DOC_FILES = index.md install.md partition.md \
              operations.md sys_admin.md brin.md partition.md troubles.md \
	      ssd2gpu.md ccache.md gstore_fdw.md plcuda.md \
	      ref_types.md ref_devfuncs.md ref_sqlfuncs.md ref_params.md \
	      release_note.md

#
# Files to be packaged
#
__PACKAGE_FILES = LICENSE README.md Makefile pg_strom.control	\
	          src sql utils test man
ifdef PGSTROM_VERSION
__STROM_TGZ = pg_strom-$(shell echo $(PGSTROM_VERSION))
else
__STROM_TGZ = pg_strom-master
endif
STROM_TGZ = $(addprefix $(STROM_BUILD_ROOT)/, $(__STROM_TGZ).tar.gz)
ifdef PGSTROM_GITHASH
__STROM_TGZ_GITHASH = $(PGSTROM_GITHASH)
else
__STROM_TGZ_GITHASH = HEAD
endif

#
# Header and Libraries of CUDA
#
CUDA_PATH_LIST := /usr/local/cuda /usr/local/cuda-*
CUDA_PATH := $(shell for x in $(CUDA_PATH_LIST);    \
           do test -e "$$x/include/cuda.h" && echo $$x; done | head -1)
IPATH := $(CUDA_PATH)/include
BPATH := $(CUDA_PATH)/bin
LPATH := $(CUDA_PATH)/lib64
NVCC  := $(CUDA_PATH)/bin/nvcc

#
# Flags to build
# --------------
# NOTE: we assume to put the following line in Makefile.custom according to
#       the purpose of this build
#
#       PGSTROM_FLAGS_CUSTOM := -g -O0 -Werror
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
ifdef PGSTROM_VERSION
PGSTROM_FLAGS += "-DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\""
endif
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"
PGSTROM_FLAGS += -DCUDA_INCLUDE_PATH=\"$(IPATH)\"
PGSTROM_FLAGS += -DCUDA_BINARY_PATH=\"$(BPATH)\"
PGSTROM_FLAGS += -DCUDA_LIBRARY_PATH=\"$(LPATH)\"
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(IPATH)
SHLIB_LINK := -L $(LPATH) -lnvrtc -lcuda

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
OBJS =  $(STROM_OBJS)
EXTENSION = pg_strom
DATA = $(shell cpp -D 'PGSTROM_CUDA(x)=$(STROM_BUILD_ROOT)/src/cuda_\#\#x.h' \
                      $(STROM_BUILD_ROOT)/src/cuda_filelist | grep -v ^\#) \
       $(PGSTROM_SQL) $(PGSTROM_TEST_SQL)

# Support utilities
SCRIPTS_built = $(STROM_UTILS) $(PGSTROM_TEST_UTILS)
# Extra files to be cleaned
EXTRA_CLEAN = $(STROM_UTILS) \
	$(shell ls $(STROM_BUILD_ROOT)/man/docs/*.md 2>/dev/null) \
	$(shell ls */Makefile 2>/dev/null | sed 's/Makefile/pg_strom.control/g') \
	$(shell ls pg-strom-*.tar.gz 2>/dev/null) \
	$(STROM_BUILD_ROOT)/man/markdown_i18n \
	$(SSBM_DBGEN_DISTS_DSS) \
	$(DBT3_DBGEN_DISTS_DSS) \
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

#
# Build chain of PostgreSQL
#
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifneq ($(STROM_BUILD_ROOT), .)
pg_strom.control: $(addprefix $(STROM_BUILD_ROOT)/, pg_strom.control)
	cp -f $< $@
endif

#
# Build documentation
#
$(STROM_BUILD_ROOT)/man/markdown_i18n: $(STROM_BUILD_ROOT)/man/markdown_i18n.c
	$(CC) $(CFLAGS) -o $@ $(addsuffix .c,$@)

docs:	$(STROM_BUILD_ROOT)/man/markdown_i18n
	for x in $(__DOC_FILES);			\
	do						\
	  $(STROM_BUILD_ROOT)/man/markdown_i18n		\
	    -f $(STROM_BUILD_ROOT)/man/$$x		\
	    -o $(STROM_BUILD_ROOT)/man/docs/$$x;	\
	done
	$(STROM_BUILD_ROOT)/man/markdown_i18n		\
	    -f $(STROM_BUILD_ROOT)/man/mkdocs.yml	\
	    -o $(STROM_BUILD_ROOT)/man/mkdocs.en.yml
	pushd $(STROM_BUILD_ROOT)/man;			\
	$(MKDOCS) build -c -f mkdocs.en.yml -d ../docs;	\
	popd
	for x in $(__DOC_FILES);			\
	do						\
	  $(STROM_BUILD_ROOT)/man/markdown_i18n -l ja	\
	    -f $(STROM_BUILD_ROOT)/man/$$x		\
	    -o $(STROM_BUILD_ROOT)/man/docs/$$x;	\
	done
	$(STROM_BUILD_ROOT)/man/markdown_i18n -l ja	\
	    -f $(STROM_BUILD_ROOT)/man/mkdocs.yml	\
	    -o $(STROM_BUILD_ROOT)/man/mkdocs.ja.yml
	pushd $(STROM_BUILD_ROOT)/man;			\
	$(MKDOCS) build -c -f mkdocs.ja.yml -d ../docs/ja; \
	popd

#
# Build utilities
#
$(GPUINFO): $(GPUINFO_SOURCE) $(STROM_HEADERS)
	$(CC) $(GPUINFO_CFLAGS) $(GPUINFO_SOURCE) -o $@

$(SSBM_DBGEN): $(SSBM_DBGEN_SOURCE) $(SSBM_DBGEN_DISTS_DSS)
	$(CC) $(SSBM_DBGEN_CFLAGS) $(SSBM_DBGEN_SOURCE) -o $@

$(SSBM_DBGEN_DISTS_DSS): $(basename $(SSBM_DBGEN_DISTS_DSS))
	@(echo "const char *static_dists_dss ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $^; \
	  echo ";") > $@

$(DBT3_DBGEN): $(DBT3_DBGEN_SOURCE) $(DBT3_DBGEN_DISTS_DSS)
	$(CC) $(DBT3_DBGEN_CFLAGS) $(DBT3_DBGEN_SOURCE) -o $@

$(DBT3_DBGEN_DISTS_DSS): $(basename $(DBT3_DBGEN_DISTS_DSS))
	@(echo "const char *static_dists_dss ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $^; \
	  echo ";") > $@

$(TESTAPP_LARGEOBJECT): $(TESTAPP_LARGEOBJECT_SOURCE)
	$(NVCC) -I $(shell $(PG_CONFIG) --pkgincludedir) \
	        -L $(shell $(PG_CONFIG) --pkglibdir) \
	        -Xcompiler \"-Wl,-rpath,$(shell $(PG_CONFIG) --pkglibdir)\" \
	        -lpq -o $@ $^

#
# Tarball
#
$(STROM_TGZ): $(shell cd $(STROM_BUILD_ROOT); git ls-files $(__PACKAGE_FILES))
	(cd $(STROM_BUILD_ROOT);                 \
	 git archive	--format=tar.gz          \
			--prefix=$(__STROM_TGZ)/ \
			-o $(__STROM_TGZ).tar.gz \
			$(__STROM_TGZ_GITHASH) $(__PACKAGE_FILES))

tarball: $(STROM_TGZ)

#
# init regression test database
#
init_regression_testdb: $(DBT3_DBGEN)
	cd $(STROM_BUILD_ROOT)/test;
	REV=`$(PSQL) $(REGRESS_DBNAME) -At -c $(REGRESS_REVISION_QUERY)`; \
	if [ "$$REV" != "$(REGRESS_REVISION)" ]; then \
	  $(CREATEDB) -l C $(REGRESS_DBNAME); \
	  cd $(STROM_BUILD_ROOT)/test && \
	  $(PSQL) $(REGRESS_DBNAME) -f testdb_init.sql; \
	fi

.PHONY: docs
