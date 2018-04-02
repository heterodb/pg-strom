#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG := pg_config
PSQL := $(shell dirname $(shell which $(PG_CONFIG)))/psql
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
# PG9.6 or later is required
PG_MIN_VERSION_NUM=90600
#PG_MAX_VERSION_NUM=

#
# Installation related
#
PGSTROM_SQL := $(STROM_BUILD_ROOT)/pg_strom--1.0.sql
PGSTROM_SQL_SRC = basis.sql aggfuncs.sql matrix.sql float2.sql test.sql

#
# Source file of CPU portion
#
__STROM_OBJS = main.o codegen.o datastore.o cuda_program.o \
		gpu_device.o gpu_context.o gpu_mmgr.o \
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
__STROM_UTILS = gpuinfo
STROM_UTILS = $(addprefix $(STROM_BUILD_ROOT)/utils/, $(__STROM_UTILS))

#
# Test support utilities
#
DBT3_DBGEN = $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, dbgen)
DBT3_DBGEN_DISTS_DSS = $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, dists.dss)
DBT3_DBGEN_SOURCE = bcd2.c build.c load_stub.c print.c rng64.c text.c \
                      bm_utils.c driver.c permute.c rnd.c speed_seed.c
DBT3_DBGEN_FLAGS = -Wno-unused-variable -Wno-unused-but-set-variable \
                   -Wno-parentheses -Wno-unused-result -Wall \
                   -I. -DLINUX=1 -DTPCH=1

#
# Header files
#
__STROM_HEADERS = pg_strom.h nvme_strom.h device_attrs.h
STROM_HEADERS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_HEADERS))

#
# Markdown (document) files
#
__DOC_FILES = index.md install.md tutorial.md features.md \
	      plcuda.md references.md release_note.md

#
# Files to be packaged
#
__PACKAGE_FILES = LICENSE README.md Makefile pg_strom.control	\
	          src sql utils test man
ifdef PGSTROM_VERSION
__STROM_TGZ = pg_strom-$(shell echo $(PGSTROM_VERSION) | sed -e 's/^v//g')
__STROM_TGZ_TAG = $(PGSTROM_VERSION)
else
__STROM_TGZ = pg_strom-master
__STROM_TGZ_TAG = HEAD
endif
STROM_TGZ = $(addprefix $(STROM_BUILD_ROOT)/, $(__STROM_TGZ).tar.gz)

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
ifdef PG_MIN_VERSION_NUM
PGSTROM_FLAGS += "-DPG_MIN_VERSION_NUM=$(PG_MIN_VERSION_NUM)"
endif
ifdef PG_MAX_VERSION_NUM
PGSTROM_FLAGS += "-DPG_MAX_VERSION_NUM=$(PG_MAX_VERSION_NUM)"
endif
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"
PGSTROM_FLAGS += -DCUDA_INCLUDE_PATH=\"$(IPATH)\"
PGSTROM_FLAGS += -DCUDA_BINARY_PATH=\"$(BPATH)\"
PGSTROM_FLAGS += -DCUDA_LIBRARY_PATH=\"$(LPATH)\"
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(IPATH)
SHLIB_LINK := -L $(LPATH) -lnvrtc -lcuda
#LDFLAGS_SL := -Wl,-rpath,'$(LPATH)'

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
OBJS =  $(STROM_OBJS)
EXTENSION = pg_strom
DATA_built = $(PGSTROM_SQL)
DATA = $(shell cpp -D 'PGSTROM_CUDA(x)=$(STROM_BUILD_ROOT)/src/cuda_\#\#x.h' \
                      $(STROM_BUILD_ROOT)/src/cuda_filelist | grep -v ^\#)

# Support utilities
SCRIPTS_built = $(STROM_UTILS) $(DBT3_DBGEN)
# Extra files to be cleaned
EXTRA_CLEAN = $(STROM_UTILS) \
	$(shell ls $(STROM_BUILD_ROOT)/man/docs/*.md) \
	$(shell ls */Makefile | sed 's/Makefile/pg_strom.control/g') \
	$(shell ls pg-strom-*.tar.gz) \
	$(STROM_BUILD_ROOT)/man/markdown_i18n \
	$(STROM_BUILD_ROOT)/test/testapp_largeobject \
	$(DBT3_DBGEN) $(DBT3_DBGEN_DISTS_DSS).h

#
# Regression Test
#
USE_MODULE_DB = 1
REGRESS = --schedule=$(STROM_BUILD_ROOT)/test/parallel_schedule
REGRESS_DBNAME = contrib_regression_$(MODULE_big)
REGRESS_REVISION = SELECT public.pgstrom_regression_test_revision()
REGRESS_OPTS = --inputdir=$(STROM_BUILD_ROOT)/test --use-existing \
               --launcher="env PGDATABASE=$(REGRESS_DBNAME)"
REGRESS_PREP = init_regression_testdb

#
# Build chain of PostgreSQL
#
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifneq ($(STROM_BUILD_ROOT), .)
pg_strom.control: $(addprefix $(STROM_BUILD_ROOT)/, pg_strom.control)
	cp -f $< $@
endif

$(PGSTROM_SQL): $(addprefix $(STROM_BUILD_ROOT)/sql/, $(PGSTROM_SQL_SRC))
	cat $^ > $@

$(STROM_UTILS): $(addsuffix .c,$(STROM_UTILS)) $(STROM_HEADERS)
	$(CC) $(CFLAGS) $(addsuffix .c,$@) $(PGSTROM_FLAGS) -I $(IPATH) -L $(LPATH) -lcuda -lnvrtc -o $@$(X)

$(addsuffix .h,$(DBT3_DBGEN_DISTS_DSS)): $(DBT3_DBGEN_DISTS_DSS)
	cat $^ > $@

$(DBT3_DBGEN): $(addprefix $(STROM_BUILD_ROOT)/test/dbt3/, $(DBT3_DBGEN_SOURCE)) \
               $(addsuffix .h,$(DBT3_DBGEN_DISTS_DSS))
	cd $(STROM_BUILD_ROOT)/test/dbt3 && $(CC) $(DBT3_DBGEN_FLAGS) $(DBT3_DBGEN_SOURCE) \
                                               -o $(notdir $(DBT3_DBGEN))

$(HTML_FILES): $(HTML_SOURCES) $(HTML_TEMPLATE)
	@$(MKDIR_P) $(STROM_BUILD_ROOT)/doc/html
	$(PYTHON_CMD) $(MENUGEN_PY) \
		-t $(HTML_TEMPLATE) \
		-v '$(HTML_VERSION)' \
		-m $(addprefix $(STROM_BUILD_ROOT)/doc/, $(notdir $(basename $@)).src.html) \
		$(HTML_SOURCES) > $@

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

$(STROM_TGZ): $(shell cd $(STROM_BUILD_ROOT); git ls-files $(__PACKAGE_FILES))
	(cd $(STROM_BUILD_ROOT);                 \
	 git archive	--format=tar.gz          \
			--prefix=$(__STROM_TGZ)/ \
			-o $(__STROM_TGZ).tar.gz \
			$(__STROM_TGZ_TAG) $(__PACKAGE_FILES))

tarball: $(STROM_TGZ)

$(STROM_BUILD_ROOT)/test/testapp_largeobject: $(STROM_BUILD_ROOT)/test/testapp_largeobject.cu
	$(NVCC) -I $(shell $(PG_CONFIG) --pkgincludedir) 					\
	        -L $(shell $(PG_CONFIG) --pkglibdir)						\
	        -Xcompiler \"-Wl,-rpath,$(shell $(PG_CONFIG) --pkglibdir)\"	\
	        -lpq -o $(STROM_BUILD_ROOT)/test/testapp_largeobject $^

init_regression_testdb: $(STROM_BUILD_ROOT)/test/testapp_largeobject
	$(STROM_BUILD_ROOT)/test/testdb_init.sh $(REGRESS_DBNAME) $(PSQL)
.PHONY: docs
