#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG ?= pg_config
PSQL   = $(shell dirname $(shell which $(PG_CONFIG)))/psql
MKDOCS = mkdocs

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = .
endif

# Custom configurations if any
-include $(STROM_BUILD_ROOT)/Makefile.custom

# GPU code build configurations
include $(STROM_BUILD_ROOT)/Makefile.cuda

#
# PG-Strom version
#
PGSTROM_VERSION := 3.3
PGSTROM_RELEASE := devel

#
# Installation related
#
__PGSTROM_SQL = pg_strom--2.2.sql pg_strom--3.0.sql \
                pg_strom--2.2--2.3.sql  pg_strom--2.3--3.0.sql \
                pg_strom--3.0--4.0.sql
PGSTROM_SQL := $(addprefix $(STROM_BUILD_ROOT)/sql/, $(__PGSTROM_SQL))

#
# Source file of CPU portion
#
__STROM_OBJS = main.o nvrtc.o extra.o \
        shmbuf.o codegen.o datastore.o cuda_program.o \
        gpu_device.o gpu_context.o gpu_mmgr.o \
        relscan.o gpu_tasks.o gpu_cache.o \
        gpuscan.o gpujoin.o gpupreagg.o \
        arrow_fdw.o arrow_nodes.o arrow_write.o arrow_pgsql.o \
        aggfuncs.o float2.o tinyint.o misc.o
STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_OBJS))

#
# Source file of GPU portion
#
__GPU_FATBIN := cuda_common cuda_numeric cuda_primitive \
                cuda_timelib cuda_textlib cuda_misclib \
                cuda_jsonlib cuda_rangetype cuda_postgis \
                cuda_gpuscan cuda_gpujoin cuda_gpupreagg cuda_gpusort
__GPU_HEADERS := $(__GPU_FATBIN) cuda_utils cuda_basetype cuda_gcache arrow_defs
GPU_HEADERS := $(addprefix $(STROM_BUILD_ROOT)/src/, \
               $(addsuffix .h, $(__GPU_HEADERS)))
GPU_FATBIN := $(addprefix $(STROM_BUILD_ROOT)/src/, \
              $(addsuffix .fatbin, $(__GPU_FATBIN)))
GPU_DEBUG_FATBIN := $(GPU_FATBIN:.fatbin=.gfatbin)
GPU_CACHE_FATBIN := $(STROM_BUILD_ROOT)/src/cuda_gcache.fatbin
GPU_CACHE_DEBUG_FATBIN := $(STROM_BUILD_ROOT)/src/cuda_gcache.gfatbin

#
# Source file of utilities
#
__STROM_UTILS = gpuinfo dbgen-ssbm
STROM_UTILS = $(addprefix $(STROM_BUILD_ROOT)/utils/, $(__STROM_UTILS))

GPUINFO := $(STROM_BUILD_ROOT)/utils/gpuinfo
GPUINFO_SOURCE := $(STROM_BUILD_ROOT)/utils/gpuinfo.c
GPUINFO_DEPEND := $(GPUINFO_SOURCE)
GPUINFO_CFLAGS = $(PGSTROM_FLAGS) -I $(CUDA_IPATH) -L $(CUDA_LPATH) \
                 -I $(STROM_BUILD_ROOT)/src \
                 -I $(STROM_BUILD_ROOT)/utils \
                 $(shell $(PG_CONFIG) --ldflags)

SSBM_DBGEN = $(STROM_BUILD_ROOT)/utils/dbgen-ssbm
__SSBM_DBGEN_SOURCE = bcd2.c  build.c load_stub.c print.c text.c \
		bm_utils.c driver.c permute.c rnd.c speed_seed.c dists.dss.h
SSBM_DBGEN_SOURCE = $(addprefix $(STROM_BUILD_ROOT)/utils/ssbm/, \
                                $(__SSBM_DBGEN_SOURCE))
SSBM_DBGEN_DISTS_DSS = $(STROM_BUILD_ROOT)/utils/ssbm/dists.dss.h
SSBM_DBGEN_CFLAGS = -DDBNAME=\"dss\" -DLINUX -DDB2 -DSSBM -DTANDEM \
                    -DSTATIC_DISTS=1 \
                    -O2 -g -I. -I$(STROM_BUILD_ROOT)/utils/ssbm \
                    $(shell $(PG_CONFIG) --ldflags)
__SSBM_SQL_FILES = ssbm-11.sql ssbm-12.sql ssbm-13.sql \
                   ssbm-21.sql ssbm-22.sql ssbm-23.sql \
                   ssbm-31.sql ssbm-32.sql ssbm-33.sql ssbm-34.sql \
                   ssbm-41.sql ssbm-42.sql ssbm-43.sql

#
# Apache Arrow utilities
#
ARROW_BUILD_ROOT = $(STROM_BUILD_ROOT)/arrow-tools
PG2ARROW     = $(ARROW_BUILD_ROOT)/pg2arrow
MYSQL2ARROW  = $(ARROW_BUILD_ROOT)/mysql2arrow
PCAP2ARROW   = $(ARROW_BUILD_ROOT)/pcap2arrow

#
# Markdown (document) files
#
__DOC_FILES = index.md install.md operations.md \
              partition.md brin.md postgis.md hll_count.md troubles.md \
	      ssd2gpu.md arrow_fdw.md gpucache.md fluentd.md \
	      ref_types.md ref_devfuncs.md ref_sqlfuncs.md ref_params.md \
	      release_v2.0.md release_v2.2.md release_v2.3.md release_v3.0.md

#
# Files to be packaged
#
__PACKAGE_FILES = LICENSE README.md Makefile Makefile.cuda pg_strom.control \
                  src sql utils arrow-tools fluentd test man
ifeq ($(PGSTROM_RELEASE),1)
__STROM_TGZ = pg_strom-$(PGSTROM_VERSION)
else
__STROM_TGZ = pg_strom-$(PGSTROM_VERSION)-$(PGSTROM_RELEASE)
endif
__STROM_TAR = $(__STROM_TGZ).tar
STROM_TAR = $(addprefix $(STROM_BUILD_ROOT)/, $(__STROM_TAR))
STROM_TGZ = $(STROM_TAR).gz

ifdef PGSTROM_GITHASH
# if 'HEAD' is given, replace it by the current githash
ifeq ($(PGSTROM_GITHASH),HEAD)
PGSTROM_GITHASH = $(shell git rev-parse HEAD)
endif
else
ifeq ($(shell test -e $(STROM_BUILD_ROOT)/.git/config && echo -n 1),1)
PGSTROM_GITHASH = $(shell git rev-parse HEAD)
ifneq ($(shell git diff | wc -l),0)
PGSTROM_GITHASH_SUFFIX=::local_changes
endif
else
ifeq ($(shell test -e $(STROM_BUILD_ROOT)/GITHASH && echo -n 1),1)
PGSTROM_GITHASH = $(shell cat $(STROM_BUILD_ROOT)/GITHASH)
else
PGSTROM_GITHASH = HEAD
endif
endif
endif

__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})
__SPECFILE = pg_strom-PG$(MAJORVERSION)

#
# Flags to build
# --------------
# NOTE: we assume to put the following line in Makefile.custom according to
#       the purpose of this build
#
#       PGSTROM_FLAGS_CUSTOM := -g -O0 -Werror
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -D__PGSTROM_MODULE__=1
PGSTROM_FLAGS += "-DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\""
# build with debug options
ifeq ($(PGSTROM_DEBUG),1)
PGSTROM_FLAGS += -g -O0 -DPGSTROM_DEBUG_BUILD=1
endif
PGSTROM_FLAGS += -DCPU_ARCH=\"$(shell uname -m)\"
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"
PGSTROM_FLAGS += -DPGSERV_INCLUDEDIR=\"$(shell $(PG_CONFIG) --includedir-server)\"
PGSTROM_FLAGS += -DCUDA_INCLUDE_PATH=\"$(CUDA_IPATH)\"
PGSTROM_FLAGS += -DCUDA_BINARY_PATH=\"$(CUDA_BPATH)\"
PGSTROM_FLAGS += -DCUDA_LIBRARY_PATH=\"$(CUDA_LPATH)\"
PGSTROM_FLAGS += -DCUDA_MAXREGCOUNT=$(MAXREGCOUNT)
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
PGSTROM_FLAGS += -DPGSTROM_GITHASH=\"$(PGSTROM_GITHASH)$(PGSTROM_GITHASH_SUFFIX)\"
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(CUDA_IPATH)
SHLIB_LINK := -L $(CUDA_LPATH) -lcuda

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
MODULEDIR = pg_strom
OBJS =  $(STROM_OBJS)
EXTENSION = pg_strom
DATA = $(GPU_HEADERS) $(PGSTROM_SQL) \
       $(STROM_BUILD_ROOT)/src/cuda_codegen.h \
       $(STROM_BUILD_ROOT)/Makefile.cuda
DATA_built = $(GPU_FATBIN) $(GPU_DEBUG_FATBIN) \
             $(GPU_CACHE_FATBIN) $(GPU_CACHE_DEBUG_FATBIN)

# Support utilities
SCRIPTS_built = $(STROM_UTILS)
# Extra files to be cleaned
EXTRA_CLEAN = $(STROM_UTILS) \
	$(shell ls $(STROM_BUILD_ROOT)/man/docs/*.md 2>/dev/null) \
	$(shell ls */Makefile 2>/dev/null | sed 's/Makefile/pg_strom.control/g') \
	$(shell ls $(STROM_BUILD_ROOT)/pg_strom-*.tar.gz 2>/dev/null) \
	$(shell dirname $(STROM_BUILD_ROOT)/pg_strom-*/GITHASH) \
	$(STROM_BUILD_ROOT)/man/markdown_i18n \
	$(SSBM_DBGEN_DISTS_DSS)

#
# Regression Test
#
USE_MODULE_DB := 1
REGRESS := --schedule=$(STROM_BUILD_ROOT)/test/parallel_schedule
REGRESS_INIT_SQL := $(STROM_BUILD_ROOT)/test/sql/init_regress.sql
REGRESS_DBNAME := contrib_regression_$(MODULE_big)
REGRESS_REVISION := 20200306
REGRESS_REVISION_QUERY := 'SELECT pgstrom.regression_testdb_revision() = $(REGRESS_REVISION)'
REGRESS_OPTS = --inputdir=$(STROM_BUILD_ROOT)/test/$(MAJORVERSION) \
               --outputdir=$(STROM_BUILD_ROOT)/test/$(MAJORVERSION) \
               --encoding=UTF-8 \
               --load-extension=pg_strom \
               --load-extension=plpython3u \
               --launcher="env PGDATABASE=$(REGRESS_DBNAME) PATH=$(shell dirname $(SSBM_DBGEN)):$$PATH PGAPPNAME=$(REGRESS_REVISION)" \
               $(shell test "`$(PSQL) -At -c $(REGRESS_REVISION_QUERY) $(REGRESS_DBNAME)`" = "t" && echo "--use-existing")
REGRESS_PREP = $(SSBM_DBGEN) $(PG2ARROW) $(REGRESS_INIT_SQL)

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
# GPU Libraries
#
$(GPU_CACHE_FATBIN): $(GPU_CACHE_FATBIN:.fatbin=.cu) $(GPU_HEADERS)
	$(NVCC) $(__NVCC_FLAGS) --relocatable-device-code=false -o $@ $<
$(GPU_CACHE_DEBUG_FATBIN): $(GPU_CACHE_DEBUG_FATBIN:.gfatbin=.cu) $(GPU_HEADERS)
	$(NVCC) $(__NVCC_DEBUG_FLAGS) --relocatable-device-code=false -o $@ $<
%.fatbin:  %.cu $(GPU_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
%.gfatbin: %.cu $(GPU_HEADERS)
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o $@ $<

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
	env LANG=en_US.utf8				\
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
	env LANG=ja_JP.utf8			 	\
		$(MKDOCS) build -c -f mkdocs.ja.yml -d ../docs/ja; \
	popd

#
# Build utilities
#
$(GPUINFO): $(GPUINFO_DEPEND)
	$(CC) $(GPUINFO_CFLAGS) \
              $(GPUINFO_SOURCE)  -o $@ -lcuda -lnvidia-ml -ldl

$(SSBM_DBGEN): $(SSBM_DBGEN_SOURCE) $(SSBM_DBGEN_DISTS_DSS)
	$(CC) $(SSBM_DBGEN_CFLAGS) $(SSBM_DBGEN_SOURCE) -o $@ -lm

$(SSBM_DBGEN_DISTS_DSS): $(basename $(SSBM_DBGEN_DISTS_DSS))
	@(echo "const char *static_dists_dss ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $^; \
	  echo ";") > $@

#
# Arrow utilities
#
$(PG2ARROW):
	make -C $(ARROW_BUILD_ROOT) pg2arrow

pg2arrow: $(PG2ARROW)

$(MYSQL2ARROW):
	make -C $(ARROW_BUILD_ROOT) mysql2arrow

mysql2arrow: $(MYSQL2ARROW)

$(PCAP2ARROW):
	make -C $(ARROW_BUILD_ROOT) pcap2arrow

pcap2arrow: $(PCAP2ARROW)

#
# Tarball
#
tarball:
	(cd $(STROM_BUILD_ROOT)                               && \
	 git archive --format=tar                                \
	             --prefix=$(__STROM_TGZ)/                    \
	             -o $(__STROM_TAR)                           \
                 $(PGSTROM_GITHASH) $(__PACKAGE_FILES)    && \
	 mkdir -p $(__STROM_TGZ)                              && \
	 echo $(PGSTROM_GITHASH) > $(__STROM_TGZ)/GITHASH     && \
	 tar -r -f $(__STROM_TAR) $(__STROM_TGZ)/GITHASH      && \
	 gzip -f $(__STROM_TAR) && test -e $(__STROM_TGZ)) || exit 1

#
# RPM Package
#
rpm: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/systemd-pg_strom.conf \
	    > $(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/pg_strom.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
        -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    -e "s/@@PGSQL_VERSION@@/$(MAJORVERSION)/g"      \
	    > $(__SPECDIR)/pg_strom-PG$(MAJORVERSION).spec
	rpmbuild -ba $(__SPECDIR)/pg_strom-PG$(MAJORVERSION).spec \
                 --undefine=_debugsource_packages

rpm-pg2arrow: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/pg2arrow.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
	    -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    > $(__SPECDIR)/pg2arrow.spec
	rpmbuild -ba $(__SPECDIR)/pg2arrow.spec --undefine=_debugsource_packages

rpm-mysql2arrow: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/mysql2arrow.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
	    -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    > $(__SPECDIR)/mysql2arrow.spec
	rpmbuild -ba $(__SPECDIR)/mysql2arrow.spec --undefine=_debugsource_packages

rpm-pcap2arrow: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/pcap2arrow.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
	    -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    > $(__SPECDIR)/pcap2arrow.spec
	rpmbuild -ba $(__SPECDIR)/pcap2arrow.spec --undefine=_debugsource_packages

rpm-arrow2csv: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/pcap2arrow.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
	    -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    > $(__SPECDIR)/arrow2csv.spec
	rpmbuild -ba $(__SPECDIR)/arrow2csv.spec --undefine=_debugsource_packages

rpm-fluentd-arrow: tarball
	cp -f $(STROM_TGZ) $(__SOURCEDIR) || exit 1
	(git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/files/fluentd-arrow.spec.in; \
	 git show --format=raw $(PGSTROM_GITHASH):$(STROM_BUILD_ROOT)/CHANGELOG) | \
	sed -e "s/@@STROM_VERSION@@/$(PGSTROM_VERSION)/g"   \
	    -e "s/@@STROM_RELEASE@@/$(PGSTROM_RELEASE)/g"   \
	    -e "s/@@STROM_TARBALL@@/$(__STROM_TGZ)/g"       \
	    -e "s/@@PGSTROM_GITHASH@@/$(PGSTROM_GITHASH)/g" \
	    > $(__SPECDIR)/fluentd-arrow.spec
	rpmbuild -ba $(__SPECDIR)/fluentd-arrow.spec --undefine=_debugsource_packages

rpm-arrow: rpm-pg2arrow rpm-mysql2arrow rpm-pcap2arrow rpm-arrow2csv rpm-fluentd-arrow

.PHONY: docs
