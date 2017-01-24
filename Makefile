#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG := pg_config
PYTHON_CMD := python
HAS_RPM_CMD := $(shell if which rpm > /dev/null; then echo 1; else echo 0; fi)

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = .
endif

# Custom configurations if any
-include $(STROM_BUILD_ROOT)/Makefile.custom

#
# PG-Strom versioning
#
PGSTROM_VERSION=1.0
PGSTROM_VERSION_NUM=$(shell echo $(PGSTROM_VERSION)			\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')
PG_VERSION_NUM=$(shell $(PG_CONFIG) --version | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')

# available platform versions
PG_MIN_VERSION=9.5.0
PG_MAX_VERSION=9.6.0
CUDA_MIN_VERSION=7.0
CUDA_MAX_VERSION=

PG_MIN_VERSION_NUM=$(shell echo $(PG_MIN_VERSION) | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'	\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')
PG_MAX_VERSION_NUM=$(shell echo $(PG_MAX_VERSION) | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'	\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')

#
# Source file of CPU portion
#
__STROM_OBJS = main.o codegen.o datastore.o aggfuncs.o \
		cuda_control.o cuda_program.o cuda_mmgr.o \
		gpuscan.o gpujoin.o gpupreagg.o gpusort.o \
		pl_cuda.o matrix.o

STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_OBJS))
__STROM_SOURCES = $(__STROM_OBJS:.o=.c)
STROM_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_SOURCES))

#
# Source file of GPU portion
#
__CUDA_OBJS = cuda_common.o \
	cuda_dynpara.o \
	cuda_matrix.o  \
	cuda_gpuscan.o \
	cuda_gpujoin.o \
	cuda_gpupreagg.o \
	cuda_gpusort.o \
	cuda_mathlib.o \
	cuda_textlib.o \
	cuda_timelib.o \
	cuda_numeric.o \
	cuda_money.o   \
	cuda_plcuda.o  \
	cuda_terminal.o
CUDA_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__CUDA_OBJS))
__CUDA_SOURCES = $(__CUDA_OBJS:.o=.c)
CUDA_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__CUDA_SOURCES))

__STROM_UTILS = gpuinfo kfunc_info
STROM_UTILS = $(addprefix $(STROM_BUILD_ROOT)/utils/, $(__STROM_UTILS))

#
# Files to be packaged
#
__RPM_SPECFILE = pg_strom.spec
RPM_SPECFILE = $(addprefix $(STROM_BUILD_ROOT)/, $(__RPM_SPECFILE))
__MISC_FILES = LICENSE README.md pg_strom.control Makefile \
	src/Makefile src/pg_strom.h src/pg_strom--1.0.sql

PACKAGE_FILES = $(__MISC_FILES)					\
	$(addprefix src/,$(__STROM_SOURCES))		\
	$(addprefix src/,$(__CUDA_SOURCES:.c=.h))	\
	$(addprefix utils/,$(addsuffix .c,$(__STROM_UTILS)))
__STROM_TGZ = pg_strom-$(PGSTROM_VERSION).tar.gz
STROM_TGZ = $(addprefix $(STROM_BUILD_ROOT)/, $(__STROM_TGZ))

#
# Source file of HTML document
#
__HTML_TEMPLATE = template.src.html
__HTML_SOURCES = manual.src.html \
	install.src.html \
	tutrial.src.html \
	pl_cuda.src.html \
	release.src.html
__IMAGE_SOURCES = lang_en.png \
	lang_ja.png \
	icon-warning.png \
	icon-caution.png \
	icon-hint.png \
	pgstrom-install-download-zip.png \
	cuda-install-target.png \
	plcuda-callflow.png \
	plcuda-overview.png \
	release-policy.png
__MANUAL_CSS = manual.css
__MENUGEN_PY = menugen.py

HTML_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__HTML_SOURCES))
HTML_FILES = $(addprefix $(STROM_BUILD_ROOT)/doc/html/, $(__HTML_SOURCES:.src.html=.html))
HTML_TEMPLATE = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__HTML_TEMPLATE))
IMAGE_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/doc/html/figs/, $(__IMAGE_SOURCES))
MANUAL_CSS = $(addprefix $(STROM_BUILD_ROOT)/doc/html/css/, $(__MANUAL_CSS))
MENUGEN_PY = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__MENUGEN_PY))

#
# Parameters for RPM package build
#
ifeq ($(HAS_RPM_CMD), 1)
__PGSQL_PKGS = $(shell rpm -q -g 'Applications/Databases' | grep -E '^postgresql[0-9]+-')
PGSQL_PKG_VERSION := $(shell if [ -n "$(__PGSQL_PKGS)" ];				\
                             then							\
                                 rpm -q $(__PGSQL_PKGS) --queryformat '%{version}\n';	\
                             else							\
                                 $(PG_CONFIG) --version | awk '{print $$NF}';		\
                             fi | uniq | sort -V | tail -1 |				\
                             sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g' |			\
                             awk '{printf "%d%d", $$1, $$2}')
CUDA_PKG_VERSION := $(shell rpm -q cuda --queryformat '%{version}\n' | 	\
                      sort -V | tail -1 |				\
                      sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g' |	\
                      awk '{printf "%d-%d", $$1, $$2}')

RPMBUILD_PARAMS := $(shell				\
    test -n "$(PGSTROM_VERSION)" &&			\
        echo " -D 'strom_version $(PGSTROM_VERSION)'";	\
    test -n "$(PGSQL_PKG_VERSION)" &&			\
        echo " -D 'pgsql_pkgver $(PGSQL_PKG_VERSION)'";	\
    test -n "$(CUDA_PKG_VERSION)" &&			\
        echo " -D 'cuda_pkgver $(CUDA_PKG_VERSION)'";	\
    test -n "$(PG_MIN_VERSION)" &&			\
        echo " -D 'pgsql_minver $(PG_MIN_VERSION)'";	\
    test -n "$(PG_MAX_VERSION)" &&			\
        echo " -D 'pgsql_maxver $(PG_MAX_VERSION)'";	\
    test -n "$(CUDA_MIN_VERSION)" &&			\
        echo " -D 'cuda_minver $(CUDA_MIN_VERSION)'";	\
    test -n "$(CUDA_MAX_VERSION)" &&			\
        echo " -D 'cuda_maxver $(CUDA_MAX_VERSION)'";	\
)
endif

#
# Header and Libraries of CUDA
#
CUDA_PATH_LIST := /usr/local/cuda /usr/local/cuda-*
CUDA_PATH := $(shell for x in $(CUDA_PATH_LIST);    \
           do test -e "$$x/include/cuda.h" && echo $$x; done | head -1)
IPATH := $(CUDA_PATH)/include
LPATH := $(CUDA_PATH)/lib64

#
# Flags to build
# --------------
# NOTE: we assume to put the following line in Makefile.custom according to
#       the purpose of this build
#
#       PGSTROM_FLAGS_CUSTOM := -DPGSTROM_DEBUG=1 -g -O0 -Werror
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"
PGSTROM_FLAGS += -DPGSTROM_VERSION_NUM=$(PGSTROM_VERSION_NUM)
PGSTROM_FLAGS += -DPG_MIN_VERSION_NUM=$(PG_MIN_VERSION_NUM)
PGSTROM_FLAGS += -DPG_MAX_VERSION_NUM=$(PG_MAX_VERSION_NUM)
PGSTROM_FLAGS += -DCUDA_INCLUDE_PATH=\"$(IPATH)\"
PGSTROM_FLAGS += -DCUDA_LIBRARY_PATH=\"$(LPATH)\"
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(IPATH)
SHLIB_LINK := -L $(LPATH) -lnvrtc -lcuda
#LDFLAGS_SL := -Wl,-rpath,'$(LPATH)'

#
# Options for regression test
#
# Regression test options
REGRESS = --schedule=$(STROM_BUILD_ROOT)/test/parallel_schedule
REGRESS_OPTS = --inputdir=$(STROM_BUILD_ROOT)/test
ifdef TEMP_INSTANCE
    REGRESS_OPTS += --temp-instance=$(STROM_BUILD_ROOT)/tmp_check
    ifndef CPUTEST
        REGRESS_OPTS += --temp-config=$(STROM_BUILD_ROOT)/test/enable.conf
    else
        REGRESS_OPTS += --temp-config=$(STROM_BUILD_ROOT)/test/disable.conf
    endif
endif

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
OBJS =  $(STROM_OBJS) $(CUDA_OBJS)
EXTENSION = pg_strom
ifeq ($(shell test $(PG_VERSION_NUM) -ge 90600; echo $??),0)
DATA = $(addprefix $(STROM_BUILD_ROOT)/src/, pg_strom--1.0.sql)
else
DATA = $(addprefix $(STROM_BUILD_ROOT)/src/, pg_strom--1.0.sql)
endif

# Support utilities
SCRIPTS_built = $(STROM_UTILS)
# Extra files to be cleaned
EXTRA_CLEAN = $(CUDA_SOURCES) $(HTML_FILES) $(STROM_UTILS) \
	$(shell test pg_strom.control -ef $(addprefix $(STROM_BUILD_ROOT)/src/, pg_strom.control) || echo pg_strom.control) \
	$(STROM_BUILD_ROOT)/__tarball $(STROM_TGZ)

#
# Build chain of PostgreSQL
#
ifndef PGSTROM_MAKEFILE_ONLY_PARAMDEF

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

pg_strom.control: $(addprefix $(STROM_BUILD_ROOT)/src/, pg_strom.control)
	test $< -ef $@ || cp -f $< $@

$(CUDA_SOURCES): $(CUDA_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(shell basename $(@:%.c=%))_code =";		\
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g'	\
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h;		\
	  echo ";") > $@

$(STROM_UTILS): $(addsuffix .c,$(STROM_UTILS))
	$(CC) $(CFLAGS) $(addsuffix .c,$@) $(PGSTROM_FLAGS) -I $(IPATH) -L $(LPATH) -lcuda -lnvrtc -o $@$(X)

$(HTML_FILES): $(HTML_SOURCES) $(HTML_TEMPLATE)
	@$(MKDIR_P) $(STROM_BUILD_ROOT)/doc/html
	$(PYTHON_CMD) $(MENUGEN_PY) \
		-t $(HTML_TEMPLATE) \
		-v '$(PGSTROM_VERSION)' \
		-m $(addprefix $(STROM_BUILD_ROOT)/doc/, $(notdir $(basename $@)).src.html) \
		$(HTML_SOURCES) > $@

html: $(HTML_FILES)

$(STROM_TGZ): $(addprefix $(STROM_BUILD_ROOT)/, $(PACKAGE_FILES))
	$(MKDIR_P) $(STROM_BUILD_ROOT)/__tarball/$(@:.tar.gz=)/src
	$(MKDIR_P) $(STROM_BUILD_ROOT)/__tarball/$(@:.tar.gz=)/utils
	$(foreach x,$(PACKAGE_FILES),cp -f $(STROM_BUILD_ROOT)/$x $(STROM_BUILD_ROOT)/__tarball/$(@:.tar.gz=)/$(x);)
	tar zc -C $(STROM_BUILD_ROOT)/__tarball $(@:.tar.gz=) > $@

tarball: $(STROM_TGZ)

ifeq ($(HAS_RPM_CMD), 1)
rpm: tarball $(RPM_SPECFILE)
	$(MKDIR_P) $(shell rpmbuild -E %{_sourcedir})
	cp -f $(STROM_TGZ) $(shell rpmbuild -E %{_sourcedir})
	rpmbuild $(RPMBUILD_PARAMS) -ba $(RPM_SPECFILE)
endif
endif
