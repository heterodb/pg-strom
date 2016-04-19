#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG := $(shell which pg_config)
PYTHON_CMD := $(shell which python)

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = .
endif

# Custom configurations if any
-include $(STROM_BUILD_ROOT)/Makefile.custom

#
# PG-Strom versioning
#
PGSTROM_VERSION=1.0devel
PGSTROM_VERSION_NUM=$(shell echo $(PGSTROM_VERSION)			\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')
PG_VERSION_NUM=$(shell $(PG_CONFIG) --version | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')

#
# Source file of CPU portion
#
__STROM_OBJS = main.o codegen.o datastore.o aggfuncs.o \
		cuda_control.o cuda_program.o cuda_mmgr.o \
		gpuscan.o gpujoin.o gpupreagg.o gpusort.o
STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__STROM_OBJS))

#
# Source file of GPU portion
#
__CUDA_OBJS = cuda_common.o \
	cuda_dynpara.o \
	cuda_gpuscan.o \
	cuda_gpujoin.o \
	cuda_gpupreagg.o \
	cuda_gpusort.o \
	cuda_mathlib.o \
	cuda_textlib.o \
	cuda_timelib.o \
	cuda_numeric.o \
	cuda_money.o   \
	cuda_terminal.o
CUDA_OBJS = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__CUDA_OBJS))
CUDA_SOURCES = $(CUDA_OBJS:.o=.c)

__GPUINFO_CMD = gpuinfo
GPUINFO_CMD = $(addprefix $(STROM_BUILD_ROOT)/src/, $(__GPUINFO_CMD))

#
# Source file of HTML document
#
__HTML_TEMPLATE = template.src.html
__HTML_SOURCES = manual.src.html \
	install.src.html \
	tutrial.src.html \
	release.src.html
__IMAGE_SOURCES = lang_en.png \
	lang_ja.png \
	icon-warning.png \
	icon-caution.png \
	icon-hint.png \
	pgstrom-install-download-zip.png \
	cuda-install-target.png
__MANUAL_CSS = manual.css
__MENUGEN_PY = menugen.py

HTML_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__HTML_SOURCES))
HTML_FILES = $(addprefix $(STROM_BUILD_ROOT)/doc/html/, $(__HTML_SOURCES:.src.html=.html))
HTML_TEMPLATE = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__HTML_TEMPLATE))
IMAGE_SOURCES = $(addprefix $(STROM_BUILD_ROOT)/doc/html/figs/, $(__IMAGE_SOURCES))
MANUAL_CSS = $(addprefix $(STROM_BUILD_ROOT)/doc/html/css/, $(__MANUAL_CSS))
MENUGEN_PY = $(addprefix $(STROM_BUILD_ROOT)/doc/, $(__MENUGEN_PY))

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
SCRIPTS_built = $(GPUINFO_CMD)
# Extra files to be cleaned
EXTRA_CLEAN = $(CUDA_SOURCES) $(HTML_FILES) $(GPUINFO_CMD) \
	$(shell test pg_strom.control -ef $(addprefix $(STROM_BUILD_ROOT)/src/, pg_strom.control) || echo pg_strom.control)

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

$(GPUINFO_CMD): $(addsuffix .c,$(GPUINFO_CMD))
	$(CC) $(CFLAGS) $^ $(PGSTROM_FLAGS) -I $(IPATH) -L $(LPATH) -lcuda -o $@$(X)

$(HTML_FILES): $(HTML_SOURCES) $(HTML_TEMPLATE)
	@$(MKDIR_P) $(STROM_BUILD_ROOT)/doc/html
	$(PYTHON_CMD) $(MENUGEN_PY) \
		-t $(HTML_TEMPLATE) \
		-v '$(PGSTROM_VERSION)' \
		-m $(addprefix $(STROM_BUILD_ROOT)/doc/, $(notdir $(basename $@)).src.html) \
		$(HTML_SOURCES) > $@

html: $(HTML_FILES)

endif
