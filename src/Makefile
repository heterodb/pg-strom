#
# PG-Strom Makefile
#
PG_CONFIG ?= pg_config

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT=.
endif

#
# PG-Strom version
#
PGSTROM_VERSION := 5.0
PGSTROM_RELEASE := devel

#
# Source of PG-Strom host code
#
__STROM_OBJS = main.o extra.o codegen.o misc.o executor.o \
               gpu_device.o gpu_scan.o gpu_join.o gpu_preagg.o \
               dpu_device.o dpu_scan.o dpu_join.o dpu_preagg.o \
               relscan.o brin.o gist.o gpu_service.o \
               arrow_fdw.o arrow_nodes.o \
               pcie.o float2.o tinyint.o aggfuncs.o
STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/,$(__STROM_OBJS))

GPU_DEVATTRS_H = $(STROM_BUILD_ROOT)/gpu_devattrs.h
GENERATED-HEADERS = $(GPU_DEVATTRS_H)

#
# Source of NVIDIA GPU device code
#
include $(STROM_BUILD_ROOT)/Makefile.cuda
__CUDA_OBJS = xpu_common cuda_gpuscan cuda_gpujoin cuda_gpupreagg \
              xpu_basetype xpu_numeric xpu_timelib xpu_textlib xpu_misclib \
              xpu_jsonlib xpu_postgis
__CUDA_HEADERS = cuda_common.h xpu_common.h xpu_opcodes.h xpu_basetype.h \
                 xpu_numeric.h xpu_textlib.h xpu_timelib.h xpu_misclib.h \
                 xpu_jsonlib.h xpu_postgis.h
__CUDA_OPT_OBJS = $(addsuffix .fatbin,$(__CUDA_OBJS))
CUDA_HEADERS = $(addprefix $(STROM_BUILD_ROOT)/,$(__CUDA_HEADERS))
CUDA_OPT_OBJS = $(addprefix $(STROM_BUILD_ROOT)/,$(__CUDA_OPT_OBJS))
CUDA_OPT_MODULE = $(STROM_BUILD_ROOT)/pgstrom-core.fatbin


#
# Installation Scripts
#
__STROM_SQL = pg_strom--5.0.sql
STROM_SQL = $(addprefix $(STROM_BUILD_ROOT)/sql/,$(__STROM_SQL))

#
# GitHash to build
#
ifdef PGSTROM_GITHASH
ifeq ($(PGSTROM_GITHASH),HEAD)
PGSTROM_GITHASH = $(shell git rev-parse HEAD)
endif
else
ifeq ($(shell test -e $(STROM_BUILD_ROOT)/.git/config && echo -n 1),1)
PGSTROM_GITHASH = $(shell git rev-parse HEAD)
ifneq ($(shell git diff | wc -l),0)
PGSTROM_GITHASH_SUFFIX = ::local_changes
endif
else
ifeq ($(shell test -e $(STROM_BUILD_ROOT)/GITHASH && echo -n 1),1)
PGSTROM_GITHASH = $(shell cat $(STROM_BUILD_ROOT)/GITHASH)
else
PGSTROM_GITHASH = HEAD
endif
endif
endif

#
# Flags to build
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -D__PGSTROM_MODULE__=1
PGSTROM_FLAGS += "-DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\""

#PGSTROM_DEBUG = 1
ifeq ($(PGSTROM_DEBUG),1)
PGSTROM_FLAGS += -g -O0 -DPGSTROM_DEBUG_BUILD=1
endif
PGSTROM_FLAGS += -D__STROM_HOST__=1
ifeq ($(shell uname -m),aarch64)
PGSTROM_FLAGS += -DHAVE_FLOAT2 -mfp16-format=ieee
endif
PGSTROM_FLAGS += -DPGSTROM_GITHASH=\"$(PGSTROM_GITHASH)$(PGSTROM_GITHASH_SUFFIX)\"
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"
PGSTROM_FLAGS += -DCUDA_MAXREGCOUNT=$(MAXREGCOUNT)
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
PGSTROM_FLAGS += -DCUDA_BUILTIN_OBJS="\"$(__CUDA_OBJS)\""
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(CUDA_IPATH)
SHLIB_LINK := -L $(CUDA_LPATH) -lcuda

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
MODULEDIR  = pg_strom
DATA = $(STROM_SQL)
OBJS = $(STROM_OBJS)
DATA_built = $(CUDA_OPT_OBJS)
EXTRA_CLEAN = $(DATA_built) $(GENERATED-HEADERS)
EXTENSION = pg_strom

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

#
# Device Attributes
#
submake-generated-headers: $(GENERATED-HEADERS)

$(GPU_DEVATTRS_H): $(CUDA_IPATH)/cuda.h
	cat $(CUDA_IPATH)/cuda.h | \
	grep -E '^[ ]+CU_DEVICE_ATTRIBUTE_' | \
	grep -v -E 'CU_DEVICE_ATTRIBUTE_MAX$$' | \
	grep -v 'Deprecated[,\.]' | \
	sed -e 's|[ ]*CU_DEVICE_ATTRIBUTE_|DEV_ATTR(|g' \
	    -e 's| =.*/\*\*<[ ]*|, "|g' \
	    -e 's|[ ]*\*/|")|g' > $@

#
# GPU Device Code
#
%.fatbin: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
