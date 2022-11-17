#
# PG-Strom Makefile
#
PG_CONFIG ?= pg_config

ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT=..
endif

#
# PG-Strom version
#
PGSTROM_VERSION := 5.0
PGSTROM_RELEASE := devel

#
# Source of PG-Strom host code
#
__STROM_OBJS = main.o extra.o codegen.o misc.o executor.o brin.o relscan.o \
               gpu_scan.o gpu_device.o gpu_direct.o gpu_service.o \
               dpu_device.o dpu_scan.o \
               float2.o tinyint.o
STROM_OBJS = $(addprefix $(STROM_BUILD_ROOT)/next/,$(__STROM_OBJS))

#
# Source of NVIDIA GPU device code
#
include $(STROM_BUILD_ROOT)/next/Makefile.cuda
__CUDA_OBJS = cuda_common cuda_gpuscan \
              xpu_common xpu_basetype xpu_numeric xpu_timelib xpu_textlib xpu_misclib
__CUDA_HEADERS = cuda_common.h xpu_common.h xpu_opcodes.h \
                 xpu_basetype.h xpu_numeric.h xpu_textlib.h xpu_timelib.h xpu_misclib.h
__CUDA_OPT_OBJS = $(addsuffix .fatbin,$(__CUDA_OBJS))
__CUDA_DBG_OBJS = $(addsuffix .debug.fatbin,$(__CUDA_OBJS))
CUDA_HEADERS = $(addprefix $(STROM_BUILD_ROOT)/next/,$(__CUDA_HEADERS))
CUDA_OPT_OBJS = $(addprefix $(STROM_BUILD_ROOT)/next/,$(__CUDA_OPT_OBJS))
CUDA_DBG_OBJS = $(addprefix $(STROM_BUILD_ROOT)/next/,$(__CUDA_DBG_OBJS))
CUDA_OPT_MODULE = $(STROM_BUILD_ROOT)/next/pgstrom-core.fatbin
CUDA_DBG_MODULE = $(STROM_BUILD_ROOT)/next/pgstrom-core.debug.fatbin


#
# Installation Scripts
#
__STROM_SQL = pg_strom--5.0.sql
STROM_SQL = $(addprefix $(STROM_BUILD_ROOT)/next/,$(__STROM_SQL))

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

PGSTROM_DEBUG = 1
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
DATA_built = $(CUDA_OPT_OBJS) $(CUDA_DBG_OBJS)
EXTRA_CLEAN = $(CUDA_OPT_OBJS) $(CUDA_DBG_OBJS)
EXTENSION = pg_strom

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

#
# GPU Device Code
#
%.fatbin: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
%.debug.fatbin: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o $@ $<
