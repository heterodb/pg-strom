#
# PG-Strom Makefile
#
include ../Makefile.common

#
# Source of PG-Strom host code
#
STROM_OBJS = main.o githash.o extra.o codegen.o misc.o executor.o \
             gpu_device.o gpu_service.o dpu_device.o \
             gpu_scan.o gpu_join.o gpu_preagg.o \
             relscan.o brin.o gist.o gpu_cache.o \
             arrow_fdw.o arrow_nodes.o \
             pcie.o float2.o tinyint.o aggfuncs.o
GENERATED-HEADERS = gpu_devattrs.h githash.c

#
# Githash.c checker
#
GITHASH_CHECKS := $(shell grep -q \"$(PGSTROM_GITHASH)\" githash.c 2>/dev/null || rm -f githash.c)

#
# Source of NVIDIA GPU device code
#
include Makefile.cuda
__CUDA_OBJS = xpu_common cuda_gpuscan cuda_gpujoin cuda_gpupreagg \
              xpu_basetype xpu_numeric xpu_timelib xpu_textlib xpu_misclib \
              xpu_jsonlib xpu_postgis
CUDA_HEADERS = cuda_common.h xpu_common.h xpu_opcodes.h xpu_basetype.h \
               xpu_numeric.h xpu_textlib.h xpu_timelib.h xpu_misclib.h \
               xpu_jsonlib.h xpu_postgis.h
CUDA_OBJS = $(addsuffix .fatbin,$(__CUDA_OBJS))

#
# Installation Scripts
#
__STROM_SQL = pg_strom--5.0.sql
STROM_SQL = $(addprefix sql/,$(__STROM_SQL))

#
# Flags to build
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -D__PGSTROM_MODULE__=1
PGSTROM_FLAGS += -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"

#PGSTROM_DEBUG = 1
ifeq ($(PGSTROM_DEBUG),1)
PGSTROM_FLAGS += -g -O0 -DPGSTROM_DEBUG_BUILD=1
endif
PGSTROM_FLAGS += -D__STROM_HOST__=1
ifeq ($(shell uname -m),aarch64)
PGSTROM_FLAGS += -DHAVE_FLOAT2 -mfp16-format=ieee
endif
ifneq ($(PGSTROM_GITHASH),)
PGSTROM_FLAGS += -DPGSTROM_GITHASH=\"$(PGSTROM_GITHASH)\"
endif
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"
PGSTROM_FLAGS += -DCUDA_MAXREGCOUNT=$(MAXREGCOUNT)
PGSTROM_FLAGS += -DCUDA_BUILTIN_OBJS="\"$(__CUDA_OBJS)\""
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(CUDA_IPATH)
SHLIB_LINK := -L $(CUDA_LPATH) -lcuda

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
MODULEDIR  = pg_strom
DATA = $(STROM_SQL) ../LICENSE Makefile.cuda
OBJS = $(STROM_OBJS)
DATA_built = $(CUDA_OBJS)
EXTRA_CLEAN = $(DATA_built) $(GENERATED-HEADERS)
EXTENSION = pg_strom

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

#
# Device Attributes
#
submake-generated-headers: $(GENERATED-HEADERS)

gpu_devattrs.h: $(CUDA_IPATH)/cuda.h
	cat $(CUDA_IPATH)/cuda.h | \
	grep -E '^[ ]+CU_DEVICE_ATTRIBUTE_' | \
	grep -v -E 'CU_DEVICE_ATTRIBUTE_MAX$$' | \
	grep -v 'Deprecated[,\.]' | \
	sed -e 's|[ ]*CU_DEVICE_ATTRIBUTE_|DEV_ATTR(|g' \
	    -e 's| =.*/\*\*<[ ]*|, "|g' \
	    -e 's|[ ]*\*/|")|g' \
        -e 's| =.*$$|, NULL)|g' > $@

githash.c:
	echo "const char *pgstrom_githash_cstring = \"$(PGSTROM_GITHASH)\";" > githash.c

#
# GPU Device Code
#
%.fatbin: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
