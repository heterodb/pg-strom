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
             float2.o tinyint.o aggfuncs.o
GENERATED-HEADERS = gpu_devattrs.h githash.c
STROM_HEADERS = arrow_defs.h arrow_ipc.h float2.h

#
# Githash.c checker
#
GITHASH_CHECKS := $(shell grep -q \"$(PGSTROM_GITHASH)\" githash.c 2>/dev/null || rm -f githash.c)

#
# Source of NVIDIA GPU device code
#
include Makefile.cuda

#
# Installation Scripts
#
__STROM_SQL = pg_strom--4.0--5.0.sql pg_strom--5.0.sql pg_strom--5.0--5.1.sql
STROM_SQL = $(addprefix sql/,$(__STROM_SQL))

#
# Flags to build
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -D__PGSTROM_MODULE__=1
PGSTROM_FLAGS += -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"

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
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"   \
        -DPGINCLUDEDIR=\"$(shell $(PG_CONFIG) --includedir-server)\" \
        -DCUDA_MAXREGCOUNT=$(MAXREGCOUNT)                 \
        -DCUDA_CORE_FILES="\"$(__CUDA_CORE_FILES)\""      \
        -DCUDA_CORE_HEADERS="\"$(__CUDA_CORE_HEADERS)\""  \
        -DCUDA_TOOLKIT_BASEDIR="\"$(CUDA_PATH)\""
#
# Pre-built Fatbin (only debugging purpose)
#
ifeq ($(WITH_FATBIN),1)
DATA_built = $(CUDA_FATBIN)
PGSTROM_FLAGS += -DWITH_FATBIN=1
endif

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
MODULEDIR  = pg_strom
DATA = $(STROM_SQL) ../LICENSE Makefile.cuda \
       $(CUDA_SRCS) $(CUDA_HEADERS) $(STROM_HEADERS)
OBJS = $(STROM_OBJS)

EXTRA_CLEAN = $(CUDA_OBJS) $(GENERATED-HEADERS) \
              $(shell ls -d pgstrom-gpucode-V*-*.fatbin 2>/dev/null)
EXTENSION = pg_strom
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(CUDA_IPATH)
SHLIB_LINK := -L $(CUDA_LPATH) -lcuda

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
%.o: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_CFLAGS)  --device-c -o $@ $<

$(CUDA_FATBIN): $(CUDA_OBJS)
	$(NVCC) $(NVCC_LDFLAGS) --device-link --fatbin -o $@ $(CUDA_OBJS)

fatbin: $(CUDA_FATBIN)
