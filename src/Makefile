# Load the common definition
PGSTROM_MAKEFILE_IN_SUBDIR := 1
include ../Makefile
-include ./Makefile.custom

#
# Definition of PG-Strom Extension
#
EXTENSION = pg_strom
ifeq ($(shell test $(PG_VERSION_NUM) -ge 90600; echo $??),0)
DATA = pg_strom--1.0.sql
else
DATA = pg_strom--1.0.sql
endif

# Header and Libraries of CUDA
CUDA_PATH_LIST := /usr/local/cuda /usr/local/cuda-*
CUDA_PATH := $(shell for x in $(CUDA_PATH_LIST);	\
	       do test -e "$$x/include/cuda.h" && echo $$x; done | head -1)
IPATH := $(CUDA_PATH)/include
LPATH := $(CUDA_PATH)/lib64

# Module definition
MODULE_big = pg_strom
OBJS =  $(STROM_OBJS) $(CUDA_OBJS)

# Support utilities
SCRIPTS_built = gpuinfo

# Regression test options
REGRESS = --schedule=test/parallel_schedule
REGRESS_OPTS = --inputdir=test
ifdef TEMP_INSTANCE
	REGRESS_OPTS += --temp-instance=tmp_check
	ifndef CPUTEST 
		REGRESS_OPTS += --temp-config=test/enable.conf
	else
		REGRESS_OPTS += --temp-config=test/disable.conf
	endif
endif

PGSTROM_FLAGS := $(shell $(PG_CONFIG) --configure | \
  awk '/'--enable-debug'/ {print "-Werror"}')
PGSTROM_FLAGS += -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"
PGSTROM_FLAGS += -DPGSTROM_VERSION_NUM=$(PGSTROM_VERSION_NUM)
PGSTROM_FLAGS += -DPGSTROM_BUILD_DATE=\"$(PGSTROM_BUILD_DATE)\"
PGSTROM_FLAGS += -DCUDA_INCLUDE_PATH=\"$(IPATH)\"
PGSTROM_FLAGS += -DCUDA_LIBRARY_PATH=\"$(LPATH)\"
PGSTROM_FLAGS += -DCMD_GPUINFO_PATH=\"$(shell $(PG_CONFIG) --bindir)/gpuinfo\"
# We assume to put the following line in Makefile.custom according to
# the purpose of this build.
#   PGSTROM_FLAGS_CUSTOM := -DPGSTROM_DEBUG=1 -g -O0
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(IPATH)
SHLIB_LINK := -L $(LPATH) -lnvrtc -lcuda

EXTRA_CLEAN := $(EXTRA_CLEAN_SRC)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(CUDA_SOURCES): $(CUDA_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(@:%.c=%)_code ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h; \
	  echo ";") > $@

gpuinfo: gpuinfo.c
	$(CC) $(CFLAGS) gpuinfo.c $(PGSTROM_FLAGS) -I $(IPATH) -L $(LPATH) -lcuda -o $@$(X)
