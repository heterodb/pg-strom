# Makefile of pg_strom
EXTENSION = pg_strom
DATA = src/pg_strom--1.0.sql

# Source file of CPU portion
STROM_OBJS = main.o codegen.o grafter.o datastore.o \
		cuda_control.o cuda_program.o cuda_mmgr.o \
		gpuscan.o #gpuhashjoin.o gpupreagg.o

# Source file of GPU portion
CUDA_OBJS = cuda_common.o \
	cuda_gpuscan.o \
	cuda_gpupreagg.o \
	cuda_hashjoin.o \
	cuda_gpusort.o \
	cuda_mathlib.o \
	cuda_textlib.o \
	cuda_timelib.o \
	cuda_numeric.o
CUDA_SOURCES = $(addprefix src/,$(CUDA_OBJS:.o=.c))

# Header and Libraries of CUDA
IPATH_LIST := /usr/local/cuda/include
LPATH_LIST := /usr/local/cuda/lib64 /usr/local/cuda/lib

IPATH := $(shell for x in $(IPATH_LIST);	\
           do test -e "$$x/cuda.h" && (echo -I $$x; break); done)
LPATH := $(shell for x in $(LPATH_LIST);	\
           do test -e "$$x/libcuda.so" && (echo -L $$x; break); done)

# Module definition
MODULE_big = pg_strom
OBJS =  $(addprefix src/,$(STROM_OBJS)) $(addprefix src/,$(CUDA_OBJS))

# Regression test options
REGRESS = --schedule=test/parallel_schedule
ifndef PGSQL_BUILD_DIR 
	REGRESS_OPTS = --inputdir=test
else
	RET := $(shell ln -sn $(CURDIR) $(PGSQL_BUILD_DIR)/contrib/pg_strom)
	REGRESS_OPTS = --inputdir=test \
               --top-builddir=$(PGSQL_BUILD_DIR) \
               --extra-install=contrib/pg_strom \
               --temp-install=tmp_check
	ifndef CPUTEST 
		REGRESS_OPTS += --temp-config=test/enable.conf
	else
		REGRESS_OPTS += --temp-config=test/disable.conf
	endif
endif

PG_CONFIG = pg_config
PGSTROM_DEBUG := $(shell $(PG_CONFIG) --configure | \
	grep -q "'--enable-debug'" && \
	echo "-Wall -DPGSTROM_DEBUG=1")
PG_CPPFLAGS := $(PGSTROM_DEBUG) $(IPATH)
SHLIB_LINK := $(LPATH) -lcuda
EXTRA_CLEAN := $(CUDA_SOURCES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(CUDA_SOURCES): $(CUDA_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(@:src/%.c=%)_code ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h; \
	  echo ";") > $@
