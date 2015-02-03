# Makefile of pg_strom
EXTENSION = pg_strom
DATA = src/pg_strom--1.0.sql

# Source file of CPU portion
<<<<<<< HEAD
STROM_OBJS = main.o codegen.o grafter.o datastore.o \
		cuda_control.o cuda_program.o cuda_mmgr.o \
		gpuscan.o #gpuhashjoin.o gpupreagg.o

# Source file of GPU portion
CUDA_OBJS = device_common.o \
	device_gpuscan.o \
	device_gpupreagg.o \
	device_hashjoin.o \
	device_mathlib.o \
	device_textlib.o \
	device_timelib.o \
	device_numeric.o
CUDA_SOURCES = $(addprefix src/,$(CUDA_OBJS:.o=.c))

# Header and Libraries of CUDA
IPATH_LIST := /usr/local/cuda/include
LPATH_LIST := /usr/local/cuda/lib64 /usr/local/cuda/lib
=======
STROM_OBJS = main.o shmem.o codegen.o mqueue.o restrack.o grafter.o \
        datastore.o gpuscan.o gpuhashjoin.o gpupreagg.o gpusort.o \
        opencl_entry.o opencl_serv.o opencl_devinfo.o opencl_devprog.o
# Source file of GPU portion
OPENCL_OBJS = opencl_common.o \
	opencl_gpuscan.o \
	opencl_gpupreagg.o \
	opencl_hashjoin.o \
	opencl_gpusort.o \
	opencl_mathlib.o \
	opencl_textlib.o \
	opencl_timelib.o \
	opencl_numeric.o
OPENCL_SOURCES = $(addprefix src/,$(OPENCL_OBJS:.o=.c))
>>>>>>> 012cfb49541d9309610e97d0953cc43244d268f8

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
	echo "-Wall -DPGSTROM_DEBUG=1 -O0")
PG_CPPFLAGS := $(PGSTROM_DEBUG) $(IPATH)
SHLIB_LINK := $(LPATH) -lcuda
EXTRA_CLEAN := $(OPENCL_SOURCES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(CUDA_SOURCES): $(CUDA_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(@:src/%.c=%)_code ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h; \
	  echo ";") > $@
