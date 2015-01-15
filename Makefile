# Makefile of pg_strom
EXTENSION = pg_strom
DATA = src/pg_strom--1.0.sql

# Source file of CPU portion
STROM_OBJS = main.o shmem.o codegen.o mqueue.o restrack.o grafter.o \
        datastore.o gpuscan.o gpuhashjoin.o gpupreagg.o \
        opencl_entry.o opencl_serv.o opencl_devinfo.o opencl_devprog.o
# Source file of GPU portion
OPENCL_OBJS = opencl_common.o \
	opencl_gpuscan.o \
	opencl_gpupreagg.o \
	opencl_hashjoin.o \
	opencl_mathlib.o \
	opencl_textlib.o \
	opencl_timelib.o \
	opencl_numeric.o
OPENCL_SOURCES = $(addprefix src/,$(OPENCL_OBJS:.o=.c))

# Header and Libraries of OpenCL (to be autoconf?)
IPATH_LIST := /usr/include \
	/usr/local/cuda/include \
	/opt/AMDAPP*/include
LPATH_LIST := /usr/lib64 \
	/usr/lib \
	/usr/local/cuda/lib64 \
	/usr/local/cuda/lib
IPATH := $(shell for x in $(IPATH_LIST);	\
           do test -e "$$x/CL/cl.h" && (echo -I $$x; break); done)
LPATH := $(shell for x in $(LPATH_LIST);	\
           do test -e "$$x/libOpenCL.so" && (echo -L $$x; break); done)

# Module definition
MODULE_big = pg_strom
OBJS =  $(addprefix src/,$(STROM_OBJS)) \
	$(addprefix src/,$(OPENCL_OBJS))

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
SHLIB_LINK := $(LPATH)
EXTRA_CLEAN := $(OPENCL_SOURCES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(OPENCL_SOURCES): $(OPENCL_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(@:src/%.c=%)_code ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h; \
	  echo ";") > $@

