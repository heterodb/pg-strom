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

# Module definition
MODULE_big = pg_strom
OBJS =  $(addprefix src/,$(STROM_OBJS)) \
	$(addprefix src/,$(OPENCL_OBJS))


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
PG_CPPFLAGS := $(PGSTROM_DEBUG)
EXTRA_CLEAN := $(OPENCL_SOURCES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(OPENCL_SOURCES): $(OPENCL_SOURCES:.c=.h)
	@(echo "const char *pgstrom_$(@:src/%.c=%)_code ="; \
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g' -e 's/$$/\\n"/g' < $*.h; \
	  echo ";") > $@

