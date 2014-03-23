# Makefile of pg_strom
MODULE_big = pg_strom
OBJS  = main.o shmem.o debug.o \
	codegen_expr.o mqueue.o \
	opencl_entry.o opencl_serv.o opencl_devinfo.o opencl_devprog.o \
	opencl_common.o

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGSTROM_DEBUG := $(shell $(PG_CONFIG) --configure | grep -q "'--enable-debug'" && echo "-Werror -Wall -O0 -DPGSTROM_DEBUG=1")
PG_CPPFLAGS := $(PGSTROM_DEBUG)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

opencl_common.c: opencl_common.h
	(echo "const char *pgstrom_opencl_common_head ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@
