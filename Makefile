# Makefile of pg_strom
MODULE_big = pg_strom
OBJS  = main.o shmem.o debug.o \
	opencl_entry.o opencl_serv.o opencl_devinfo.o

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGSTROM_DEBUG := $(shell $(PG_CONFIG) --configure | grep -q "'--enable-debug'" && echo "-Werror -Wall -O0 -DPGSTROM_DEBUG=1")
PG_CPPFLAGS := $(PGSTROM_DEBUG)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
