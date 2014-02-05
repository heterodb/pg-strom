# Makefile of pg_strom
MODULE_big = pg_strom
OBJS  = main.o shmem.o debug.o \
	opencl_entry.o opencl_serv.o

PG_CPPFLAGS = -Werror -Wall -O0 -DPGSTROM_DEBUG

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
