# Makefile of pg_strom
MODULE_big = pg_strom
OBJS = main.o shmem.o opencl_entry.o

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
