# Makefile of pg_strom
MODULE_big = pg_strom
OBJS = main.o shmseg.o plan.o exec.o utilcmds.o blkload.o \
		cuda_serv.o openmp_serv.o
DATA_built = cuda_kernel.ptx

CUDA_DIR := /usr/local/cuda
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_NVCC := $(CUDA_DIR)/bin/nvcc

PG_CPPFLAGS := -I$(CUDA_INCLUDE)
SHLIB_LINK := -lcuda -Wl,-rpath,'$(CUDA_DIR)/lib64' -Wl,-rpath,'$(CUDA_DIR)/lib'

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

cuda_kernel.ptx: cuda_kernel.gpu cuda_cmds.h
	$(CUDA_NVCC) -ptx -arch=compute_20 $< -o $@
