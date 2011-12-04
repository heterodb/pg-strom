# pg_rapid Makefile

MODULE = pg_strom
OBJS = pg_strom.o utilcmds.o blkload.o plan.o scan.o cuda_api.o

CUDAROOT := /usr/local/cuda
NVCC := $(shell which nvcc 2>/dev/null || echo $(CUDAROOT)/bin/nvcc)
PG_CONFIG = pg_config

NVCFLAGS  := -O2 --shared -I. -D_GNU_SOURCE
NVCFLAGS  += -I$(shell $(PG_CONFIG) --includedir-server)
NVCFLAGS  += -Xcompiler=-Wall -Xcompiler=-g -Xcompiler=-fpic
NVLDFLAGS := -Xlinker=--as-needed
NVLDFLAGS += -Xlinker=-rpath,$(shell $(PG_CONFIG) --libdir)
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib64
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib

all: $(MODULE).so

install: $(MODULE).so
	install -m 755 $^ $(shell $(PG_CONFIG) --pkglibdir)

clean:
	rm -f $(MODULE).so $(OBJS)

$(MODULE).so: $(OBJS)
	$(NVCC) $(NVCFLAGS) -o $@ $^ $(NVLDFLAGS)
.c.o:
	$(NVCC) $(NVCFLAGS) -c $< $(NVLDFLAGS)
.cu.o:
	$(NVCC) $(NVCFLAGS) -c $< $(NVLDFLAGS)
.SUFFIXES: .c .cu .o
