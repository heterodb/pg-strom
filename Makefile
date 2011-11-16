# pg_rapid Makefile

MODULE = pg_rapid
OBJS = pg_rapid.o utilcmds.o blkload.o cuda.o

CUDAROOT := /usr/local/cuda
NVCC := $(shell which nvcc 2>/dev/null || echo $(CUDAROOT)/bin/nvcc)
PG_CONFIG = pg_config

NVCFLAGS  := -O2 --shared -I. -D_GNU_SOURCE
NVCFLAGS  += -I$(shell $(PG_CONFIG) --includedir-server)
NVCFLAGS  += -Xcompiler=-Wall -Xcompiler=-g -Xcompiler=-fpic
NVLDFLAGS := -Xlinker=--as-needed
NVLDFLAGS += -Xlinker=-rpath,$(shell $(PG_CONFIG) --libdir)
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib64

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
