#
# included by other Makefile?
#
ifndef MAKEFILE_PARAMDEF_ONLY
MAKEFILE_PARAMDEF_ONLY = 0
endif

#
# Common definitions for PG-Strom Makefile
#
PG_CONFIG=pg_config

#
# PG-Strom versioning
#
PGSTROM_VERSION=1.0devel
PGSTROM_VERSION_NUM=$(shell echo $(PGSTROM_VERSION)			\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')
PG_VERSION_NUM=$(shell $(PG_CONFIG) --version | awk '{print $$NF}'	\
	| sed -e 's/\./ /g' -e 's/[A-Za-z].*$$//g'			\
	| awk '{printf "%d%02d%02d", $$1, $$2, (NF >=3) ? $$3 : 0}')

# Source file of CPU portion
STROM_OBJS = main.o codegen.o datastore.o aggfuncs.o \
		cuda_control.o cuda_program.o cuda_mmgr.o \
		gpuscan.o gpujoin.o gpupreagg.o gpusort.o

# Source file of GPU portion
CUDA_OBJS = cuda_common.o \
	cuda_dynpara.o \
	cuda_gpuscan.o \
	cuda_gpujoin.o \
	cuda_gpupreagg.o \
	cuda_gpusort.o \
	cuda_mathlib.o \
	cuda_textlib.o \
	cuda_timelib.o \
	cuda_numeric.o \
	cuda_money.o   \
	cuda_terminal.o
CUDA_SOURCES = $(CUDA_OBJS:.o=.c)

ifeq ($(MAKEFILE_PARAMDEF_ONLY), 0)
#
# Extra files to be cleaned
#
EXTRA_CLEAN_SRC=$(CUDA_SOURCES)
EXTRA_CLEAN_DOC=

all:
	$(MAKE) -C src $@ $(MAKEFLAGS)

check installcheck:
	$(MAKE) -C test $* $(MAKEFLAGS)

html:
	$(MAKE) -C doc $@ $(MAKEFLAGS)

%:
	$(MAKE) -C src $* $(MAKEFLAGS)

clean:
	$(MAKE) -C src clean
	$(MAKE) -C doc clean
	$(MAKE) -C test clean

endif	# MAKEFILE_PARAMDEF_ONLY
