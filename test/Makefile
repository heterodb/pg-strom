# Load the common definition
PGSTROM_MAKEFILE_IN_SUBDIR := 1
include ../Makefile

# Regression test options
REGRESS = --schedule=parallel_schedule
REGRESS_OPTS = --inputdir=.
ifdef TEMP_INSTANCE
	REGRESS_OPTS += --temp-instance=tmp_check
	ifndef CPUTEST 
		REGRESS_OPTS += --temp-config=enable.conf
	else
		REGRESS_OPTS += --temp-config=disable.conf
	endif
endif

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
