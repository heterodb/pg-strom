# Load the common definition
ifndef STROM_BUILD_ROOT
STROM_BUILD_ROOT = ..
endif

ifdef MAKEFILE_PARAMDEF_ONLY
include $(STROM_BUILD_ROOT)/Makefile
else
MAKEFILE_PARAMDEF_ONLY = 1
include $(STROM_BUILD_ROOT)/Makefile
undefine MAKEFILE_PARAMDEF_ONLY
endif

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

ifndef MAKEFILE_PARAMDEF_ONLY
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
endif	# MAKEFILE_PARAMDEF_ONLY

