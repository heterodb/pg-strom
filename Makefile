# pg_boost Makefile

MODULE_big = pg_boost
OBJS = pg_boost.o shmmgr.o

PG_CPPFLAGS += -DPG_DEBUG
SHLIB_LINK += -lpthread
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

SHLIB_LINK += $(filter -lpthread, $(LIBS))
