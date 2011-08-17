# pg_boost Makefile

MODULE_big = pg_boost
OBJS = pg_boost.o msegment.o

PG_CPPFLAGS = -DPG_DEBUG
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

SHLIB_LINK += $(filter -lpthread, $(LIBS))
