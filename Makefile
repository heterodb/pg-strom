# pg_boost Makefile

MODULE_big = pg_boost
OBJS = shmseg.o

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

SHLIB_LINK += $(filter -lpthread, $(LIBS))
