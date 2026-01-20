#
# PG-Strom Makefile
#
include ../Makefile.common

#
# Makefile for arrow-tools
#
CXX          ?= g++
PG_CONFIG    ?= pg_config

HAS_PG_CONFIG = $(shell which $(PG_CONFIG)>/dev/null 2>&1 && echo yes)
HAS_LIBARROW = $(shell pkgconf --exists arrow>/dev/null 2>&1 && echo -n yes)
HAS_PARQUET = $(shell pkgconf --exists parquet>/dev/null 2>&1 && echo -n yes)

ifeq ($(HAS_LIBARROW),yes)
ALL_PROGS = vcf2arrow tsv2arrow arrow2csv arrow2tsv
ifeq ($(HAS_PG_CONFIG),yes)
ALL_PROGS += pg2arrow
endif
else
$(error libarrow is missing)
endif

PG2ARROW_OBJS    = pg2arrow.o arrow_meta.o
VCF2ARROW_OBJS   = vcf2arrow.o
TSV2ARROW_OBJS   = tsv2arrow.o
ARROW2CSV_OBJS   = arrow2csv.o arrow_meta.o
ARROW2TSV_OBJS   = arrow2tsv.o
CLEAN_OBJS = $(PG2ARROW_OBJS) $(TSV2ARROW_OBJS) $(ARROW2TSV_OBJS) \
             $(ARROW2CSV_OBJS) $(VCF2ARROW_OBJS) $(ALL_PROGS)

CFLAGS = -O2 -fPIC -g -Wall -I. -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 \
         -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"

ifeq ($(HAS_PG_CONFIG),yes)
CFLAGS += -I $(shell $(PG_CONFIG) --includedir)
endif
ifeq ($(DEBUG),1)
CFLAGS += -g -O0
endif

PG2ARROW_LDFLAGS = $(shell $(PG_CONFIG) --ldflags) \
                -L $(shell $(PG_CONFIG) --libdir) -lpq
ifeq ($(HAS_LIBARROW),yes)
CPPFLAGS += -DHAS_LIBARROW=1 $(shell pkgconf --cflags arrow)
ifeq ($(HAS_PARQUET),yes)
CPPFLAGS += -DHAS_PARQUET=1  $(shell pkgconf --cflags parquet)
PG2ARROW_LDFLAGS  += $(shell pkgconf --libs parquet)
VCF2ARROW_LDFLAGS += $(shell pkgconf --libs parquet)
TSV2ARROW_LDFLAGS += $(shell pkgconf --libs parquet)
ARROW2CSV_LDFLAGS += $(shell pkgconf --libs parquet)
ARROW2TSV_LDFLAGS += $(shell pkgconf --libs parquet)
else
PG2ARROW_LDFLAGS  += $(shell pkgconf --libs arrow)
VCF2ARROW_LDFLAGS += $(shell pkgconf --libs arrow)
TSV2ARROW_LDFLAGS += $(shell pkgconf --libs arrow)
ARROW2CSV_LDFLAGS += $(shell pkgconf --libs arrow)
ARROW2TSV_LDFLAGS += $(shell pkgconf --libs arrow)
endif
endif
CPPFLAGS += -std=c++17 $(CFLAGS) -Wno-sign-compare

PREFIX		?= /usr/local
BINDIR		?= $(PREFIX)/bin

all: $(ALL_PROGS)

#
# Only if pg_config is available
#
ifeq ($(HAS_PG_CONFIG),yes)
pg2arrow: $(PG2ARROW_OBJS)
	$(CXX) $(CPPFLAGS) -o $@ $(PG2ARROW_OBJS) $(PG2ARROW_LDFLAGS)

install-pg2arrow: pg2arrow
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 $< $(DESTDIR)$(BINDIR)
endif

#
# VCF2Arrow
#
vcf2arrow: $(VCF2ARROW_OBJS)
	$(CXX) -o $@ $(VCF2ARROW_OBJS) $(VCF2ARROW_LDFLAGS)

install-vcf2arrow: vcf2arrow
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 vcf2arrow $(DESTDIR)$(BINDIR)

#
# Arrow2CSV
#
arrow2csv: $(ARROW2CSV_OBJS)
	$(CXX) $(CPPFLAGS) -o $@ $(ARROW2CSV_OBJS) $(ARROW2CSV_LDFLAGS)

install-arrow2csv: arrow2csv
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 $< $(DESTDIR)$(BINDIR)

#
# Arrow2TSV
#
arrow2tsv: $(ARROW2TSV_OBJS)
	$(CXX) $(CPPFLAGS) -o $@ $(ARROW2TSV_OBJS) $(ARROW2TSV_LDFLAGS)

install-arrow2tsv: arrow2tsv
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 $< $(DESTDIR)$(BINDIR)

#
# TSV2Arrow
#
install-tsv2arrow: tsv2arrow
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 tsv2arrow $(DESTDIR)$(BINDIR)

tsv2arrow: $(TSV2ARROW_OBJS)
	$(CXX) $(CPPFLAGS) -o $@ $< $(TSV2ARROW_LDFLAGS)

.cpp.o:
	$(CXX) $(CPPFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(addprefix install-,$(ALL_PROGS))

clean:
	rm -f $(CLEAN_OBJS)
