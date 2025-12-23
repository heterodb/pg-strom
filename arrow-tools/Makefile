#
# PG-Strom Makefile
#
include ../Makefile.common

#
# Makefile for arrow-tools
#
CXX          ?= g++
PG_CONFIG    ?= pg_config
MYSQL_CONFIG ?= mysql_config

HAS_PG_CONFIG = $(shell which $(PG_CONFIG)>/dev/null 2>&1 && echo yes)
HAS_MYSQL_CONFIG = $(shell which $(MYSQL_CONFIG)>/dev/null 2>&1 && echo yes)
HAS_PF_RING = $(shell test -e /usr/include/pfring.h && echo -n yes)
HAS_LIBARROW = $(shell pkgconf --exists arrow>/dev/null 2>&1 && echo -n yes)
HAS_PARQUET = $(shell pkgconf --exists parquet>/dev/null 2>&1 && echo -n yes)

ALL_PROGS = arrow2csv-legacy vcf2arrow-legacy
ifeq ($(HAS_PG_CONFIG),yes)
ALL_PROGS += pg2arrow-legacy
ifeq ($(HAS_LIBARROW),yes)
ALL_PROGS += pg2arrow
endif
endif
ifeq ($(HAS_MYSQL_CONFIG),yes)
ALL_PROGS += mysql2arrow-legacy
endif
ifeq ($(HAS_PF_RING),yes)
ALL_PROGS += pcap2arrow
endif
ifeq ($(HAS_LIBARROW),yes)
ALL_PROGS += vcf2arrow tsv2arrow arrow2csv
endif

PG2ARROW_OBJS = pg2arrow.o arrow_meta.o
PG2ARROW_LEGACY_OBJS = __pgsql2arrow.o pgsql_client.o \
                       arrow_nodes.o arrow_write.o arrow_pgsql.o
MYSQL2ARROW_LEGACY_OBJS = __mysql2arrow.o mysql_client.o \
                          arrow_nodes.o arrow_write.o
PCAP2ARROW_OBJS  = pcap2arrow.o arrow_nodes.o arrow_write.o
VCF2ARROW_LEGACY_OBJS = vcf2arrow_legacy.o arrow_nodes.o arrow_write.o
VCF2ARROW_OBJS   = vcf2arrow.o
TSV2ARROW_OBJS   = tsv2arrow.o
ARROW2CSV_OBJS   = arrow2csv.o arrow_meta.o
ARROW2CSV_LEGACY_OBJS = arrow2csv_legacy.o arrow_nodes.o
CLEAN_OBJS = $(PG2ARROW_OBJS) $(TSV2ARROW_OBJS) $(ARROW2CSV_OBJS) $(VCF2ARROW_OBJS) \
             $(PCAP2ARROW_OBJS) $(VCF2ARROW_LEGACY_OBJS) \
             $(PG2ARROW_LEGACY_OBJS) $(MYSQL2ARROW_LEGACY_OBJS) \
             $(ARROW2CSV_LEGACY_OBJS) $(ALL_PROGS)

CFLAGS = -O2 -fPIC -g -Wall -I. -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 \
         -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"

ifeq ($(HAS_PG_CONFIG),yes)
CFLAGS += -I $(shell $(PG_CONFIG) --includedir)
endif
ifeq ($(HAS_MYSQL_CONFIG),yes)
CFLAGS += $(shell $(MYSQL_CONFIG) --include)
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
VCF2ARROW_LDFLAGS += $(shell pkgconf --libs parquet) -lstdc++
TSV2ARROW_LDFLAGS += $(shell pkgconf --libs parquet)
ARROW2CSV_LDFLAGS += $(shell pkgconf --libs parquet)
else
PG2ARROW_LDFLAGS  += $(shell pkgconf --libs arrow)
VCF2ARROW_LDFLAGS += $(shell pkgconf --libs arrow) -lstdc++
TSV2ARROW_LDFLAGS += $(shell pkgconf --libs arrow)
ARROW2CSV_LDFLAGS += $(shell pkgconf --libs arrow)
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

pg2arrow-legacy: $(PG2ARROW_LEGACY_OBJS)
	$(CC) -o $@ $(PG2ARROW_LEGACY_OBJS) -lpq -lpthread \
	$(shell $(PG_CONFIG) --ldflags) \
	-L $(shell $(PG_CONFIG) --libdir)

install-pg2arrow-legacy: pg2arrow-legacy
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 $< $(DESTDIR)$(BINDIR)

__pgsql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__PG2ARROW__ -c -o $@ $<
endif

#
# Only if mysql_config is available
#
ifeq ($(HAS_MYSQL_CONFIG),yes)
mysql2arrow-legacy: $(MYSQL2ARROW_LEGACY_OBJS)
	$(CC) -o $@ $(MYSQL2ARROW_LEGACY_OBJS) \
	$(shell $(MYSQL_CONFIG) --libs) \
	-Wl,-rpath,$(shell $(MYSQL_CONFIG) --variable=pkglibdir)

install-mysql2arrow-legacy: mysql2arrow-legacy
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 $< $(DESTDIR)$(BINDIR)

__mysql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__MYSQL2ARROW__ -c -o $@ $<
endif

#
# Pcap2Arrow
#
ifeq ($(HAS_PF_RING),yes)
pcap2arrow: $(PCAP2ARROW_OBJS)
	$(CC) -o $@ $(PCAP2ARROW_OBJS) -lpthread -lpfring -lpcap

install-pcap2arrow: pcap2arrow
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 pcap2arrow $(DESTDIR)$(BINDIR)
endif

#
# VCF2Arrow(legacy)
#
vcf2arrow-legacy: $(VCF2ARROW_LEGACY_OBJS)
	$(CC) -o $@ $(VCF2ARROW_LEGACY_OBJS)

install-vcf2arrow-legacy: vcf2arrow-legacy
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 vcf2arrow-legacy $(DESTDIR)$(BINDIR)

#
# VCF2Arrow(legacy)
#
vcf2arrow: $(VCF2ARROW_OBJS)
	$(CC) -o $@ $(VCF2ARROW_OBJS) $(VCF2ARROW_LDFLAGS)

install-vcf2arrow: vcf2arrow
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 vcf2arrow $(DESTDIR)$(BINDIR)

#
# Arrow2CSV(legacy)
#
arrow2csv-legacy: $(ARROW2CSV_LEGACY_OBJS)
	$(CC) -o $@ $(ARROW2CSV_LEGACY_OBJS)

install-arrow2csv-legacy: arrow2csv-legacy
	mkdir -p $(DESTDIR)$(BINDIR) && \
	install -m 0755 arrow2csv-legacy $(DESTDIR)$(BINDIR)

#
# Arrow2CSV
#
arrow2csv: $(ARROW2CSV_OBJS)
	$(CXX) $(CPPFLAGS) -o $@ $(ARROW2CSV_OBJS) $(ARROW2CSV_LDFLAGS)

install-arrow2csv: arrow2csv
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
