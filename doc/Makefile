# Load the common definition
PGSTROM_MAKEFILE_IN_SUBDIR := 1
include ../Makefile

EXTRA_CLEAN := $(EXTRA_CLEAN_DOC)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifndef JADE
JADE = jade
endif

SGMLINCLUDE = -D .

SPFLAGS += -wall -wno-unused-param -wno-empty -wfully-tagged

JADE.html.call = $(JADE) $(JADEFLAGS) $(SPFLAGS) $(SGMLINCLUDE) $(CATALOG) -d stylesheet.dsl -t sgml -i output-html

JADE.html.single.call = $(JADE.html.call) -V nochunks -V rootchunk -V '(define %root-filename% \#f)' -V '(define use-output-dir \#f)'

ALLSGML := $(wildcard $(srcdir)/*.sgml)
# to build bookindex
ALMOSTALLSGML := $(filter-out %bookindex.sgml,$(ALLSGML))
GENERATED_SGML = version.sgml bookindex.sgml

all: html

html: html-stamp

html.single: html.single-stamp

html-stamp: pgstrom.sgml $(ALLSGML) $(GENERATED_SGML) stylesheet.dsl website-docs.css
	$(MKDIR_P) html
	$(JADE.html.call) -i include-index $<
	cp stylesheet.css website-docs.css html
	touch $@

html.single-stamp: pgstrom.sgml $(ALLSGML) $(GENERATED_SGML) stylesheet.dsl website-docs.css
	$(MKDIR_P) html
	$(JADE.html.single.call) -i include-index $<
	cp stylesheet.css website-docs.css html
	touch $@

version.sgml:
	@echo "<!ENTITY stromversion \"$(PGSTROM_VERSION)\">" > $@

HTML.index: pgstrom.sgml $(ALMOSTALLSGML) stylesheet.dsl
	@$(MKDIR_P) html
	$(JADE.html.call) -V html-index $<

bookindex.sgml: HTML.index
ifdef COLLATEINDEX
	LC_ALL=C $(PERL) $(COLLATEINDEX) -f -g -i 'bookindex' -o $@ $<
else
	collateindex.pl $< $@
endif

.PHONY: html html.single all
