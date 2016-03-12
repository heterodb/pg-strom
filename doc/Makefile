# Load the common definition
PGSTROM_MAKEFILE_IN_SUBDIR := 1
include ../Makefile

PYTHON := $(shell which python)
HTMLSOURCES := manual.src.html install.src.html tutrial.src.html \
		troubles.src.html release.src.html
HTMLFILES := $(addprefix html/,$(HTMLSOURCES:.src.html=.html))
EXTRA_CLEAN := $(HTMLFILES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

all: html

html: $(HTMLFILES)

$(HTMLFILES): $(HTMLSOURCES) template.src.html
	@$(MKDIR_P) html
	@./menugen.py -t template.src.html \
		-v '$(PGSTROM_VERSION)' \
		-m $(@:html/%.html=%.src.html) $(HTMLSOURCES) > $@
	@echo $@

.PHONY: html html.single all
