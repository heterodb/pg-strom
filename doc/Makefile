# Load the common definition
PGSTROM_MAKEFILE_IN_SUBDIR := 1
include ../Makefile

PYTHON_CMD := $(shell which python)
HTMLSOURCES := manual.src.html install.src.html tutrial.src.html \
		release.src.html
HTMLFILES := $(addprefix html/,$(HTMLSOURCES:.src.html=.html))
HTML_TEMPLATE := template.src.html
MENUGEN_PY := menugen.py

EXTRA_CLEAN := $(HTMLFILES)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

all: html

html: $(HTMLFILES)

$(HTMLFILES): $(HTMLSOURCES) $(HTML_TEMPLATE)
	@$(MKDIR_P) html
	$(PYTHON_CMD) $(MENUGEN_PY)	\
		-t $(HTML_TEMPLATE)	\
		-v '$(PGSTROM_VERSION)'	\
		-m $(@:html/%.html=%.src.html) $(HTMLSOURCES) > $@

.PHONY: html html.single all
