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

PYTHON_CMD := $(shell which python)
HTMLSOURCES := manual.src.html install.src.html tutrial.src.html \
		release.src.html
HTMLFILES := $(addprefix html/,$(HTMLSOURCES:.src.html=.html))
HTML_TEMPLATE := template.src.html
MENUGEN_PY := menugen.py

ifndef MAKEFILE_PARAMDEF_ONLY

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
endif	# MAKEFILE_PARAMDEF_ONLY
