#
# Markdown (document) files
#
MKDOCS ?= mkdocs
__DOC_FILES :=	index.md	\
		install.md	\
		operations.md	\
		partition.md	\
		brin.md		\
		postgis.md	\
		hll_count.md	\
		troubles.md	\
		ssd2gpu.md	\
		arrow_fdw.md	\
		gpucache.md	\
		fluentd.md	\
		ref_types.md	\
		ref_devfuncs.md	\
		ref_sqlfuncs.md	\
		ref_params.md	\
		release_v2.0.md	\
		release_v2.2.md	\
		release_v2.3.md	\
		release_v3.0.md	\
		release_v5.0.md

DOC_FILES := $(addprefix ./, $(__DOC_FILES))

CC	?= gcc
CFLAGS	?= -g -O2 -Wall


all: html

markdown_i18n: markdown_i18n.c
	$(CC) $(CFLAGS) -o $@ $^

html: markdown_i18n
	for x in $(DOC_FILES);	\
	do			\
	  ./markdown_i18n -f $$x -o ./docs/$$x; \
	done
	./markdown_i18n -f mkdocs.yml -o mkdocs.en.yml
	env LANG=en_US.utf8	\
	  $(MKDOCS) build -c -f mkdocs.en.yml -d ../docs
	for x in $(DOC_FILES);	\
	do			\
	  ./markdown_i18n -l ja -f $$x -o ./docs/$$x; \
	done
	./markdown_i18n -l ja -f mkdocs.yml -o mkdocs.ja.yml
	env LANG=en_US.utf8	\
	  $(MKDOCS) build -c -f mkdocs.ja.yml -d ../docs/ja


clean:
	rm -f markdown_i18n
