# ----------------------------------------
#
# Makefile for packaging
#
# ----------------------------------------
include Makefile.common

__PGSTROM_TGZ := pg_strom-$(PGSTROM_VERSION)
PGSTROM_TAR   := $(__PGSTROM_TGZ).tar
PGSTROM_TGZ   := $(__PGSTROM_TGZ).tar.gz

SWDC ?= $(shell test -d ../swdc/.git && realpath ../swdc || echo /dev/null)
__SWDC_URL = git@github.com:heterodb/swdc.git

__ARCH = $(shell rpmbuild -E %{_arch})
__DIST = $(shell rpmbuild -E %{dist} | grep -o -E '\.el[0-9]+' | sed 's/\.el/rhel/g')
__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})
__RPMDIR = $(shell rpmbuild -E %{_rpmdir})/$(__ARCH)
__SRPMDIR = $(shell rpmbuild -E %{_srcrpmdir})
__SPECFILE = $(__SPECDIR)/pg_strom.spec

rpm: rpm-pg_strom

all: rpm-pg_strom rpm-mysql2arrow rpm-pcap2arrow

__precheck_swdc:
	(cd $(SWDC);	\
	 test "`git remote get-url origin`" = "$(__SWDC_URL)" || exit 1; \
	 git pull || exit 1)

swdc: __precheck_swdc rpm-pg_strom
	(PGSTROM_TGZ_FULLPATH="`realpath $(PGSTROM_TGZ)`";	\
	 cd $(SWDC);						\
	 RPMS="`rpmspec -q --rpms $(__SPECFILE) | grep -v debuginfo`"; \
	 DEST=docs/yum/$(__DIST)-$(__ARCH);		 	\
	 mkdir -p $${DEST};					\
	 for f in $$RPMS; do					\
	   __file="$${f}.rpm";					\
	   rpmsign --addsign "$(__RPMDIR)/$${__file}" &&	\
	   install -m 644 "$(__RPMDIR)/$${__file}" $${DEST} &&	\
	   git add "$${DEST}/$${__file}";			\
	 done;							\
	 RPMS="`rpmspec -q --rpms $(__SPECFILE) | grep debuginfo`";\
	 DEST=docs/yum/$(__DIST)-debuginfo;			\
	 mkdir -p $${DEST};					\
	 for f in $$RPMS; do					\
	   __file="$${f}.rpm";					\
	   rpmsign --addsign "$(__RPMDIR)/$${__file}" &&	\
	   install -m 644 "$(__RPMDIR)/$${__file}" $${DEST} &&	\
	   git add "$${DEST}/$${__file}";			\
	 done;							\
	 RPMS="`rpmspec -q --srpm $(__SPECFILE)`";		\
	 DEST=docs/yum/$(__DIST)-source;			\
	 mkdir -p $${DEST};					\
	 for f in $$RPMS; do					\
	   __file="`echo $${f} | sed 's/$(__ARCH)/src.rpm/g'`" && \
	   rpmsign --addsign "$(__SRPMDIR)/$${__file}" &&	\
	   install -m 644 "$(__SRPMDIR)/$${__file}" $${DEST} &&	\
	   git add "$${DEST}/$${__file}";			\
	 done;							\
	 install -m 644 $${PGSTROM_TGZ_FULLPATH} ./docs/tgz &&	\
	   git add ./docs/tgz/$(PGSTROM_TGZ);			\
	./update-index.sh)

tarball:
	git archive --format=tar			\
	            --prefix=$(__PGSTROM_TGZ)/		\
	            -o $(PGSTROM_TAR) $(GITHASH)	\
	            LICENSE README.md Makefile		\
	            src arrow-tools test/ssbm
	TARFILE=`realpath $(PGSTROM_TAR)` && 		\
	TEMP=`mktemp -d` &&				\
	mkdir -p $${TEMP}/$(__PGSTROM_TGZ) &&		\
	git show --format=raw $(GITHASH):Makefile.common \
	| awk '/^GITHASH_IF_NOT_GIVEN/{ print "GITHASH_IF_NOT_GIVEN := $(GITHASH)"; next } { print }' \
	> $${TEMP}/$(__PGSTROM_TGZ)/Makefile.common &&	\
	pushd $${TEMP} &&				\
	echo $${TARFILE} &&				\
	tar rf $${TARFILE} $(__PGSTROM_TGZ)/Makefile.common && \
	popd &&						\
	gzip -f $(PGSTROM_TAR)

rpm-pg_strom: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/systemd-pg_strom.conf > \
		$(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" > $(__SPECFILE)
	rpmbuild -ba $(__SPECFILE)

rpm-mysql2arrow: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/mysql2arrow.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" >		\
		$(__SPECDIR)/mysql2arrow.spec
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		awk 'BEGIN {flag=0;} /^%changelog$$/{flag=1; next;} { if (flag>0) print; }' >> \
		$(__SPECDIR)/mysql2arrow.spec
	rpmbuild -ba $(__SPECDIR)/mysql2arrow.spec

rpm-pcap2arrow: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/pcap2arrow.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" >		\
		$(__SPECDIR)/pcap2arrow.spec
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		awk 'BEGIN {flag=0;} /^%changelog$$/{flag=1; next;} { if (flag>0) print; }' >> \
		$(__SPECDIR)/mysql2arrow.spec
	rpmbuild -ba $(__SPECDIR)/pcap2arrow.spec
