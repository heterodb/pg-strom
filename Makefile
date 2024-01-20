# ----------------------------------------
#
# Makefile for packaging
#
# ----------------------------------------
include Makefile.common

__PGSTROM_TGZ := pg_strom-$(PGSTROM_VERSION)
PGSTROM_TGZ   := $(__PGSTROM_TGZ).tar.gz

SWDC ?= $(shell test -d ../swdc/.git && realpath ../swdc || echo /dev/null)
__SWDC_URL = git@github.com:heterodb/swdc.git

__ARCH = $(shell rpmbuild -E %{_arch})
__DIST = $(shell rpmbuild -E %{dist} | grep -o -E '\.el[0-9]+' | sed 's/\.el/rhel/g')
__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})
__RPMDIR = $(shell rpmbuild -E %{_rpmdir})/$(__ARCH)
__SRPMDIR = $(shell rpmbuild -E %{_srcrpmdir})
__SPECFILE = $(__SPECDIR)/pg_strom-PG$(PG_MAJORVERSION).spec

rpm: rpm-pg_strom

all: rpm-pg_strom rpm-mysql2arrow rpm-pcap2arrow

__precheck_swdc:
	rpm -qf `which $(PG_CONFIG)` || exit 1
	(cd $(SWDC);	\
	 test "`git remote get-url origin`" = "$(__SWDC_URL)" || exit 1; \
	 git pull || exit 1)

swdc: __precheck_swdc rpm-pg_strom
	(PGSTROM_TGZ_FULLPATH="`realpath $(PGSTROM_TGZ)`";	\
	 cd $(SWDC);						\
	 RPM="`rpmspec -q --rpms $(__SPECFILE) | grep -v debug`.rpm"; \
	 DEST=docs/yum/$(__DIST)-$(__ARCH);		 	\
	   rpmsign --addsign $(__RPMDIR)/$${RPM} && 		\
	   mkdir -p $${DEST} &&					\
	   install -m 644 $(__RPMDIR)/$${RPM} $${DEST} &&	\
	   git add $${DEST}/$${RPM};				\
	 RPM="`rpmspec -q --rpms $(__SPECFILE) | grep debuginfo`.rpm"; \
	 DEST=docs/yum/$(__DIST)-debuginfo;			\
	   rpmsign --addsign $(__RPMDIR)/$${RPM} &&		\
	   mkdir -p $${DEST} &&					\
	   install -m 644 $(__RPMDIR)/$${RPM} $${DEST} &&	\
	   git add $${DEST}/$${RPM};				\
	 RPM=`rpmspec -q --srpm $(__SPECFILE) | sed 's/$(__ARCH)/src.rpm/g'`; \
	 DEST=docs/yum/$(__DIST)-source;				\
	   rpmsign --addsign $(__SRPMDIR)/$${RPM} &&		\
	   mkdir -p $${DEST} &&					\
	   install -m 644 $(__SRPMDIR)/$${RPM} $${DEST} &&	\
	   git add $${DEST}/$${RPM};				\
	 install -m 644 $${PGSTROM_TGZ_FULLPATH} ./docs/tgz &&	\
	   git add ./docs/tgz/$(PGSTROM_TGZ);			\
	./update-index.sh)

tarball:
	git archive --format=tar.gz \
	            --prefix=$(__PGSTROM_TGZ)/ \
	            -o $(PGSTROM_TGZ) $(GITHASH) \
	            LICENSE Makefile Makefile.common \
	            src arrow-tools test/ssbm

rpm-pg_strom: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/systemd-pg_strom.conf > \
		$(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	(git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"	\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g"		\
		    -e "s/@@PGSQL_VERSION@@/$(PG_MAJORVERSION)/g";\
	 git show --format=raw $(GITHASH):CHANGELOG) > $(__SPECFILE)
	rpmbuild -ba $(__SPECFILE) --undefine=_debugsource_packages

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
	rpmbuild -ba $(__SPECDIR)/mysql2arrow.spec			\
		--undefine=_debugsource_packages

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
	rpmbuild -ba $(__SPECDIR)/pcap2arrow.spec			\
		--undefine=_debugsource_packages
