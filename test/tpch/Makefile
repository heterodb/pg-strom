#
# $Id: makefile.suite,v 1.25 2009/10/22 19:10:21 jms Exp $
#
# Revision History
# ===================
# $Log: makefile.suite,v $
# Revision 1.25  2009/10/22 19:10:21  jms
# update revision to 2.9.0, disable bug55 fix
#
# Revision 1.24  2009/10/22 19:06:10  jms
# update revision to 2.9.0, disable bug55 fix
#
# Revision 1.23  2009/06/28 14:01:08  jms
# bug fix for DOP
#
# Revision 1.22  2008/09/15 16:37:46  jms
# release 2.8.0 makefile.suite
#
# Revision 1.21  2008/03/21 18:26:54  jms
# recursive zip for reference data, chmod for update_release
#
# Revision 1.20  2008/03/21 17:38:39  jms
# changes for 2.6.3
#
# Revision 1.19  2007/03/08 20:36:03  jms
# update release number to 2.6.1
#
# Revision 1.18  2007/02/06 18:15:56  jms
# remove update release from general target
#
# Revision 1.17  2007/01/25 19:35:50  jms
# add sln file used by VS2005
#
# Revision 1.16  2007/01/05 20:05:41  jms
# update release number
#
# Revision 1.15  2006/09/07 17:25:57  jms
# correct dss.ddl
#
# Revision 1.14  2006/08/01 17:21:22  jms
# fix bad merge
#
# Revision 1.13  2006/08/01 16:55:44  jms
# move to 2.4.1
#
# Revision 1.12  2006/06/29 20:46:17  jms
# 2.4.0 changes from Meikel
#
# Revision 1.10  2006/05/25 22:30:44  jms
# qgen porting for 32b/64b
#
# Revision 1.9  2006/04/26 23:17:09  jms
# checking release.h prior to release build
#
# Revision 1.8  2006/04/26 23:03:00  jms
# release 2.3.4-1
#
# Revision 1.7  2006/04/12 18:13:58  jms
# release 2.3.3
#
# Revision 1.6  2006/03/09 18:59:19  jms
# move to version 2.3.2
#
# Revision 1.5  2006/01/28 23:54:32  jms
# add reference data to release
#
# Revision 1.4  2005/10/28 03:00:32  jms
# fix release target
#
# Revision 1.3  2005/10/28 02:54:14  jms
# increment build count with each release creation
#
# Revision 1.2  2005/01/03 20:08:58  jms
# change line terminations
#
# Revision 1.1.1.1  2004/11/24 23:31:47  jms
# re-establish external server
#
# Revision 1.5  2004/03/26 20:39:23  jms
# add tpch tag to release files
#
# Revision 1.4  2004/03/16 14:45:57  jms
# correct release target in makefile
#
# Revision 1.3  2004/03/02 20:49:01  jms
# simplify distributions, add Windows IDE files
# releases should use make release from now on
#
# Revision 1.2  2004/02/18 14:05:53  jms
# porting changes for LINUX and 64 bit RNG
#
# Revision 1.1.1.1  2003/04/03 18:54:21  jms
# recreation after CVS crash
#
# Revision 1.1.1.1  2003/04/03 18:54:21  jms
# initial checkin
#
#
#
################
## CHANGE NAME OF ANSI COMPILER HERE
################
CC      = gcc
# Current values for DATABASE are: INFORMIX, DB2, TDAT (Teradata)
#                                  SQLSERVER, SYBASE, ORACLE, VECTORWISE
# Current values for MACHINE are:  ATT, DOS, HP, IBM, ICL, MVS, 
#                                  SGI, SUN, U2200, VMS, LINUX, WIN32 
# Current values for WORKLOAD are:  TPCH
DATABASE = POSTGRESQL
MACHINE  = LINUX
WORKLOAD = TPCH
#
CFLAGS	= -g -DDBNAME=\"dss\" -D$(MACHINE) -D$(DATABASE) -D$(WORKLOAD) -DRNG_TEST -D_FILE_OFFSET_BITS=64 
LDFLAGS = -O
# The OBJ,EXE and LIB macros will need to be changed for compilation under
#  Windows NT
OBJ     = .o
EXE     =
LIBS    = -lm
#
# NO CHANGES SHOULD BE NECESSARY BELOW THIS LINE
###############
VERSION=2
RELEASE=13
PATCH=0
BUILD=`grep BUILD release.h | cut -f3 -d' '`
NEW_BUILD=`expr ${BUILD} + 1`
TREE_ROOT=/tmp/tree
#
DBGEN = dbgen-tpch$(EXE)
#
HDR = dss.h rnd.h config.h dsstypes.h shared.h bcd2.h rng64.h release.h dists.dss.h
#
SRC = build.c driver.c bm_utils.c rnd.c print.c load_stub.c bcd2.c \
	speed_seed.c text.c permute.c rng64.c
#
OBJS = $(SRC:.c=.o)

#
SETS = dists.dss 
DOC=README HISTORY PORTING.NOTES BUGS
DDL  = dss.ddl dss.ri
OTHER=makefile.suite $(SETS) $(DDL)
# case is *important* in TEST_RES
TEST_RES = O.res L.res c.res s.res P.res S.res n.res r.res
#
DBGENSRC=$(SRC) $(HDR) $(OTHER) $(DOC)
FQD=queries/1.sql queries/2.sql queries/3.sql queries/4.sql queries/5.sql queries/6.sql queries/7.sql \
	queries/8.sql queries/9.sql queries/10.sql queries/11.sql queries/12.sql queries/13.sql \
	queries/14.sql queries/15.sql queries/16.sql queries/17.sql queries/18.sql queries/19.sql queries/20.sql \
	queries/21.sql queries/22.sql
VARIANTS= variants/8a.sql variants/12a.sql variants/13a.sql variants/14a.sql variants/15a.sql 
ANS   = answers/q1.out answers/q2.out answers/q3.out answers/q4.out answers/q5.out answers/q6.out answers/q7.out answers/q8.out \
	answers/q9.out answers/q10.out answers/q11.out answers/q12.out answers/q13.out answers/q14.out answers/q15.out \
	answers/q16.out answers/q17.out answers/q18.out answers/q19.out answers/q20.out answers/q21.out answers/q22.out
QSRC  = $(FQD) $(VARIANTS) $(ANS)
TREE_DOC=tree.readme tree.changes appendix.readme appendix.version answers.readme queries.readme variants.readme
REFERENCE=reference/[tcR]*
REFERENCE_DATA=referenceData/[13]*
SCRIPTS= check55.sh column_split.sh dop.sh gen_tasks.sh last_row.sh load_balance.sh new55.sh check_dirs.sh
ALLSRC=$(DBGENSRC) $(REFERENCE) $(QSRC) $(SCRIPTS)
JUNK  = 
#
all: $(DBGEN)

$(DBGEN): $(OBJS) $(SETS) 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f $(DBGEN) $(OBJS) $(JUNK) dists.dss.h
lint:
	lint $(CFLAGS) -u -x -wO -Ma -p $(SRC)

rnd$(OBJ): rnd.h

dists.dss.h: dists.dss
	(echo "const char *static_dists_dss =";		\
	 cat $^ | while read line;					\
	 do											\
	   echo $$line | sed -e 's/^/  "/g' -e 's/$$/\\n"/g'; \
	 done; echo ";") > $@

$(OBJS): $(HDR)
