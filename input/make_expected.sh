#! /bin/sh

#####################################################################
# PG-strom Regression Test cases                                    #
#####################################################################
targets=(agg_init explain_agg group_agg nogrp_agg overflow_agg where_agg zero_agg)

echo "target files to make are...."
echo "***        ${targets[*]}         ****"

echo "go"
for target in "${targets[@]}"
do
	touch expected/${target}.out
done

##############################################################################
#  Execute PG-strom Regression Test without PG-strom  on temp-check install  #
##############################################################################
../../src/test/regress/pg_regress --top-builddir=../.. --extra-install=contrib/pg_strom --inputdir=input --temp-install=tmp_check --temp-config=input/disable.conf ${targets[*]}


for target in "${targets[@]}"
do
	cp -f results/${target}.out expected
done

exit 0
