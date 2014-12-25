#! /bin/sh

#######################
# Test cases Maker    #
#######################

# Add test case names you want to test.
targets=(agg_init explain_agg group_agg nogrp_agg overflow_agg where_agg zero_agg)

echo "target files to make are...."
echo "***        ${targets[*]}         ****"

# Make DUMMY expected files. ( because pg_regress needs expected files )
for target in "${targets[@]}"
do
	touch expected/${target}.out
done

######################################################################################
#  Execute PG-strom Regression Test *** without PG-strom *** on temp-check install   #
######################################################################################
../../src/test/regress/pg_regress --top-builddir=../.. --extra-install=contrib/pg_strom --inputdir=input --temp-install=tmp_check --temp-config=input/disable.conf ${targets[*]}

# Copy result files to expected directory.
for target in "${targets[@]}"
do
	cp -f results/${target}.out expected
done

exit 0
