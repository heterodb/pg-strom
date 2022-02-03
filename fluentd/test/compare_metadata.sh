#!/bin/bash

declare -r PG2ARROW_RELATIVE_PATH="../../arrow-tools/pg2arrow"
declare -r PG2ARROW_OPTION="--dump"

set -Ceu

function usage(){
    cat << EOL
Usage: ./compare_metadata.sh <arrow_file_path> <expected_file_path>

OPTIONS:
    -s      Save expected file.
EOL
    exit 1
}

[[ $# -ge 1 ]] && [[ $# -le 3 ]] || {
    echo "Wrong arguments." >&2
    usage
}

if [[ $(echo "$@" | grep -c -- "-h$") -gt 0  ]];then
    usage
fi

declare -r arrow_file_path=$1
declare -r expected_file_path=$2
declare -r script_path=$(cd $(dirname $0) ; pwd)
declare -r pg2arrow_path="${script_path}/${PG2ARROW_RELATIVE_PATH}"

# Checking args
[[ ! -r ${arrow_file_path} ]] && {
    echo "${arrow_file_path} is not readable." >&2
    exit 1
}

# Check whether
if [[ $(echo "$@" | grep -c -- "-s$") -gt 0 ]];then
    # Saving expected file.
    ${pg2arrow_path} ${PG2ARROW_OPTION} ${arrow_file_path} >| ${expected_file_path}
    exit $?
fi

[[ ! -r ${expected_file_path} ]] && {
    echo "${expected_file_path} is not readable." >&2
    exit 1
}

# Wrong option
[[ $# -eq 3 ]] && {
    echo "Wrong arguments." >&2
    usage
}

# Compare
declare -r regression_diff_file_path="${script_path}/regression.diffs"

echo "Files ${arrow_file_path} and ${expected_file_path} differ" >> ${regression_diff_file_path}
echo "------------------------" >> ${regression_diff_file_path}
diff <(${pg2arrow_path} ${PG2ARROW_OPTION} ${arrow_file_path}) <(cat ${expected_file_path}) | tee -a ${regression_diff_file_path}
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Not match"
    exit 1
else
    echo "Matched"
    exit 0
fi