#!/bin/bash

set -u

declare -r ARROW2CSV_RELATIVE_PATH="../../arrow-tools/arrow2csv"

declare my_file_path=$1

find ${my_file_path} -type f | xargs -I{} -P 1 arrow2csv {} | wc -l