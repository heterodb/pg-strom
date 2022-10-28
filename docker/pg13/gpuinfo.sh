#!/bin/sh
# When running in a VM or container, the NVIDIA driver will likely not be available under '/sys/module/nvidia/version'
# This script is a hack that prepends a NVIDIA_DRIVER_VERSION value before the `gpuinfo` output so PG-Strom will start up.
# For more information, please see https://github.com/heterodb/pg-strom/issues/552#issuecomment-1288449738
echo "PLATFORM:NVIDIA_DRIVER_VERSION=000.00"
gpuinfo.orig -md