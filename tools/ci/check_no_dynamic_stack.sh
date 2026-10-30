#!/bin/bash
# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
#
# Fail if any esp-nn kernel adjusts the stack pointer at run time.
#
# The kernels run on the caller's task stack, so an allocation sized from a
# tensor shape can overflow it and corrupt the calling task. -Werror=vla
# catches the C form; this catches whatever reaches the binary, including
# alloca and hand-written frames. On Xtensa a fixed frame uses `entry a1, N`;
# only a run-time-sized one needs `movsp`.
#
# Run after a build (default build dir: test_app/build):
#   tools/ci/check_no_dynamic_stack.sh [build-dir]
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
build=${1:-$root/test_app/build}
objdump=${OBJDUMP:-xtensa-esp32s3-elf-objdump}

if ! command -v "$objdump" >/dev/null 2>&1; then
    echo "check_no_dynamic_stack: $objdump not found" >&2
    exit 2
fi

lib=$build/esp-idf/esp-nn/libesp-nn.a
if [ ! -f "$lib" ]; then
    echo "check_no_dynamic_stack: $lib not found" >&2
    exit 2
fi

hits=$("$objdump" -d "$lib" | awk '
    /^[0-9a-f]+ <.*>:$/ { fn = $2 }
    /movsp/            { print fn }' | sort -u)

if [ -n "$hits" ]; then
    echo "check_no_dynamic_stack: run-time stack adjustment in $lib:"
    echo "$hits"
    exit 1
fi

echo "check_no_dynamic_stack: no run-time stack adjustment in esp-nn"
