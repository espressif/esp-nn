#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
#
# SPDX-License-Identifier: Apache-2.0
#
# Merge one built target into a single flash image and stage it for the test
# stage. Run from the test_app directory, after the target is built.
#
# Usage: stage_merged_bin.sh <target>

set -euo pipefail

TARGET="$1"
PROJECT_DIR="${CI_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
IDF_SLUG="${IDF_SLUG:-${CI_JOB_NAME:-local}}"
IDF_SLUG="${IDF_SLUG#build_idf_}"

# Only stage what the test stage can actually run; run_emulator_tests.py owns
# the list of emulated targets so it is not duplicated here.
if ! python3 "${PROJECT_DIR}/tools/ci/run_emulator_tests.py" --is-emulated "${TARGET}"; then
    echo "${TARGET} is not emulated in CI; not staging an image for it"
    exit 0
fi

OUT_DIR="${PROJECT_DIR}/test_app/ci_bins/${IDF_SLUG}/${TARGET}"
OUT_BIN="${OUT_DIR}/merged-binary.bin"
mkdir -p "${OUT_DIR}"

# idf.py merge-bin exists from IDF v5.3. Older releases need esptool directly.
idf_has_merge_bin() {
    local version
    version="$(idf.py --version 2>/dev/null | grep -o 'v[0-9]\+\.[0-9]\+' | head -1)"
    [ -n "${version}" ] || return 1
    python3 -c 'import sys
major, minor = sys.argv[1].lstrip("v").split(".")[:2]
sys.exit(0 if (int(major), int(minor)) >= (5, 3) else 1)' "${version}"
}

if idf_has_merge_bin; then
    # -o must be absolute: idf.py runs esptool with cwd=build_dir.
    idf.py merge-bin -o "${OUT_BIN}"
else
    # @flash_args carries the target's own offsets and flag spellings.
    ( cd build && esptool.py --chip "${TARGET}" merge_bin -o "${OUT_BIN}" @flash_args )
fi

echo "staged ${TARGET}: $(du -h "${OUT_BIN}" | cut -f1) at ${OUT_BIN}"
