#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
#
# SPDX-License-Identifier: Apache-2.0
"""Run every staged esp-nn test image in esp-emulator and report the results.

Reads the merged flash images staged by tools/ci/stage_merged_bin.sh, runs each
one, and takes the verdict from the TEST_RESULT line the test app prints. Writes
a profile CSV per run plus a combined CSV, and keeps the full log of a run that
failed.
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

# Targets esp-emu 0.41 models; everything else is build-only in CI.
EMU_TARGETS = ('esp32c3', 'esp32c5', 'esp32c6', 'esp32h2', 'esp32p4', 'esp32s31')

# One rule per (target, IDF range) combination that must not be emulated.
# `min_idf` is the first working release, `max_idf` the last; give at least one.
# `reason` reaches the skip report, so name the cause, not the symptom.
SkipRule = namedtuple('SkipRule', 'target reason min_idf max_idf')
SkipRule.__new__.__defaults__ = (None, None)

# Add a rule here to stop emulating a target, and say why above it.
SKIP_RULES = (
    # IDF < v5.5 hardcodes -march=..._xesppie for ESP32-P4 and never checks the
    # chip revision, so the image carries the pre-rev-v3 PIE encodings (for
    # example esp.movx.w.cfg as 0x90d6805f, not 0x9086a01b). Rev v3.x silicon and
    # esp-emu both reject those. The build still proves the kernels assemble.
    SkipRule('esp32p4', 'pre-rev-v3 PIE encodings', min_idf=(5, 5)),
)

# Staged directory names come from the CI job name: v5.4, v6.0, master.
IDF_VERSION_RE = re.compile(r'^v(\d+)\.(\d+)')

# The app halts in the GDB stub on panic, so esp-emu would otherwise idle until
# its own timeout. Stop on any of these.
PANIC_MARKERS = (
    'Guru Meditation',
    'Entering gdb stub now.',
    'Illegal instruction',
    'Task watchdog got triggered',
    'CORRUPT HEAP',
    'assert failed',
    'abort() was called',
)

# Every suite the test_app runs, in RUN_TEST order: a green job must have seen
# all of them, so deleting a RUN_TEST line cannot quietly shrink the run. Case
# counts are not listed because they legitimately differ per target (the S3 has
# one extra bounded-dot check), and a suite that verified nothing already fails
# through empty_suites.
EXPECTED_SUITES = (
    'add_s8', 'mul_s8', 'mul_broadcast_ch_s8', 'depthwise_conv_s8', 'conv_s8',
    'relu6_s8', 'avg_pool_s8', 'max_pool_s8', 'fc_s8', 'fc_per_ch_s8',
    'fc_per_ch_s8_batch', 'fc_align_s8', 'fc_perf', 'softmax_s8',
    'hard_swish_s8', 'mean_nhwc_s8',
)

# Each of these is printed once per boot (the ROM banner and the reset-reason
# line), so seeing either one twice means the app reset: a watchdog, a
# brown-out, or a panic that reboots instead of halting in the GDB stub.
# Without this the run would just stall to the timeout as NO_VERDICT. They are
# counted separately because a single boot prints both.
BOOT_RES = (re.compile(r'^ESP-ROM:'), re.compile(r'^rst:0x[0-9a-fA-F]+[ ,(]'))

# Stamped into every report row so a profile can be tied to the emulator
# that produced it.
EMU_SOURCE = 'esp-emu ' + os.environ.get('ESP_EMU_VERSION', 'unknown')

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
TEST_RE = re.compile(r'^TEST: (?P<kernel>[^,]+), passed=(?P<passed>\d+), '
                     r'failed=(?P<failed>\d+)(?:, skipped=(?P<skipped>\d+))?')
PROFILE_RE = re.compile(r'^PROFILE: (?P<kernel>[^,]+), ansi=(?P<ansi>\d+), '
                        r'opt=(?P<opt>\d+), speedup=(?P<speedup>[0-9.]+)x')
SUMMARY_RE = re.compile(r'^TEST_SUMMARY: passed=(?P<passed>\d+), '
                        r'failed=(?P<failed>\d+)(?:, skipped=(?P<skipped>\d+))?')
RESULT_RE = re.compile(r'^TEST_RESULT: (?P<result>\w+)')

# Cycle columns are emulator instruction counts (deterministic, good for
# regression diffs, not device performance); `source` says so in every row.
CSV_FIELDS = ('idf_version', 'target', 'kernel', 'passed', 'failed', 'skipped',
              'emu_ansi_cycles', 'emu_opt_cycles', 'emu_ratio', 'source')

PRINT_LOCK = threading.Lock()


class Run:
    def __init__(self, idf_version, target, image, log_path):
        self.idf_version = idf_version
        self.target = target
        self.image = image
        self.log_path = log_path
        self.kernels = {}       # kernel -> row dict
        self.order = []         # kernel names, in emission order
        self.passed = 0
        self.empty_suites = []      # suites that verified nothing (passed=0)
        self.seen_suites = set()    # suites that reported a TEST: verdict
        self.failed = 0
        self.skipped = 0
        self.result = None
        self.reason = ''
        self.boots = [0] * len(BOOT_RES)
        self.cmd = '(not started)'

    def row(self, kernel):
        if kernel not in self.kernels:
            self.kernels[kernel] = {'idf_version': self.idf_version,
                                    'target': self.target, 'kernel': kernel,
                                    'passed': '', 'failed': '', 'skipped': '',
                                    'emu_ansi_cycles': '', 'emu_opt_cycles': '',
                                    'emu_ratio': '', 'source': EMU_SOURCE}
            self.order.append(kernel)
        return self.kernels[kernel]

    def rows(self):
        return [self.kernels[k] for k in self.order]

    @property
    def missing_suites(self):
        # A PROFILE: line alone does not count: the suite has to report a
        # verdict, so a deleted RUN_TEST line (or a suite that died mid-way)
        # is caught.
        return [s for s in EXPECTED_SUITES if s not in self.seen_suites]

    @property
    def ok(self):
        return (self.result == 'PASS' and self.failed == 0 and self.passed > 0
                and not self.empty_suites and not self.missing_suites)


def find_images(bins_dir):
    """Yield (idf_version, target, image path) for every staged image."""
    if not os.path.isdir(bins_dir):
        return
    for idf_version in sorted(os.listdir(bins_dir)):
        idf_dir = os.path.join(bins_dir, idf_version)
        if not os.path.isdir(idf_dir):
            continue
        for target in sorted(os.listdir(idf_dir)):
            image = os.path.join(idf_dir, target, 'merged-binary.bin')
            if os.path.isfile(image):
                yield idf_version, target, image


def idf_version_tuple(name):
    """(major, minor) for a vX.Y directory name, or None for master and friends."""
    match = IDF_VERSION_RE.match(name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def skip_reason(idf_version, target):
    """Why this image must not be emulated, or None to run it."""
    if target not in EMU_TARGETS:
        return 'not emulated by esp-emu'
    version = idf_version_tuple(idf_version)
    # An unparsed name is a development branch, which is always new enough.
    if version is None:
        return None
    for rule in SKIP_RULES:
        if rule.target != target:
            continue
        if rule.min_idf and version < rule.min_idf:
            return 'needs IDF >= v%d.%d: %s' % (rule.min_idf + (rule.reason,))
        if rule.max_idf and version > rule.max_idf:
            return 'needs IDF <= v%d.%d: %s' % (rule.max_idf + (rule.reason,))
    return None


def parse_line(run, line):
    """Feed one clean line to the run. Returns True when the run is over."""

    for i, boot_re in enumerate(BOOT_RES):
        if boot_re.match(line):
            run.boots[i] += 1
            if run.boots[i] > 1:
                run.result = 'REBOOT'
                run.reason = 'the app reset and started over (reboot loop)'
                return True
    match = TEST_RE.match(line)
    if match:
        row = run.row(match.group('kernel'))
        run.seen_suites.add(match.group('kernel'))
        row['passed'] = match.group('passed')
        row['failed'] = match.group('failed')
        row['skipped'] = match.group('skipped') or ''
        if not int(match.group('passed')) and not int(match.group('failed')):
            # A suite that verified nothing lowers coverage without failing
            # anything, whether its cases bailed before a verdict or were all
            # skipped (out of memory). Make that visible and fatal.
            run.empty_suites.append(match.group('kernel'))
        return False

    match = PROFILE_RE.match(line)
    if match:
        row = run.row(match.group('kernel'))
        row['emu_ansi_cycles'] = match.group('ansi')
        row['emu_opt_cycles'] = match.group('opt')
        row['emu_ratio'] = match.group('speedup')
        return False

    match = SUMMARY_RE.match(line)
    if match:
        run.passed = int(match.group('passed'))
        run.failed = int(match.group('failed'))
        run.skipped = int(match.group('skipped') or 0)
        return False

    match = RESULT_RE.match(line)
    if match:
        run.result = match.group('result')
        return True

    for marker in PANIC_MARKERS:
        if marker in line:
            run.result = 'PANIC'
            run.reason = marker
            return True
    return False


def run_one(emu, run, timeout):
    cmd = [emu, '--chip', run.target, '--firmware', run.image,
           '--log-color', 'never', '--net', 'user',
           '--timeout', '%ds' % timeout]
    run.cmd = ' '.join(cmd)

    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1,
                            universal_newlines=True, errors='replace',
                            start_new_session=True)
    done = False
    with open(run.log_path, 'w') as log:
        for raw in proc.stdout:
            log.write(raw)
            if parse_line(run, ANSI_RE.sub('', raw).strip()):
                done = True
                break
            if time.monotonic() > deadline:
                break

    # esp-emu has no orderly shutdown; the verdict is already in hand.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()

    if run.result == 'REBOOT':
        pass  # reason already set; the run is over
    elif not done:
        run.result = run.result or 'NO_VERDICT'
        run.reason = run.reason or 'no TEST_RESULT line before the %ds cap' % timeout
    elif run.result == 'FAIL':
        run.reason = '%d case(s) failed' % run.failed
    elif run.result == 'PASS' and run.passed == 0:
        run.reason = 'reported PASS with no cases run'
    elif run.result == 'PASS' and run.empty_suites:
        run.reason = 'suite(s) verified no cases: %s' % ', '.join(run.empty_suites)
    elif run.result == 'PASS' and run.missing_suites:
        run.reason = 'suite(s) never ran: %s' % ', '.join(run.missing_suites)


def write_reports(runs, out_dir):
    profiles_dir = os.path.join(out_dir, 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'profiles.csv'), 'w', newline='') as handle:
        combined = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        combined.writeheader()
        for run in runs:
            combined.writerows(run.rows())
            name = '%s-%s.csv' % (run.idf_version, run.target)
            with open(os.path.join(profiles_dir, name), 'w', newline='') as per_run:
                writer = csv.DictWriter(per_run, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(run.rows())

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('idf_version', 'target', 'passed', 'failed', 'skipped',
                         'result', 'reason'))
        for run in runs:
            writer.writerow((run.idf_version, run.target, run.passed, run.failed,
                             run.skipped, run.result or 'NO_VERDICT', run.reason))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins-dir', default='test_app/ci_bins',
                        help='directory holding <idf>/<target>/merged-binary.bin')
    parser.add_argument('--out-dir', default='test_results',
                        help='directory for the CSV reports and the logs')
    parser.add_argument('--timeout', type=int, default=600,
                        help='seconds one target may run before it is a failure')
    parser.add_argument('--jobs', '-j', type=int, default=0,
                        help='concurrent emulators (default: min(4, CPU count))')
    parser.add_argument('--is-emulated', metavar='TARGET',
                        help='exit 0 if TARGET is emulated in CI, 1 if not; '
                             'used by stage_merged_bin.sh so build jobs only '
                             'upload images the test stage will run')
    parser.add_argument('--emu', default=os.environ.get('ESP_EMU', 'esp-emu'),
                        help='esp-emu executable')
    args = parser.parse_args()

    if args.is_emulated:
        sys.exit(0 if args.is_emulated in EMU_TARGETS else 1)

    emu = shutil.which(args.emu) or args.emu
    logs_dir = os.path.join(args.out_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    images = list(find_images(args.bins_dir))
    if not images:
        sys.exit('no images found under %s' % args.bins_dir)

    # Keep discovery order: the reports must not depend on completion order.
    runs, skipped = [], []
    for idf_version, target, image in images:
        reason = skip_reason(idf_version, target)
        if reason:
            skipped.append(('%s/%s' % (idf_version, target), reason))
            continue
        log_path = os.path.join(logs_dir, '%s-%s.log' % (idf_version, target))
        runs.append(Run(idf_version, target, image, log_path))
    if not runs:
        sys.exit('nothing to emulate: all %d image(s) skipped' % len(skipped))

    # Each emulator is CPU-bound, so do not oversubscribe the runner.
    jobs = args.jobs or min(4, os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(runs))) if runs else 1
    print('running %d target(s), %d at a time' % (len(runs), jobs), flush=True)

    done = [0]

    def worker(run):
        try:
            run_one(emu, run, args.timeout)
        except Exception as exc:                # keep one bad run from hiding the rest
            run.result = run.result or 'ERROR'
            run.reason = '%s: %s' % (type(exc).__name__, exc)
        if run.ok:
            body = ['    PASS  %d cases, %d skipped' % (run.passed, run.skipped)]
        else:
            body = ['    $ %s' % run.cmd,
                    '    FAIL  %s (%s) passed=%d failed=%d skipped=%d, log kept at %s'
                    % (run.result, run.reason, run.passed, run.failed, run.skipped,
                       run.log_path)]
        # count and print together, so the index matches the printed order
        with PRINT_LOCK:
            done[0] += 1
            print('\n'.join(['==> [%d/%d] %s %s'
                             % (done[0], len(runs), run.idf_version, run.target)]
                            + body), flush=True)

    if runs:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(worker, runs))

    write_reports(runs, args.out_dir)

    # Keep the log of a failed run only; the CSV is the artifact on success.
    for run in runs:
        if run.ok and os.path.exists(run.log_path):
            os.remove(run.log_path)

    if skipped:
        by_reason = {}
        for name, reason in skipped:
            by_reason.setdefault(reason, []).append(name)
        print('\nskipped %d image(s):' % len(skipped))
        for reason in sorted(by_reason):
            print('  %s -> %s' % (', '.join(by_reason[reason]), reason))
    failures = [run for run in runs if not run.ok]
    print('\n%d/%d target(s) passed, %d case(s) skipped'
          % (len(runs) - len(failures), len(runs), sum(r.skipped for r in runs)))
    if failures:
        sys.exit('failed: %s' % ', '.join('%s/%s' % (r.idf_version, r.target)
                                          for r in failures))


if __name__ == '__main__':
    main()
