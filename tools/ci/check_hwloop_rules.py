#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
#
# SPDX-License-Identifier: Apache-2.0
"""Check every PIE hardware loop in an ELF against the ESP32-P4 rules.

On ESP32-P4 before rev v3.0, an interrupt at the boundary of a hardware loop
that breaks these rules corrupts the loop counter (esp-idf#19025):

- the esp.lp.setup is 4-byte aligned;
- the last body instruction (the loop end) is 4-byte aligned and 4 bytes long;
- the body is at least 24 bytes;
- the last body instruction is not a PIE (esp.*) or FPU instruction.

Run it on a test_app ELF built for a P4 revision below v3.0:

    tools/ci/check_hwloop_rules.py test_app/build/test_app.elf

The objdump comes from --objdump, then $OBJDUMP, then riscv32-esp-elf-objdump.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

MIN_BODY_BYTES = 24

# "4001a00c:	00c6c02b          	esp.lp.setup	0,a3,24 # 4001a024 <f+0x9c>"
INSN_RE = re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)(?:\s+(.*))?$')
FUNC_RE = re.compile(r'^[0-9a-f]+ <(.+)>:$')
SETUP_TARGET_RE = re.compile(r'#\s*([0-9a-f]+)')
# F and D extension mnemonics start with f; fence is an integer instruction.
FPU_RE = re.compile(r'^f(?!ence)')


def parse(lines):
    """Return {address: (length, mnemonic, operands, function)}."""
    insns = {}
    func = '?'
    for line in lines:
        m = FUNC_RE.match(line)
        if m:
            func = m.group(1)
            continue
        m = INSN_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        length = len(m.group(2)) // 2
        insns[addr] = (length, m.group(3), m.group(4) or '', func)
    return insns


def check(insns):
    """Return (loop count, list of violation strings)."""
    loops = 0
    errors = []
    for addr in sorted(insns):
        length, mnem, ops, func = insns[addr]
        if mnem != 'esp.lp.setup':
            continue
        loops += 1
        where = f'{func} @ 0x{addr:08x}'
        m = SETUP_TARGET_RE.search(ops)
        if not m:
            errors.append(f'{where}: cannot read the loop end from "{ops}"')
            continue
        end = int(m.group(1), 16)
        if addr % 4:
            errors.append(f'{where}: esp.lp.setup is not 4-byte aligned')
        if end not in insns:
            errors.append(f'{where}: loop end 0x{end:08x} is not an instruction')
            continue
        end_len, end_mnem, _, _ = insns[end]
        body = end + end_len - (addr + length)
        if end % 4:
            errors.append(f'{where}: loop end 0x{end:08x} is not 4-byte aligned')
        if end_len != 4:
            errors.append(f'{where}: loop end {end_mnem} is {end_len} bytes, not 4')
        if body < MIN_BODY_BYTES:
            errors.append(f'{where}: body is {body} bytes, needs at least {MIN_BODY_BYTES}')
        if end_mnem.startswith('esp.') or FPU_RE.match(end_mnem):
            errors.append(f'{where}: loop ends on {end_mnem}, a PIE or FPU instruction')
    return loops, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('elf', help='ELF to check')
    ap.add_argument('--objdump', default=os.environ.get('OBJDUMP', 'riscv32-esp-elf-objdump'))
    args = ap.parse_args()

    if shutil.which(args.objdump) is None:
        print(f'check_hwloop_rules: {args.objdump} not found', file=sys.stderr)
        return 2
    if not os.path.isfile(args.elf):
        print(f'check_hwloop_rules: {args.elf} not found', file=sys.stderr)
        return 2

    dis = subprocess.run([args.objdump, '-d', args.elf], check=True,
                         capture_output=True, text=True).stdout
    loops, errors = check(parse(dis.splitlines()))

    if loops == 0:
        print(f'check_hwloop_rules: no esp.lp.setup in {args.elf}; '
              'is it an optimized ESP32-P4 build?', file=sys.stderr)
        return 1
    if errors:
        print(f'check_hwloop_rules: {len(errors)} rule violations in {loops} hardware loops:')
        for e in errors:
            print(f'  {e}')
        return 1
    print(f'check_hwloop_rules: all {loops} hardware loops follow the ESP32-P4 rules')
    return 0


if __name__ == '__main__':
    sys.exit(main())
