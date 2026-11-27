/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * For the ESP32-P4 rev < v3 ISA (xesppie), GCC 13.2 (IDF v5.3) encodes the
 * esp.vldbc.8.ip and esp.vldbc.16.ip increment in 4-byte units. The chip
 * steps by the element size (1 or 2 bytes). With that assembler an
 * increment of 2 does not assemble, and an increment of 4 runs as 2.
 * - Known affected: GCC 13.2 (IDF v5.3).
 * - Encode it correctly: GCC 14.2.0_20260121 and GCC 14.4.
 * - Not affected: the rev >= v3 ISA (xespv), and the xespv2p1 ISA that
 *   IDF v6 uses for rev < v3.
 * The compiler cannot tell toolchains apart, so the gate covers every
 * xesppie build: ESP_NN_VLDBC_IP_BROKEN is 1, and kernels then use other
 * loads. esp.vldbc.32.ip and an increment of 0 are encoded correctly.
 */
#ifndef ESP_NN_VLDBC_IP_BROKEN
#if defined(__riscv_xesppie)
#define ESP_NN_VLDBC_IP_BROKEN 1
#else
#define ESP_NN_VLDBC_IP_BROKEN 0
#endif
#endif
