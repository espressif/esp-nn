/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* PIE context probe for pie_context_test.c (ESP32-P4 only).
 * Assembly loads/stores the full PIE state and keeps the function without RVC.
 *
 * uint32_t nn_test_pie_hold(const nn_test_pie_ctx_t *in, nn_test_pie_ctx_t *out,
 *                           uint32_t spins, uint32_t full)
 *
 * Loads the PIE registers from *in, runs a hardware loop of `spins`
 * iterations (8 addi each) during which the task may be interrupted or
 * preempted, then stores the PIE registers to *out. Returns the addi count,
 * which must be spins * 8.
 *
 * When `full` is 0, q3, SAR and FFT_BIT_WIDTH are neither loaded nor
 * compared (the caller skips them): ESP-IDF v5.3 - v5.5 do not save / restore
 * them correctly on a PIE context switch (IDF-16495).
 *
 * Layout of nn_test_pie_ctx_t (16-byte aligned, 240 bytes):
 *   0 q0..q7 | 128 QACC l.l l.h h.l h.h | 192 UA_STATE | 208 XACC low 32 bits
 *   | 212 XACC high 8 bits | 216 SAR | 220 SAR_BYTES | 224 FFT_BIT_WIDTH | 228 unused
 *
 * The hardware loop follows the ESP32-P4 placement rules (see
 * ESP_NN_HWLP_BEGIN in src/common/common_functions.h): the whole function is
 * built without RVC, esp.lp.setup is 4-byte aligned, the body is 32 bytes and
 * its last instruction is a plain integer one. PIE GPR operands are limited to
 * x8-x15 / x24-x31, so only a0-a5 and t3 are used.
 */

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32P4

    .section .text.nn_test_pie_hold, "ax", @progbits
    .option push
    .option norvc

    .balign 4
    .global nn_test_pie_hold
    .type   nn_test_pie_hold, @function
nn_test_pie_hold:
    mv    a4, a0
    esp.vld.128.ip q0, a4, 16
    esp.vld.128.ip q1, a4, 16
    esp.vld.128.ip q2, a4, 16
    beqz  a3, 3f
    esp.vld.128.ip q3, a4, 16
    j     4f
3:  addi  a4, a4, 16
4:  esp.vld.128.ip q4, a4, 16
    esp.vld.128.ip q5, a4, 16
    esp.vld.128.ip q6, a4, 16
    esp.vld.128.ip q7, a4, 16
    esp.ld.qacc.l.l.128.ip a4, 16
    esp.ld.qacc.l.h.128.ip a4, 16
    esp.ld.qacc.h.l.128.ip a4, 16
    esp.ld.qacc.h.h.128.ip a4, 16
    esp.ld.ua.state.ip a4, 16
    lw    a5, 0(a4)
    esp.movx.w.xacc.l a5
    lw    a5, 4(a4)
    esp.movx.w.xacc.h a5
    lw    a5, 12(a4)
    esp.movx.w.sar.bytes a5
    beqz  a3, 5f
    lw    a5, 8(a4)
    esp.movx.w.sar a5
    lw    a5, 16(a4)
    esp.movx.w.fft.bit.width a5
5:  li    t3, 0
    beqz  a2, 2f
    .balign 4
    esp.lp.setup 0, a2, 1f
    addi  t3, t3, 1
    addi  t3, t3, 1
    addi  t3, t3, 1
    addi  t3, t3, 1
    addi  t3, t3, 1
    addi  t3, t3, 1
    addi  t3, t3, 1
1:  addi  t3, t3, 1
2:  mv    a4, a1
    esp.vst.128.ip q0, a4, 16
    esp.vst.128.ip q1, a4, 16
    esp.vst.128.ip q2, a4, 16
    esp.vst.128.ip q3, a4, 16
    esp.vst.128.ip q4, a4, 16
    esp.vst.128.ip q5, a4, 16
    esp.vst.128.ip q6, a4, 16
    esp.vst.128.ip q7, a4, 16
    esp.st.qacc.l.l.128.ip a4, 16
    esp.st.qacc.l.h.128.ip a4, 16
    esp.st.qacc.h.l.128.ip a4, 16
    esp.st.qacc.h.h.128.ip a4, 16
    esp.st.ua.state.ip a4, 16
    esp.movx.r.xacc.l a5
    sw    a5, 0(a4)
    esp.movx.r.xacc.h a5
    sw    a5, 4(a4)
    esp.movx.r.sar a5
    sw    a5, 8(a4)
    esp.movx.r.sar.bytes a5
    sw    a5, 12(a4)
    esp.movx.r.fft.bit.width a5
    sw    a5, 16(a4)
    mv    a0, t3
    ret
    .size   nn_test_pie_hold, .-nn_test_pie_hold

    .option pop

#endif /* CONFIG_IDF_TARGET_ESP32P4 */
