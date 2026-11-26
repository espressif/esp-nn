/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include "esp_nn_filter_sum_esp32s3.h"

/* QACC helpers for ESP32-S3 int8 kernels.
 *
 * QACC holds 16 lanes of 20 bits. A dump stores 128 + 32 bits per half, so
 * one dump takes 52 bytes: QACC_L at +0 and QACC_H at +32. The macros below
 * are asm text. Register names are passed in as strings, so a kernel can use
 * fixed registers or %[operands]. They use SAR. */

#define ESP_NN_QACC_S_(x) #x
#define ESP_NN_QACC_S(x) ESP_NN_QACC_S_(x)

/* Dumps QACC at B + OFF, through the pointer register P. */
#define ESP_NN_QACC_DUMP(B, OFF, P)                                         \
    "addi   " P ", " B ", " ESP_NN_QACC_S(OFF) "\n"                         \
    "ee.st.qacc_l.l.128.ip  " P ", 16\n"                                    \
    "ee.st.qacc_l.h.32.ip   " P ", 16\n"                                    \
    "ee.st.qacc_h.l.128.ip  " P ", 16\n"                                    \
    "ee.st.qacc_h.h.32.ip   " P ", -16\n"

/* Eight 20-bit lanes from the five words at B + OFF into QA (lanes 0-3) and
 * QB (lanes 4-7). A lane inside one word takes a shift pair. A lane across
 * two words takes a funnel shift and a sign extend. T0-T4 are temps. */
#define ESP_NN_QACC_UNPACK_HALF(B, OFF, QA, QB, T0, T1, T2, T3, T4)         \
    "l32i   " T1 ", " B ", " ESP_NN_QACC_S(OFF) "\n"                        \
    "l32i   " T2 ", " B ", " ESP_NN_QACC_S(OFF) " + 4\n"                    \
    "sext   " T0 ", " T1 ", 19\n"            /* bits 0-19 */                \
    "ssai   20\n"                                                           \
    "src    " T3 ", " T2 ", " T1 "\n"        /* bits 20-39 */               \
    "sext   " T3 ", " T3 ", 19\n"                                           \
    "slli   " T4 ", " T2 ", 4\n"             /* bits 40-59 */               \
    "srai   " T4 ", " T4 ", 12\n"                                           \
    "ee.movi.32.q   " QA ", " T0 ", 0\n"                                    \
    "ee.movi.32.q   " QA ", " T3 ", 1\n"                                    \
    "ee.movi.32.q   " QA ", " T4 ", 2\n"                                    \
    "l32i   " T1 ", " B ", " ESP_NN_QACC_S(OFF) " + 8\n"                    \
    "ssai   28\n"                                                           \
    "src    " T0 ", " T1 ", " T2 "\n"        /* bits 60-79 */               \
    "sext   " T0 ", " T0 ", 19\n"                                           \
    "l32i   " T2 ", " B ", " ESP_NN_QACC_S(OFF) " + 12\n"                   \
    "ee.movi.32.q   " QA ", " T0 ", 3\n"                                    \
    "ssai   16\n"                                                           \
    "src    " T0 ", " T2 ", " T1 "\n"        /* bits 80-99 */               \
    "sext   " T0 ", " T0 ", 19\n"                                           \
    "slli   " T4 ", " T2 ", 8\n"             /* bits 100-119 */             \
    "srai   " T4 ", " T4 ", 12\n"                                           \
    "l32i   " T1 ", " B ", " ESP_NN_QACC_S(OFF) " + 16\n"                   \
    "ee.movi.32.q   " QB ", " T0 ", 0\n"                                    \
    "ee.movi.32.q   " QB ", " T4 ", 1\n"                                    \
    "ssai   24\n"                                                           \
    "src    " T0 ", " T1 ", " T2 "\n"        /* bits 120-139 */             \
    "sext   " T0 ", " T0 ", 19\n"                                           \
    "srai   " T4 ", " T1 ", 12\n"            /* bits 140-159 */             \
    "ee.movi.32.q   " QB ", " T0 ", 2\n"                                    \
    "ee.movi.32.q   " QB ", " T4 ", 3\n"

/* Lanes 0-3, 4-7, 8-11 and 12-15 of a dump at B into q0, q3, q1 and q2. */
#define ESP_NN_QACC_UNPACK(B, T0, T1, T2, T3, T4)                           \
    ESP_NN_QACC_UNPACK_HALF(B, 0, "q0", "q3", T0, T1, T2, T3, T4)           \
    ESP_NN_QACC_UNPACK_HALF(B, 32, "q1", "q2", T0, T1, T2, T3, T4)

/* Per-lane sum of n >= 0 vectors of 16 int8, stride bytes apart, from a
 * 16-byte aligned p. out is 64 bytes, 16-byte aligned. It also takes the
 * QACC dump. n is at most 4096, so the 20-bit lanes cannot overflow. */
static inline void esp_nn_qacc_lane_sum_s8_esp32s3(const int8_t *p, int stride, int n,
                                                   int32_t *out)
{
    int32_t t0, t1, t2, t3, t4, dp;
    asm volatile(
        "ee.vldbc.8     q1, %[one]\n"
        "ee.zero.qacc\n"
        "loopgtz        %[n], 1f\n"
        "ee.vld.128.xp  q0, %[p], %[stride]\n"
        "ee.vmulas.s8.qacc  q0, q1\n"
        "1:\n"
        ESP_NN_QACC_DUMP("%[out]", 0, "%[dp]")
        ESP_NN_QACC_UNPACK("%[out]", "%[t0]", "%[t1]", "%[t2]", "%[t3]", "%[t4]")
        "st.qr  q0, %[out], 0\n"
        "st.qr  q3, %[out], 16\n"
        "st.qr  q1, %[out], 32\n"
        "st.qr  q2, %[out], 48\n"
        : [p] "+r" (p), [n] "+r" (n), [dp] "=&r" (dp),
          [t0] "=&r" (t0), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3),
          [t4] "=&r" (t4)
        : [stride] "r" (stride), [out] "r" (out), [one] "r" (&esp_nn_fs_one)
        : "memory");
}
