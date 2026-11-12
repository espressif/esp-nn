/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <common_functions.h>

/*
 * Bit-exact esp_nn_multiply_by_quantized_mult() for mult >= 0, as a few
 * Xtensa instructions instead of generic 64-bit C.
 *
 * The reference adds a sign-dependent nudge and divides by 2^31 with
 * truncation. Both branches equal floor((v * mult + 2^30) / 2^31):
 * - ab >= 0: truncation is floor.
 * - ab < 0: trunc((ab + 1 - 2^30) / 2^31) = floor((ab + 2^30) / 2^31).
 * The saturation case needs mult == INT32_MIN, which a mult >= 0 rules out.
 */
static inline int32_t esp_nn_requant_s3(int32_t x, int32_t mult, int32_t shift)
{
    const int32_t left_shift = shift > 0 ? shift : 0;
    const int32_t right_shift = left_shift - shift;
    const int32_t v = (int32_t)((uint32_t)x << left_shift);
    int32_t hi;
    uint32_t lo;
    asm ("mulsh %0, %1, %2" : "=a" (hi) : "a" (v), "a" (mult));
    asm ("mull  %0, %1, %2" : "=a" (lo) : "a" (v), "a" (mult));
    const uint32_t lo_n = lo + 0x40000000u;
    hi += (lo_n < lo);
    const int32_t r = (int32_t)(((uint32_t)hi << 1) | (lo_n >> 31));
    return esp_nn_div_by_power_of_two(r, right_shift);
}

/* Accumulate a dot of `blocks` 16-byte vectors into ACCX, without
 * clearing it. Both pointers are 16-byte aligned. blocks >= 1. */
static inline void esp_nn_dot_accx_s3(const int8_t *a, const int8_t *b,
                                      int blocks)
{
    asm volatile (
        "ee.vld.128.ip      q0, %[a], 16                \n"
        "ee.vld.128.ip      q1, %[b], 16                \n"
        "loopgtz            %[n], .Ldot%=               \n"
        "ee.vmulas.s8.accx.ld.ip q0, %[a], 16, q0, q1   \n"
        "ee.vld.128.ip      q1, %[b], 16                \n"
        ".Ldot%=:                                       \n"
        "ee.vmulas.s8.accx  q0, q1                      \n"
        : [a] "+r" (a), [b] "+r" (b)
        : [n] "r" (blocks - 1)
        : "memory"
    );
}

static inline void esp_nn_accx_zero_s3(void)
{
    asm volatile ("ee.zero.accx" ::: "memory");
}

/* Low 32 bits of ACCX. The nops cover the MAC-to-read latency. */
static inline int32_t esp_nn_accx_read_s3(void)
{
    int32_t acc;
    asm volatile (
        "nop                    \n"
        "nop                    \n"
        "rur.accx_0 %[acc]      \n"
        : [acc] "=r" (acc) :: "memory");
    return acc;
}
