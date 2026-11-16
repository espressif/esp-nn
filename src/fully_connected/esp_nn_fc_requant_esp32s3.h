/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

/* Bit-exact esp_nn_multiply_by_quantized_mult() for mult >= 0 (the TFLite
 * range [0, 2^31)) and shift >= -31. It is the per-row cost of short FC
 * rows, so it drops what that range makes dead.
 * - The INT32_MIN * INT32_MIN saturation needs a negative mult.
 * - With mult >= 0, the sign-dependent nudge and the truncating >> 31 equal
 *   floor((v * mult + 2^30) / 2^31). */
static inline int32_t fc_requant(int32_t x, int32_t mult, int32_t shift)
{
    const int32_t left = shift > 0 ? shift : 0;
    const int32_t right = left - shift; /* max(-shift, 0) */
    const int32_t v = (int32_t)((uint32_t)x << left);
    /* (p + 2^30) >> 31 is p >> 31 plus bit 30 of p: no 64-bit add. */
    const int64_t p = (int64_t)v * mult;
    const int32_t r = (int32_t)(p >> 31) + (int32_t)(((uint32_t)p >> 30) & 1);

    /* Divide by 2^right, rounding half away from zero. ssr + sll leave the
     * bits shifted out at the top of frac (0 for right = 0, as SAR = 0).
     * Round up when they are above one half, or equal to it and r >= 0. */
    int32_t res;
    uint32_t frac;
    asm ("ssr %[sh] \n sra %[res], %[r] \n sll %[frac], %[r] \n"
         : [res] "=&r" (res), [frac] "=r" (frac) : [sh] "r" (right), [r] "r" (r));
    return res + (frac > 0x7fffffffu + (uint32_t)(r < 0));
}
