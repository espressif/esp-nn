/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <common_functions.h>

/*
 * Requantize `lanes` accumulators of one output channel and store them with
 * a byte stride. Bit-exact with esp_nn_requantize.
 *
 * For shift e = -shift in [1, 21] and |x| < 2^29 the reference reduces to
 * one high multiply:
 *   r = (mulhsu(2x, 2M) + 1 + 2^e - 2 * (x < 0)) >> (e + 1)
 * mulhsu(2x, 2M) is floor(2xM / 2^31). The doubling mul's nudge, 2^30 on
 * xM, is the +1 at that scale, so the low product word is not needed.
 * The divide-by-POT rounds half away from zero, which is the -(x < 0)
 * term (scaled by 2). Nested floors compose, so the two roundings fold
 * into one shift. out_offset is folded in as out_offset * 2^(e+1).
 * The bounds keep every sum inside int32.
 *
 * `acc_bound` bounds |res[i]|; other shifts and larger values take the
 * generic path. The high multiply is unsigned in the multiplier, so a
 * negative multiplier takes the generic path too.
 */
static inline __attribute__((always_inline))
bool esp_nn_conv_requant_fast_ok(int32_t comb, int32_t mult, int32_t shift,
                                 int32_t acc_bound, int32_t out_offset)
{
    const int32_t lim = (1 << 29) - acc_bound;
    return mult >= 0 && shift <= -1 && shift >= -21 && lim > 0 && comb < lim &&
            comb > -lim && out_offset >= -128 && out_offset <= 127;
}

/* Per-channel constants of the fast requant: p[0] = 2 * comb,
 * p[1] = 2 * mult, p[2] = the rounding term with out_offset folded in.
 * The shift is e1 = 1 - shift. */
static inline __attribute__((always_inline))
void esp_nn_conv_requant_params(int32_t comb, int32_t mult, int32_t shift,
                                int32_t out_offset, int32_t *p)
{
    const int32_t e1 = 1 - shift;
    p[0] = comb * 2;
    p[1] = (int32_t) ((uint32_t) mult << 1);
    p[2] = 1 + (1 << (e1 - 1)) + out_offset * (1 << e1);
}

/*
 * The fast path only: the caller has checked esp_nn_conv_requant_fast_ok.
 * Four lanes per pass in asm. The core issues one instruction per cycle, so
 * each pass interleaves four lanes to hide the mulhsu latency. esp.addx2
 * folds the doublings and esp.sat does the clamp. That is about 10.5
 * instructions per lane, against about 15 from C (branchy min/max).
 */
static inline __attribute__((always_inline))
void esp_nn_conv_requant_lanes_fast(const int32_t *res, int32_t lanes,
                                    const int32_t *prm, int32_t shift,
                                    int8_t *out, int32_t stride,
                                    int32_t act_min, int32_t act_max)
{
    const int32_t e1 = 1 - shift;
    const int32_t comb2 = prm[0];
    const uint32_t m2 = (uint32_t) prm[1];
    const int32_t rnd = prm[2];
    const int32_t *quad_end = res + (lanes & ~3);
    if (res != quad_end) {
        int32_t x0, x1, x2, x3, t;
        /* x = comb2 + 2 res; a = mulhsu(x, m2) + 2 (a < 0 ? -1 : 0) + rnd;
         * out = clamp(a >> e1). a has the sign of x (m2 >= 0; for m2 == 0
         * both signs give out_offset), so one temp holds the sign. */
        asm volatile (
            "1:                              \n\t"
            "lw     %[x0], 0(%[res])         \n\t"
            "lw     %[x1], 4(%[res])         \n\t"
            "lw     %[x2], 8(%[res])         \n\t"
            "lw     %[x3], 12(%[res])        \n\t"
            "esp.addx2 %[x0], %[c2], %[x0]   \n\t"
            "esp.addx2 %[x1], %[c2], %[x1]   \n\t"
            "esp.addx2 %[x2], %[c2], %[x2]   \n\t"
            "esp.addx2 %[x3], %[c2], %[x3]   \n\t"
            "mulhsu %[x0], %[x0], %[m2]      \n\t"
            "mulhsu %[x1], %[x1], %[m2]      \n\t"
            "mulhsu %[x2], %[x2], %[m2]      \n\t"
            "mulhsu %[x3], %[x3], %[m2]      \n\t"
            "addi   %[res], %[res], 16       \n\t"
            "srai   %[t], %[x0], 31          \n\t"
            "esp.addx2 %[x0], %[x0], %[t]    \n\t"
            "srai   %[t], %[x1], 31          \n\t"
            "esp.addx2 %[x1], %[x1], %[t]    \n\t"
            "srai   %[t], %[x2], 31          \n\t"
            "esp.addx2 %[x2], %[x2], %[t]    \n\t"
            "srai   %[t], %[x3], 31          \n\t"
            "esp.addx2 %[x3], %[x3], %[t]    \n\t"
            "add    %[x0], %[x0], %[rnd]     \n\t"
            "add    %[x1], %[x1], %[rnd]     \n\t"
            "add    %[x2], %[x2], %[rnd]     \n\t"
            "add    %[x3], %[x3], %[rnd]     \n\t"
            "sra    %[x0], %[x0], %[e1]      \n\t"
            "sra    %[x1], %[x1], %[e1]      \n\t"
            "sra    %[x2], %[x2], %[e1]      \n\t"
            "sra    %[x3], %[x3], %[e1]      \n\t"
            "esp.sat %[x0], %[lo], %[hi]     \n\t"
            "esp.sat %[x1], %[lo], %[hi]     \n\t"
            "esp.sat %[x2], %[lo], %[hi]     \n\t"
            "esp.sat %[x3], %[lo], %[hi]     \n\t"
            "sb     %[x0], 0(%[out])         \n\t"
            "add    %[out], %[out], %[st]    \n\t"
            "sb     %[x1], 0(%[out])         \n\t"
            "add    %[out], %[out], %[st]    \n\t"
            "sb     %[x2], 0(%[out])         \n\t"
            "add    %[out], %[out], %[st]    \n\t"
            "sb     %[x3], 0(%[out])         \n\t"
            "add    %[out], %[out], %[st]    \n\t"
            "bne    %[res], %[end], 1b       \n\t"
            : [res] "+r"(res), [out] "+r"(out),
              [x0] "=&r"(x0), [x1] "=&r"(x1), [x2] "=&r"(x2), [x3] "=&r"(x3),
              [t] "=&r"(t)
            : [c2] "r"(comb2), [m2] "r"(m2), [rnd] "r"(rnd), [e1] "r"(e1),
              [lo] "r"(act_min), [hi] "r"(act_max), [st] "r"(stride),
              [end] "r"(quad_end)
            : "memory"
        );
    }
    for (int32_t p = 0; p < (lanes & 3); p++) {
        const int32_t x2 = res[p] * 2 + comb2;
        int32_t a;
        asm ("mulhsu %0, %1, %2" : "=r"(a) : "r"(x2), "r"(m2));
        const int32_t sg = x2 >> 31;
        int32_t r = (a + rnd + sg + sg) >> e1;
        r = max(r, act_min);
        r = min(r, act_max);
        out[p * stride] = (int8_t) r;
    }
}

static inline __attribute__((always_inline))
void esp_nn_conv_requant_lanes_pie(const int32_t *res, int32_t lanes,
                                   int32_t comb, int32_t mult, int32_t shift,
                                   int32_t acc_bound, int8_t *out,
                                   int32_t stride, int32_t out_offset,
                                   int32_t act_min, int32_t act_max)
{
    if (esp_nn_conv_requant_fast_ok(comb, mult, shift, acc_bound, out_offset)) {
        int32_t prm[3];
        esp_nn_conv_requant_params(comb, mult, shift, out_offset, prm);
        esp_nn_conv_requant_lanes_fast(res, lanes, prm, shift, out, stride,
                                       act_min, act_max);
        return;
    }
    /* The exact requant, also with CONFIG_NN_SKIP_NUDGE, so all pixels of
     * a layer round the same way. */
    for (int32_t p = 0; p < lanes; p++) {
        int32_t r = esp_nn_multiply_by_quantized_mult(res[p] + comb, mult, shift);
        r += out_offset;
        r = max(r, act_min);
        r = min(r, act_max);
        out[p * stride] = (int8_t) r;
    }
}
