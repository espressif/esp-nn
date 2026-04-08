/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Optimized 3x3 convolution for ESP32-S3.
 *
 * Key optimization vs the general aligned asm:
 * The general asm reloads input for each output channel (128× per pixel).
 * This version pre-loads the 3x3 input window into scratch (9 rows × in_ch bytes),
 * then iterates output channels with the input in L1 cache.
 *
 * For Conv[11] (26×26×128→12×12×128, 3×3 s2):
 * - Input window: 3 × 3 × 128 = 1,152 bytes (fits in L1)
 * - Filter per OC: 3 × 3 × 128 = 1,152 bytes
 * - Total for all 128 OC: 147,456 bytes (cycles through L1)
 * - Input loaded once vs 128× in the general asm
 */

#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <common_functions.h>

/*
 * Check if a conv can use the optimized 3x3 path.
 * Requirements:
 * - filter_wd == 3 && filter_ht == 3
 * - in_channels >= 16 (SIMD worth it)
 * - in_channels % 16 == 0 (aligned for ee.vld.128)
 */
int esp_nn_conv_s8_3x3_can_use(int filter_wd, int filter_ht,
                                int in_channels)
{
    return (filter_wd == 3 && filter_ht == 3 &&
            in_channels >= 16 && (in_channels % 16) == 0);
}

/*
 * Scratch size for the 3x3 optimized path:
 * - im2col buffer: 3 × 3 × in_channels bytes (input window)
 * - corrections: out_channels × 4 bytes
 */
int esp_nn_conv_s8_3x3_scratch_size(int in_channels, int out_channels)
{
    int im2col = 9 * in_channels;          /* 3×3 input window */
    int corrections = out_channels * 4;    /* bias + filter_sum * offset */
    return im2col + corrections + 32;      /* + alignment */
}

/*
 * 3x3 convolution: im2col per pixel, then dot product per output channel.
 * Uses ACCX dot product (ee.vmulas.s8.accx) for the 3×3×in_ch window.
 */
void esp_nn_conv_s8_3x3_opt(const int8_t *input,
                             const uint16_t input_wd,
                             const uint16_t input_ht,
                             const uint16_t in_channels,
                             const int32_t input_offset,
                             const uint16_t stride_wd,
                             const uint16_t stride_ht,
                             const int8_t *filter_data,
                             const int32_t *bias,
                             int8_t *out_data,
                             const uint16_t out_wd,
                             const uint16_t out_ht,
                             const uint16_t out_channels,
                             const int32_t out_offset,
                             const int32_t *out_shift,
                             const int32_t *out_mult,
                             const int32_t activation_min,
                             const int32_t activation_max,
                             void *scratch)
{
    const int window_len = 9 * in_channels; /* 3×3 window */
    const int window_len_aligned = (window_len + 15) & ~15;

    /* Scratch layout: [im2col_buf | corrections] */
    int8_t *im2col_buf = (int8_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);
    int32_t *corrections = (int32_t *)(im2col_buf + window_len_aligned);

    /* Pre-compute corrections: filter_sum * input_offset + bias */
    const int8_t *f_ptr = filter_data;
    for (int oc = 0; oc < out_channels; oc++) {
        int32_t filter_sum = 0;
        for (int i = 0; i < window_len; i++) {
            filter_sum += f_ptr[i];
        }
        corrections[oc] = filter_sum * input_offset;
        if (bias) corrections[oc] += bias[oc];
        f_ptr += window_len;
    }

    /* Zero-pad the tail of im2col buffer for aligned SIMD reads */
    memset(im2col_buf + window_len, 0, window_len_aligned - window_len);

    const int in_row_stride = input_wd * in_channels;

    for (int out_y = 0; out_y < out_ht; out_y++) {
        for (int out_x = 0; out_x < out_wd; out_x++) {
            /* Phase 1: Build im2col for this output pixel (one-time per pixel) */
            const int in_y = out_y * stride_ht;
            const int in_x = out_x * stride_wd;
            int8_t *dst = im2col_buf;
            for (int fy = 0; fy < 3; fy++) {
                const int8_t *src = input + (in_y + fy) * in_row_stride + in_x * in_channels;
                memcpy(dst, src, 3 * in_channels);
                dst += 3 * in_channels;
            }

            /* Phase 2: Dot product against each output channel's filter */
            const int8_t *filter_ptr = filter_data;
            for (int oc = 0; oc < out_channels; oc++) {
                /* ACCX dot product: im2col_buf · filter_ptr */
                int32_t acc = 0;

                /* Use SIMD dot product via ACCX */
                const int8_t *a = im2col_buf;
                const int8_t *b = filter_ptr;
                int remaining = window_len_aligned;

                __asm__ volatile("ee.zero.accx");

                /* Primed unaligned load for input */
                __asm__ volatile(
                    "ee.ld.128.usar.ip q0, %[a], 16\n"
                    : [a] "+r" (a) : : "memory"
                );

                while (remaining >= 32) {
                    __asm__ volatile(
                        "ee.vld.128.ip q4, %[a], 16\n"
                        "ee.vmulas.s8.accx.ld.ip.qup q3, %[b], 16, q2, q1, q0, q4\n"
                        "ee.vld.128.ip q2, %[a], 16\n"
                        "ee.vmulas.s8.accx.ld.ip.qup q1, %[b], 16, q0, q3, q4, q2\n"
                        "ee.orq q0, q2, q2\n"
                        "ee.orq q2, q4, q4\n"
                        : [a] "+r" (a), [b] "+r" (b)
                        : : "memory"
                    );
                    remaining -= 32;
                }
                if (remaining >= 16) {
                    __asm__ volatile(
                        "ee.vmulas.s8.accx.ld.ip q4, %[a], 16, q2, q1\n"
                        "ee.src.q.ld.ip q1, %[b], 16, q0, q4\n"
                        "ee.orq q2, q0, q0\n"
                        : [a] "+r" (a), [b] "+r" (b)
                        : : "memory"
                    );
                    remaining -= 16;
                }
                __asm__ volatile(
                    "ee.vmulas.s8.accx q2, q1\n"
                    "movi.n %[tmp], 0\n"
                    "ee.srs.accx %[acc], %[tmp], 0\n"
                    : [acc] "=r" (acc), [tmp] "=r" (remaining)
                    : : "memory"
                );

                acc += corrections[oc];
                acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[oc], out_shift[oc]);
                acc += out_offset;
                acc = max(acc, activation_min);
                acc = min(acc, activation_max);
                *out_data++ = (int8_t)acc;

                filter_ptr += window_len;
            }
        }
    }
}
