/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>
#include "../common/esp_nn_filter_sum_riscv_pie.h"

/**
 * Fully connected layer for s8 using ESP32-P4 PIE SIMD.
 *
 * Uses esp.vmulas.s8.xacc.ld.ip for fused 16-wide s8 MAC + load.
 * Per-channel correction `filter_sum * input_offset` is precomputed once
 * so the PIE dot product runs with non-zero input_offset (the common
 * TFLite case). Non-zero filter_offset is rare and falls back to scalar.
 *
 * Inner loop is software-pipelined:
 *   iteration N: MAC(q0,q1) + load_next_input(q0)
 *                load_next_filter(q1)     <- hides MAC latency
 *                counter_update           <- independent of above
 */

/* Core dot product: PIE-accelerated when row_len >= 16 */
static inline __attribute__((always_inline))
int32_t fc_dot_s8_pie(const int8_t *input, const int8_t *filter, int32_t row_len)
{
    int32_t result = 0;

    if (row_len >= 32) {
        /* Double-pumped: process 32 elements per iteration
         * Uses q0/q1 for first pair, q2/q3 for second pair.
         * Software pipelined: first block is loaded up front, each loop
         * iteration MACs one block and loads the next -> N-1 iterations. */
        int32_t c32 = (row_len >> 5) - 1;
        int32_t rem16 = row_len & 16;
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"

            /* Prime the pipeline: load first 32 bytes */
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"

            "beqz   %[c32], 2f                      \n\t"
            /* Zero-overhead hardware loop; end label sits ON the last insn */
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            /* MAC pair 1 + load next input[0:16] */
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            /* Load next filter[0:16] while MAC settles */
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            /* MAC pair 2 + load next input[16:32] */
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            "1:                                     \n\t"
            /* Load next filter[16:32] */
            "esp.vld.128.ip  q3, x31, 16            \n\t"

            "2:                                     \n\t"
            /* Drain pipeline: final two MACs */
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"

            /* Handle 16-element remainder if any */
            "beqz   %[rem16], 3f                    \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "3:                                     \n\t"

            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(input), [flt] "r"(filter), [c32] "r"(c32), [rem16] "r"(rem16)
            : "x30", "x31"
        );
    } else if (row_len >= 16) {
        /* Exactly one full 16-element block for 16-31 element rows */
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(input), [flt] "r"(filter)
            : "x30", "x31"
        );
    }

    /* Scalar remainder */
    for (int32_t idx = row_len & ~15; idx < row_len; idx++) {
        result += (int32_t)input[idx] * (int32_t)filter[idx];
    }

    return result;
}


void esp_nn_fully_connected_s8_riscv_pie(const int8_t *input_data,
                                        const int32_t input_offset,
                                        const uint16_t row_len,
                                        const int8_t *filter_data,
                                        const int32_t filter_offset,
                                        const int32_t *bias,
                                        int8_t *out_data,
                                        const uint16_t out_channels,
                                        const int32_t out_offset,
                                        const int32_t out_shift,
                                        const int32_t out_mult,
                                        const int32_t activation_min,
                                        const int32_t activation_max)
{
    /* Enable PIE once for all channels */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    /* SIMD path with optional corrections. Math:
     *   sum((x+io)*(w+fo)) = sum(x*w) + io*sum(w) + fo*sum(x) + row_len*io*fo
     * fc_dot_s8_pie computes sum(x*w); the rest is folded into per-ch corrections.
     * Below one SIMD width that two-pass setup is pure overhead, so tiny rows
     * take a single-pass scalar loop instead (measured at parity with ANSI). */
    if (row_len < 16) {
        for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
            const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
            int32_t result = 0;
            for (int32_t i = 0; i < row_len; i++) {
                result += (filter_row[i] + filter_offset) * (input_data[i] + input_offset);
            }
            if (bias) {
                result += bias[out_c];
            }
            result = esp_nn_requantize(result, out_mult, out_shift);
            result += out_offset;
            result = max(result, activation_min);
            result = min(result, activation_max);
            out_data[out_c] = (int8_t) result;
        }
        return;
    }

    int32_t input_sum = 0;
    if (filter_offset != 0) {
        for (int32_t i = 0; i < row_len; i++) {
            input_sum += input_data[i];
        }
    }
    int32_t global_corr = filter_offset * input_sum
                          + (int32_t)row_len * input_offset * filter_offset;

    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
        /* Per-channel correction, inline (no out_channels-sized VLA) */
        int32_t corr = global_corr;
        if (input_offset != 0) {
            corr += esp_nn_filter_sum_s8_riscv_pie(filter_row, row_len) * input_offset;
        }
        if (bias) {
            corr += bias[out_c];
        }
        int32_t result = fc_dot_s8_pie(input_data, filter_row, row_len);
        result += corr;
        result = esp_nn_requantize(result, out_mult, out_shift);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}

void esp_nn_fully_connected_per_ch_s8_riscv_pie(const int8_t *input_data,
                                        const int32_t input_offset,
                                        const uint16_t row_len,
                                        const int8_t *filter_data,
                                        const int32_t filter_offset,
                                        const int32_t *bias,
                                        int8_t *out_data,
                                        const uint16_t out_channels,
                                        const int32_t out_offset,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t activation_min,
                                        const int32_t activation_max)
{
    /* Enable PIE once for all channels */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    /* Tiny rows: single-pass scalar (see comment in the per-tensor variant) */
    if (row_len < 16) {
        for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
            const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
            int32_t result = 0;
            for (int32_t i = 0; i < row_len; i++) {
                result += (filter_row[i] + filter_offset) * (input_data[i] + input_offset);
            }
            if (bias) {
                result += bias[out_c];
            }
            result = esp_nn_requantize(result, out_mult[out_c], out_shift[out_c]);
            result += out_offset;
            result = max(result, activation_min);
            result = min(result, activation_max);
            out_data[out_c] = (int8_t) result;
        }
        return;
    }

    int32_t input_sum = 0;
    if (filter_offset != 0) {
        for (int32_t i = 0; i < row_len; i++) {
            input_sum += input_data[i];
        }
    }
    int32_t global_corr = filter_offset * input_sum
                          + (int32_t)row_len * input_offset * filter_offset;

    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
        /* Per-channel correction, inline (no out_channels-sized VLA) */
        int32_t corr = global_corr;
        if (input_offset != 0) {
            corr += esp_nn_filter_sum_s8_riscv_pie(filter_row, row_len) * input_offset;
        }
        if (bias) {
            corr += bias[out_c];
        }
        int32_t result = fc_dot_s8_pie(input_data, filter_row, row_len);
        result += corr;
        result = esp_nn_requantize(result, out_mult[out_c], out_shift[out_c]);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}
