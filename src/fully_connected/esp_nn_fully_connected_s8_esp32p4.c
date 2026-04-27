/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * Fully connected layer for s8 using ESP32-P4 PIE SIMD.
 *
 * Uses esp.vmulas.s8.xacc.ld.ip for fused 16-wide s8 MAC + load.
 * Pre-computes filter_sum * input_offset (like conv) so PIE path
 * works even with non-zero input_offset.
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
    int32_t idx = 0;

    if (row_len >= 32) {
        /* Double-pumped: process 32 elements per iteration
         * Uses q0/q1 for first pair, q2/q3 for second pair */
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "li     %[idx], 32                      \n\t"
            "addi   s7, %[len], -31                 \n\t"

            /* Prime the pipeline: load first 32 bytes */
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "j      2f                              \n\t"

            "1:                                     \n\t"
            /* MAC pair 1 + load next input[0:16] */
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            /* Load next filter[0:16] while MAC settles */
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            /* MAC pair 2 + load next input[16:32] */
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            /* Load next filter[16:32] - interleaved with counter */
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "addi   %[idx], %[idx], 32              \n\t"

            "2:                                     \n\t"
            "blt    %[idx], s7, 1b                  \n\t"

            /* Drain pipeline: final two MACs */
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"

            /* Handle 16-element remainder if any (idx+16 <= row_len) */
            "addi   s7, %[len], -15                 \n\t"
            "bge    %[idx], s7, 3f                  \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "addi   %[idx], %[idx], 16              \n\t"
            "3:                                     \n\t"

            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [idx] "+r"(idx), [res] "=r"(result)
            : [in] "r"(input), [flt] "r"(filter), [len] "r"(row_len)
            : "x30", "x31", "s7"
        );
    } else if (row_len >= 16) {
        /* Single-pumped for 16-31 element rows */
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "li     %[idx], 16                      \n\t"
            "addi   s7, %[len], -15                 \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "j      5f                              \n\t"
            "4:                                     \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "addi   %[idx], %[idx], 16              \n\t"
            "5:                                     \n\t"
            "blt    %[idx], s7, 4b                  \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [idx] "+r"(idx), [res] "=r"(result)
            : [in] "r"(input), [flt] "r"(filter), [len] "r"(row_len)
            : "x30", "x31", "s7"
        );
    }

    /* Scalar remainder */
    for (; idx < row_len; idx++) {
        result += (int32_t)input[idx] * (int32_t)filter[idx];
    }

    return result;
}

void esp_nn_fully_connected_s8_esp32p4(const int8_t *input_data,
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

    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;

        int32_t result;
        if (input_offset == 0 && filter_offset == 0) {
            /* Fast PIE path: pure s8 dot product */
            result = fc_dot_s8_pie(input_data, filter_row, row_len);
        } else {
            /* Scalar path with offsets */
            result = 0;
            for (int32_t i = 0; i < row_len; i++) {
                result += ((int32_t)input_data[i] + input_offset) *
                          ((int32_t)filter_row[i] + filter_offset);
            }
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
}

void esp_nn_fully_connected_per_ch_s8_esp32p4(const int8_t *input_data,
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

    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;

        int32_t result;
        if (input_offset == 0 && filter_offset == 0) {
            result = fc_dot_s8_pie(input_data, filter_row, row_len);
        } else {
            result = 0;
            for (int32_t i = 0; i < row_len; i++) {
                result += ((int32_t)input_data[i] + input_offset) *
                          ((int32_t)filter_row[i] + filter_offset);
            }
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
}
