/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stddef.h>
#include <common_functions.h>
#include <esp_nn_ansi_headers.h>
#include "esp_nn_fc_internal.h"
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
            /* for (k = 0; k < c32; k++) xacc += dot(in + 32 * k, flt + 32 * k, 32); */
            ESP_NN_HWLOOP_BEGIN
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            /* MAC pair 1 + load next input[0:16] */
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            /* Load next filter[0:16] while MAC settles */
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            /* MAC pair 2 + load next input[16:32] */
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            /* Load next filter[16:32] */
            ESP_NN_HWLOOP_LAST("esp.vld.128.ip  q3, x31, 16", 1)
            ESP_NN_HWLOOP_END

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

/* Requantize, add the output offset, clamp, and narrow to s8. Shared by
 * the per-tensor, per-channel, and batch kernels so a fix to this step
 * lands in one place. */
static inline int8_t fc_requantize_s8(int32_t acc, int32_t mult, int32_t shift,
                                      int32_t out_offset,
                                      int32_t act_min, int32_t act_max)
{
    acc = esp_nn_requantize(acc, mult, shift);
    acc += out_offset;
    acc = max(acc, act_min);
    acc = min(acc, act_max);
    return (int8_t) acc;
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
    ESP_NN_PIE_ENABLE();

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
            out_data[out_c] = fc_requantize_s8(result, out_mult, out_shift,
                                               out_offset, activation_min,
                                               activation_max);
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
        out_data[out_c] = fc_requantize_s8(result, out_mult, out_shift,
                                           out_offset, activation_min,
                                           activation_max);
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
    ESP_NN_PIE_ENABLE();

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
            out_data[out_c] = fc_requantize_s8(result, out_mult[out_c], out_shift[out_c],
                                               out_offset, activation_min,
                                               activation_max);
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
        out_data[out_c] = fc_requantize_s8(result, out_mult[out_c], out_shift[out_c],
                                           out_offset, activation_min,
                                           activation_max);
    }
}

/* Batched per-channel FC for P4 and S31. One call covers every input row
 * against the same filter. Layout: input [batches][row_len],
 * out_data [batches][out_channels].
 *
 * When filter_offset == 0 the loop is out-channel outer and batch inner,
 * matching the S3 batch kernel:
 *   1. input_offset * sum(filter_row) + bias does not depend on the batch,
 *      so each filter row is summed once;
 *   2. that filter row is then reused for every input row.
 * Rows shorter than 16 keep that order and run the dot in scalar code,
 * because the PIE dot is only used at 16 elements and above. A nonzero
 * filter offset still needs a per-row input sum, so that case calls the
 * per-row PIE kernel once per batch row. A single row has nothing to
 * reuse, so it calls the per-row kernel too. Unlike the S3 batch kernel,
 * the PIE path does not fall back when an operand is unaligned. That is
 * safe because ESP_NN_PIE_ENABLE() below sets the unaligned-load bit.
 * Without that bit, esp.vld.128 loads from the address rounded down to
 * 16 bytes. The other paths return before that enable, and the per-row
 * kernel sets the same bit. */
void esp_nn_fully_connected_per_ch_s8_batch_riscv_pie(const int8_t *input_data,
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
                                        const int32_t activation_max,
                                        const int32_t batches)
{
    if (filter_offset != 0 || batches == 1) {
        for (int32_t b = 0; b < batches; b++) {
            esp_nn_fully_connected_per_ch_s8_riscv_pie(
                    input_data + b * row_len, input_offset, row_len, filter_data,
                    filter_offset, bias, out_data + b * out_channels, out_channels,
                    out_offset, out_shift, out_mult, activation_min, activation_max);
        }
        return;
    }

    if (row_len < 16) {
        for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
            const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
            const int32_t mult = out_mult[out_c];
            const int32_t shift = out_shift[out_c];
            const int32_t bch = bias ? bias[out_c] : 0;
            for (int32_t b = 0; b < batches; b++) {
                const int8_t *in = input_data + (int32_t)b * row_len;
                int32_t result = bch;
                for (int32_t i = 0; i < row_len; i++) {
                    result += (int32_t)filter_row[i] * ((int32_t)in[i] + input_offset);
                }
                out_data[(int32_t)b * out_channels + out_c] =
                        fc_requantize_s8(result, mult, shift, out_offset,
                                         activation_min, activation_max);
            }
        }
        return;
    }

    ESP_NN_PIE_ENABLE();

    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        const int8_t *filter_row = filter_data + (int32_t)row_len * out_c;
        int32_t corr = 0;
        if (input_offset != 0) {
            corr = esp_nn_filter_sum_s8_riscv_pie(filter_row, row_len) * input_offset;
        }
        if (bias) {
            corr += bias[out_c];
        }
        const int32_t mult = out_mult[out_c];
        const int32_t shift = out_shift[out_c];
        for (int32_t b = 0; b < batches; b++) {
            int32_t result = fc_dot_s8_pie(input_data + (int32_t)b * row_len,
                                           filter_row, row_len);
            result += corr;
            out_data[(int32_t)b * out_channels + out_c] =
                    fc_requantize_s8(result, mult, shift, out_offset,
                                     activation_min, activation_max);
        }
    }
}

#if ESP_NN_FC_BATCH_VIA_CONV
/* Weight zero-point 0 on a shape the per-chip gate accepts runs as the 1x1
 * convolution in scratch. Anything else runs the batch kernel above. */
void esp_nn_fully_connected_per_ch_s8_batch_scratch_riscv_pie(const int8_t *input_data,
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
                                    const int32_t activation_max,
                                    const int32_t batches,
                                    void *scratch,
                                    size_t scratch_size)
{
    if (esp_nn_fc_per_ch_s8_batch_via_conv(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult, activation_min,
                                           activation_max, batches,
                                           scratch, scratch_size)) {
        return;
    }
    esp_nn_fully_connected_per_ch_s8_batch_riscv_pie(input_data, input_offset, row_len,
            filter_data, filter_offset, bias, out_data, out_channels,
            out_offset, out_shift, out_mult, activation_min, activation_max,
            batches);
}
#endif /* ESP_NN_FC_BATCH_VIA_CONV */
