/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * FC multi-path dispatcher for ESP32-S3.
 * - Pre-computes offset corrections per channel in C
 * - Dispatches to s8 MAC assembly (aligned, large row_len) or s16 assembly (fallback)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <common_functions.h>

/* Original s16 assembly (renamed) */
extern void esp_nn_fc_s16_esp32s3(const int8_t *input_data,
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
                                   const int32_t activation_max);

extern void esp_nn_fc_per_ch_s16_esp32s3(const int8_t *input_data,
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
                                          const int32_t activation_max);

/* Shared s8 dot product from common — handles unaligned filter via USAR+QUP */
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a,
                                                const int8_t *b,
                                                int32_t len_div16);

void esp_nn_fully_connected_s8_esp32s3(const int8_t *input_data,
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
    /* Quick check: s8 fast path only for aligned, row_len%16, no filter_offset */
    if (__builtin_expect(filter_offset != 0 || row_len < 16
        || ((uintptr_t)input_data & 15), 0)) {
        /* Fallback to original s16 assembly — tail call, no extra overhead */
        esp_nn_fc_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                              filter_offset, bias, out_data, out_channels,
                              out_offset, out_shift, out_mult,
                              activation_min, activation_max);
        return;
    }
    {
        int32_t row_len_div16 = row_len >> 4;

        /* Pre-compute per-channel corrections once */
        int32_t corrections[out_channels];
        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            int32_t corr = 0;
            if (input_offset != 0) {
                int32_t filter_sum = 0;
                for (int i = 0; i < row_len; i++) {
                    filter_sum += f_ptr[i];
                }
                corr = filter_sum * input_offset;
            }
            if (bias) {
                corr += bias[ch];
            }
            corrections[ch] = corr;
        }

        int32_t row_len_rem = row_len & 15;
        int32_t simd_bytes = row_len_div16 << 4;

        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            int32_t acc = esp_nn_dot_s8_unaligned_esp32s3(input_data, f_ptr, row_len_div16);

            /* Scalar remainder for non-multiple-of-16 row_len */
            for (int i = 0; i < row_len_rem; i++) {
                acc += (int32_t)input_data[simd_bytes + i] * (int32_t)f_ptr[simd_bytes + i];
            }

            acc += corrections[ch];

            acc = esp_nn_multiply_by_quantized_mult(acc, out_mult, out_shift);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[ch] = (int8_t)acc;
        }
    }
}

void esp_nn_fully_connected_per_ch_s8_esp32s3(const int8_t *input_data,
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
    if (__builtin_expect(filter_offset != 0 || row_len < 16
        || ((uintptr_t)input_data & 15), 0)) {
        esp_nn_fc_per_ch_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                                     filter_offset, bias, out_data, out_channels,
                                     out_offset, out_shift, out_mult,
                                     activation_min, activation_max);
        return;
    }
    {
        int32_t row_len_div16 = row_len >> 4;

        /* Pre-compute per-channel corrections once */
        int32_t corrections[out_channels];
        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            int32_t corr = 0;
            if (input_offset != 0) {
                int32_t filter_sum = 0;
                for (int i = 0; i < row_len; i++) {
                    filter_sum += f_ptr[i];
                }
                corr = filter_sum * input_offset;
            }
            if (bias) {
                corr += bias[ch];
            }
            corrections[ch] = corr;
        }

        int32_t row_len_rem = row_len & 15;
        int32_t simd_bytes = row_len_div16 << 4;

        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            int32_t acc = esp_nn_dot_s8_unaligned_esp32s3(input_data, f_ptr, row_len_div16);

            for (int i = 0; i < row_len_rem; i++) {
                acc += (int32_t)input_data[simd_bytes + i] * (int32_t)f_ptr[simd_bytes + i];
            }

            acc += corrections[ch];

            acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[ch], out_shift[ch]);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[ch] = (int8_t)acc;
        }
    }
}
