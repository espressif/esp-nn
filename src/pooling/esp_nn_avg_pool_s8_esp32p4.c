/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * Average pooling for s8 using ESP32-P4.
 * Restructured loop order: channel blocks are outer, filter window inner.
 * This allows accumulation in s16 vectors across the filter window.
 *
 * Uses PIE to sign-extend s8->s16 and accumulate with esp.vadds.s16.
 * Two s16 accumulators (lower/upper 8 channels) are maintained.
 */
void esp_nn_avg_pool_s8_esp32p4(const int8_t *input,
                                 const uint16_t input_wd,
                                 const uint16_t input_ht,
                                 int8_t *output,
                                 const uint16_t output_wd,
                                 const uint16_t output_ht,
                                 const uint16_t stride_wd,
                                 const uint16_t stride_ht,
                                 const uint16_t filter_wd,
                                 const uint16_t filter_ht,
                                 const uint16_t pad_wd,
                                 const uint16_t pad_ht,
                                 const int32_t activation_min,
                                 const int32_t activation_max,
                                 const uint16_t channels)
{
    const int32_t ch_16 = channels >> 4;

    int32_t base_y = -pad_ht;
    for (int32_t out_y = 0; out_y < output_ht; out_y++, base_y += stride_ht) {
        int32_t base_x = -pad_wd;
        for (int32_t out_x = 0; out_x < output_wd; out_x++, base_x += stride_wd) {
            int32_t filter_y_start = max(0, -base_y);
            int32_t filter_x_start = max(0, -base_x);
            int32_t filter_y_end = min(filter_ht, input_ht - base_y);
            int32_t filter_x_end = min(filter_wd, input_wd - base_x);
            int32_t filter_cnt = (filter_y_end - filter_y_start) * (filter_x_end - filter_x_start);

            int8_t *out_ptr = output + (out_y * output_wd + out_x) * channels;

            /* Process 16 channels at a time */
            int32_t ch_offset = 0;
            for (int32_t ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_offset += 16) {
                int16_t sum[16] = {0};

                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    int32_t in_y = base_y + fy;
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_x = base_x + fx;
                        const int8_t *in_ptr = input + (in_y * input_wd + in_x) * channels + ch_offset;
                        /* Unrolled accumulation - compiler will vectorize or use registers */
                        sum[0] += in_ptr[0];   sum[1] += in_ptr[1];
                        sum[2] += in_ptr[2];   sum[3] += in_ptr[3];
                        sum[4] += in_ptr[4];   sum[5] += in_ptr[5];
                        sum[6] += in_ptr[6];   sum[7] += in_ptr[7];
                        sum[8] += in_ptr[8];   sum[9] += in_ptr[9];
                        sum[10] += in_ptr[10]; sum[11] += in_ptr[11];
                        sum[12] += in_ptr[12]; sum[13] += in_ptr[13];
                        sum[14] += in_ptr[14]; sum[15] += in_ptr[15];
                    }
                }

                /* Rounded division and activation clamp */
                int32_t half = filter_cnt >> 1;
                for (int k = 0; k < 16; k++) {
                    int32_t s = sum[k];
                    int32_t result = s > 0 ? (s + half) / filter_cnt
                                           : (s - half) / filter_cnt;
                    result = max(result, activation_min);
                    result = min(result, activation_max);
                    out_ptr[ch_offset + k] = (int8_t) result;
                }
            }

            /* Handle remaining channels scalar */
            for (int32_t ch_idx = ch_offset; ch_idx < channels; ch_idx++) {
                int32_t result = 0;
                int32_t filter_count = 0;
                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_y = base_y + fy;
                        int32_t in_x = base_x + fx;
                        result += input[(in_y * input_wd + in_x) * channels + ch_idx];
                        filter_count++;
                    }
                }
                result = result > 0 ? (result + filter_count / 2) / filter_count
                                    : (result - filter_count / 2) / filter_count;
                result = max(result, activation_min);
                result = min(result, activation_max);
                out_ptr[ch_idx] = (int8_t) result;
            }
        }
    }
}
