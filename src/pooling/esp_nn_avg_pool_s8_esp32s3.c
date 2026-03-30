/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized avg pool wrapper.
 * Routes to existing assembly for channels%4==0,
 * provides int16-accumulation C path for other cases.
 */

#include <stdint.h>
#include <string.h>
#include <common_functions.h>

/* Existing S3 assembly (handles depth%4==0) */
extern void esp_nn_avg_pool_s8_esp32s3_asm(const int8_t *input,
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
                             const uint16_t channels);

void esp_nn_avg_pool_s8_esp32s3(const int8_t *input,
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
    /* Use existing assembly for channels % 4 == 0 */
    if (channels % 4 == 0) {
        esp_nn_avg_pool_s8_esp32s3_asm(input, input_wd, input_ht, output,
                                        output_wd, output_ht, stride_wd, stride_ht,
                                        filter_wd, filter_ht, pad_wd, pad_ht,
                                        activation_min, activation_max, channels);
        return;
    }

    /* C path with int16 accumulation for non-aligned channels */
    int16_t acc_buf[channels];

    int32_t base_y = -pad_ht;
    for (int32_t out_y = 0; out_y < output_ht; out_y++, base_y += stride_ht) {
        int32_t base_x = -pad_wd;
        for (int32_t out_x = 0; out_x < output_wd; out_x++, base_x += stride_wd) {
            int32_t fy_start = max(0, -base_y);
            int32_t fx_start = max(0, -base_x);
            int32_t fy_end = min(filter_ht, input_ht - base_y);
            int32_t fx_end = min(filter_wd, input_wd - base_x);
            int32_t filter_cnt = (fy_end - fy_start) * (fx_end - fx_start);

            memset(acc_buf, 0, channels * sizeof(int16_t));

            for (int32_t fy = fy_start; fy < fy_end; fy++) {
                for (int32_t fx = fx_start; fx < fx_end; fx++) {
                    int32_t in_idx = ((base_y + fy) * input_wd + (base_x + fx)) * channels;
                    for (int c = 0; c < channels; c++) {
                        acc_buf[c] += (int16_t)input[in_idx + c];
                    }
                }
            }

            int32_t half_cnt = filter_cnt / 2;
            int32_t out_idx = (out_y * output_wd + out_x) * channels;
            for (int c = 0; c < channels; c++) {
                int32_t result = acc_buf[c];
                result = result > 0 ? (result + half_cnt) / filter_cnt
                                    : (result - half_cnt) / filter_cnt;
                result = max(result, activation_min);
                result = min(result, activation_max);
                output[out_idx + c] = (int8_t)result;
            }
        }
    }
}
