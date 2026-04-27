/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized mean reduction for NHWC int8 tensors.
 * Uses int16 accumulation for small spatial sizes (H*W <= 256),
 * int32 for larger. Accumulates all channels at once per spatial position.
 */

#include <stdint.h>
#include <string.h>
#include <common_functions.h>

void esp_nn_mean_nhwc_s8_esp32s3(const int8_t *input,
                                  int8_t *output,
                                  const int32_t height,
                                  const int32_t width,
                                  const int32_t channels,
                                  const int32_t input_zero_point,
                                  const int32_t output_zero_point,
                                  const int32_t multiplier,
                                  const int32_t shift)
{
    const int32_t num_elements = height * width;
    const int32_t zp_correction = num_elements * input_zero_point;

    if (num_elements <= 256 && channels <= 512) {
        /* int16 accumulation (safe: 256 * 127 = 32,512 < 32,767) */
        /* Process 8 channels at a time using int16 accumulators */
        int16_t acc16[channels];
        memset(acc16, 0, channels * sizeof(int16_t));

        const int8_t *ptr = input;
        for (int i = 0; i < num_elements; i++) {
            /* Inner loop — compiler should auto-vectorize with -O2 */
            for (int c = 0; c < channels; c++) {
                acc16[c] += (int16_t)ptr[c];
            }
            ptr += channels;
        }

        /* Requantize per channel */
        for (int c = 0; c < channels; c++) {
            int32_t sum = (int32_t)acc16[c] - zp_correction;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    } else if (channels <= 512) {
        /* int32 accumulation for larger spatial sizes */
        int32_t acc[channels];
        memset(acc, 0, channels * sizeof(int32_t));

        const int8_t *ptr = input;
        for (int i = 0; i < num_elements; i++) {
            for (int c = 0; c < channels; c++) {
                acc[c] += ptr[c];
            }
            ptr += channels;
        }

        for (int c = 0; c < channels; c++) {
            int32_t sum = acc[c] - zp_correction;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    } else {
        /* Per-channel fallback for huge channel counts */
        for (int c = 0; c < channels; c++) {
            int32_t sum = 0;
            for (int i = 0; i < num_elements; i++) {
                sum += input[i * channels + c];
            }
            sum -= zp_correction;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    }
}
