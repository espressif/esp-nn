/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized mean reduction for NHWC int8 tensors.
 * Accumulates all channels at once per spatial position using int32 accumulators.
 * Memory access pattern: sequential (NHWC layout, contiguous channel data).
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
    const int32_t spatial_size = num_elements * channels;

    /* Allocate int32 accumulators on stack for up to 512 channels.
     * For larger channel counts, fall back to per-channel accumulation. */
    if (channels <= 512) {
        int32_t acc[channels];
        memset(acc, 0, channels * sizeof(int32_t));

        /* Accumulate all spatial positions — sequential memory access */
        const int8_t *ptr = input;
        for (int i = 0; i < num_elements; i++) {
            for (int c = 0; c < channels; c++) {
                acc[c] += ptr[c];
            }
            ptr += channels;
        }

        /* Requantize per channel */
        const int32_t zp_correction = num_elements * input_zero_point;
        for (int c = 0; c < channels; c++) {
            int32_t sum = acc[c] - zp_correction;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    } else {
        /* Fallback for very large channel counts */
        for (int c = 0; c < channels; c++) {
            int32_t sum = 0;
            for (int i = 0; i < num_elements; i++) {
                sum += input[i * channels + c];
            }
            sum -= num_elements * input_zero_point;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    }
}
