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
#include <esp_nn_ansi_headers.h>

/* Per-channel accumulators come from a caller-provided scratch buffer
 * (channels * 4 bytes, see the getter); without one the reference runs. */
static void *mean_scratch_buf_s3 = NULL;

int32_t esp_nn_get_mean_scratch_size_esp32s3(const int32_t height, const int32_t width,
                                             const int32_t channels)
{
    (void) height;
    (void) width;
    return channels * (int32_t) sizeof(int32_t);
}

void esp_nn_set_mean_scratch_buf_esp32s3(void *buffer)
{
    mean_scratch_buf_s3 = buffer;
}

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

    if (mean_scratch_buf_s3 == NULL) {
        esp_nn_mean_nhwc_s8_ansi(input, output, height, width, channels,
                                 input_zero_point, output_zero_point, multiplier, shift);
        return;
    }

    if (num_elements <= 256) {
        /* int16 accumulation (safe: 256 * 127 = 32,512 < 32,767) */
        int16_t *acc16 = (int16_t *) mean_scratch_buf_s3;
        memset(acc16, 0, channels * sizeof(int16_t));

        const int8_t *ptr = input;
        for (int i = 0; i < num_elements; i++) {
            for (int c = 0; c < channels; c++) {
                acc16[c] += (int16_t)ptr[c];
            }
            ptr += channels;
        }

        for (int c = 0; c < channels; c++) {
            int32_t sum = (int32_t)acc16[c] - zp_correction;
            int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[c] = (int8_t)result;
        }
    } else {
        /* int32 accumulation for larger spatial sizes */
        int32_t *acc = (int32_t *) mean_scratch_buf_s3;
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
    }
}
