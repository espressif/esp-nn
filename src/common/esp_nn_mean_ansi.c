/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Quantized mean reduction over spatial dimensions (axes 1,2).
 * Specialized for 4D tensors [N, H, W, C] → [N, 1, 1, C].
 * This is the common case in Squeeze-and-Excite blocks.
 */

#include <stdint.h>
#include <common_functions.h>

void esp_nn_mean_nhwc_s8_ansi(const int8_t *input,
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

    for (int c = 0; c < channels; c++) {
        /* Sum over spatial dimensions */
        int32_t sum = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                sum += input[(h * width + w) * channels + c];
            }
        }

        /* Apply zero point correction */
        sum -= num_elements * input_zero_point;

        /* Requantize: multiply_by_quantized_mult(sum, multiplier, shift) */
        int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
        result += output_zero_point;
        result = max(result, -128);
        result = min(result, 127);
        output[c] = (int8_t)result;
    }
}
