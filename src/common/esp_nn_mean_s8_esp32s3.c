/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 mean reduction over the spatial dimensions of an NHWC int8 tensor.
 * Four channels at a time, each with its own register accumulator, so one
 * pass over the positions serves four channels. The rest go one at a time.
 */

#include <stdint.h>
#include <common_functions.h>

int32_t esp_nn_get_mean_scratch_size_esp32s3(const int32_t height, const int32_t width,
                                             const int32_t channels)
{
    (void) height;
    (void) width;
    (void) channels;
    return 0;
}

void esp_nn_set_mean_scratch_buf_esp32s3(void *buffer)
{
    (void) buffer;
}

static inline int8_t mean_requant(int32_t sum, const int32_t multiplier, const int32_t shift,
                                  const int32_t output_zero_point)
{
    int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
    result += output_zero_point;
    result = max(result, -128);
    result = min(result, 127);
    return (int8_t) result;
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
    int32_t c = 0;

    for (; c + 4 <= channels; c += 4) {
        int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const int8_t *ptr = input + c;
        for (int32_t i = 0; i < num_elements; i++) {
            s0 += ptr[0];
            s1 += ptr[1];
            s2 += ptr[2];
            s3 += ptr[3];
            ptr += channels;
        }
        output[c + 0] = mean_requant(s0 - zp_correction, multiplier, shift, output_zero_point);
        output[c + 1] = mean_requant(s1 - zp_correction, multiplier, shift, output_zero_point);
        output[c + 2] = mean_requant(s2 - zp_correction, multiplier, shift, output_zero_point);
        output[c + 3] = mean_requant(s3 - zp_correction, multiplier, shift, output_zero_point);
    }
    for (; c < channels; c++) {
        int32_t sum = 0;
        const int8_t *ptr = input + c;
        for (int32_t i = 0; i < num_elements; i++) {
            sum += *ptr;
            ptr += channels;
        }
        output[c] = mean_requant(sum - zp_correction, multiplier, shift, output_zero_point);
    }
}
