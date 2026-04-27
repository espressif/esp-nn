/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * Elementwise multiply for s8 optimized for ESP32-P4.
 * Uses inlined fast requantization with 4x unrolled loop.
 * Interleaves independent computations to hide latency.
 */
void esp_nn_mul_elementwise_s8_esp32p4(const int8_t *input1_data,
                                        const int8_t *input2_data,
                                        const int32_t input1_offset,
                                        const int32_t input2_offset,
                                        int8_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const int32_t size)
{
    const int32_t left_shift = out_shift > 0 ? out_shift : 0;
    const int32_t right_shift = left_shift - out_shift;
    const int64_t nudge = (int64_t)1 << 30;

    int i = 0;
    for (; i <= size - 4; i += 4) {
        int32_t prod0 = (input1_data[i+0] + input1_offset) * (input2_data[i+0] + input2_offset);
        int32_t prod1 = (input1_data[i+1] + input1_offset) * (input2_data[i+1] + input2_offset);
        int32_t prod2 = (input1_data[i+2] + input1_offset) * (input2_data[i+2] + input2_offset);
        int32_t prod3 = (input1_data[i+3] + input1_offset) * (input2_data[i+3] + input2_offset);

        int32_t s0 = prod0 << left_shift;
        int32_t s1 = prod1 << left_shift;
        int32_t s2 = prod2 << left_shift;
        int32_t s3 = prod3 << left_shift;

        int32_t r0 = (int32_t)(((int64_t)s0 * out_mult + nudge) >> 31);
        int32_t r1 = (int32_t)(((int64_t)s1 * out_mult + nudge) >> 31);
        int32_t r2 = (int32_t)(((int64_t)s2 * out_mult + nudge) >> 31);
        int32_t r3 = (int32_t)(((int64_t)s3 * out_mult + nudge) >> 31);

        if (right_shift > 0) {
            int32_t rnd = (1 << (right_shift - 1));
            r0 = (r0 + rnd - (r0 < 0)) >> right_shift;
            r1 = (r1 + rnd - (r1 < 0)) >> right_shift;
            r2 = (r2 + rnd - (r2 < 0)) >> right_shift;
            r3 = (r3 + rnd - (r3 < 0)) >> right_shift;
        }

        r0 = max(activation_min, min(r0 + out_offset, activation_max));
        r1 = max(activation_min, min(r1 + out_offset, activation_max));
        r2 = max(activation_min, min(r2 + out_offset, activation_max));
        r3 = max(activation_min, min(r3 + out_offset, activation_max));

        output[i+0] = (int8_t) r0;
        output[i+1] = (int8_t) r1;
        output[i+2] = (int8_t) r2;
        output[i+3] = (int8_t) r3;
    }

    for (; i < size; i++) {
        int32_t prod = (input1_data[i] + input1_offset) * (input2_data[i] + input2_offset);
        int32_t out = esp_nn_requantize(prod, out_mult, out_shift);
        out = max(activation_min, min(out + out_offset, activation_max));
        output[i] = (int8_t) out;
    }
}
