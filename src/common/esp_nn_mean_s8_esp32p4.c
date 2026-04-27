/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-P4 optimized spatial mean reduction using QACC per-lane accumulation.
 * Processes 16 channels in parallel via esp.vmulas.s8.qacc (same pattern as avg_pool).
 */

#include <stdint.h>
#include <common_functions.h>

void esp_nn_mean_nhwc_s8_esp32p4(const int8_t *input,
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
    const int32_t ch_16 = channels >> 4;

    const int8_t one_val = 1;
    if (ch_16 > 0) {
        /* Enable PIE and broadcast 1 into q7 */
        asm volatile (
            "csrsi  0x7f2, 0b01        \n\t"
            "li     x29, 0b10          \n\t"
            "esp.movx.w.cfg x29        \n\t"
            ::: "x29"
        );
        asm volatile (
            "mv     x30, %0             \n\t"
            "esp.vldbc.8.ip q7, x30, 0  \n\t"
            :: "r"(&one_val) : "x30"
        );
    }

    /* Process all channels - QACC for 16-channel blocks, scalar for remainder */
    int ch = 0;
    for (int ch_blk = 0; ch_blk < ch_16; ch_blk++, ch += 16) {
        /* Single asm block: broadcast ones, zero QACC, accumulate all spatial
         * positions. Keeping in one block prevents compiler from clobbering
         * q7 between the broadcast and the MAC loop. */
        const int8_t *base_ptr = input + ch;
        asm volatile (
            /* Broadcast 1 into q7 */
            "mv     x30, %[one]             \n\t"
            "esp.vldbc.8.ip q7, x30, 0      \n\t"
            /* Zero QACC */
            "esp.zero.qacc                   \n\t"
            /* Accumulate loop: stride = channels between spatial positions */
            "mv     x30, %[base]            \n\t"
            "mv     s7,  %[cnt]             \n\t"
            "1:                             \n\t"
            "esp.vld.128.ip  q0, x30, 0     \n\t"
            "esp.vmulas.s8.qacc q0, q7      \n\t"
            "add    x30, x30, %[stride]     \n\t"
            "addi   s7, s7, -1              \n\t"
            "bnez   s7, 1b                  \n\t"
            :
            : [one] "r"(&one_val), [base] "r"(base_ptr),
              [cnt] "r"(num_elements), [stride] "r"((int32_t)channels)
            : "x30", "s7"
        );

        int32_t sums[16] __attribute__((aligned(16)));
        ESP_NN_QACC_EXTRACT_S32(sums);

        int32_t zp_correction = num_elements * input_zero_point;
        for (int k = 0; k < 16; k++) {
            int32_t result = sums[k] - zp_correction;
            result = esp_nn_multiply_by_quantized_mult(result, multiplier, shift);
            result += output_zero_point;
            result = max(result, -128);
            result = min(result, 127);
            output[ch + k] = (int8_t)result;
        }
    }

    /* Remaining channels scalar */
    for (; ch < channels; ch++) {
        int32_t sum = 0;
        for (int hw = 0; hw < num_elements; hw++) {
            sum += input[hw * channels + ch];
        }
        sum -= num_elements * input_zero_point;
        int32_t result = esp_nn_multiply_by_quantized_mult(sum, multiplier, shift);
        result += output_zero_point;
        result = max(result, -128);
        result = min(result, 127);
        output[ch] = (int8_t)result;
    }
}
