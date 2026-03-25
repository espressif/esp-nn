/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <limits.h>
#include <common_functions.h>

/**
 * Max pooling for s8 using ESP32-P4 PIE SIMD.
 * Vectorizes the channel dimension: processes 16 channels per iteration
 * using esp.vmax.s8 to find running maximum across the filter window.
 */
void esp_nn_max_pool_s8_esp32p4(const int8_t *input,
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
    /* Enable PIE */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    /* Broadcast activation_min and activation_max into vectors */
    int8_t act_min_val = (int8_t) activation_min;
    int8_t act_max_val = (int8_t) activation_max;
    int8_t int8_min_val = INT8_MIN;

    asm volatile (
        "mv              x30, %0     \n\t"
        "esp.vldbc.8.ip  q4, x30, 0 \n\t"  /* q4 = broadcast(activation_min) */
        "mv              x30, %1     \n\t"
        "esp.vldbc.8.ip  q5, x30, 0 \n\t"  /* q5 = broadcast(activation_max) */
        "mv              x30, %2     \n\t"
        "esp.vldbc.8.ip  q6, x30, 0 \n\t"  /* q6 = broadcast(INT8_MIN) for init */
        :: "r"(&act_min_val), "r"(&act_max_val), "r"(&int8_min_val)
        : "x30"
    );

    const int32_t ch_16 = channels >> 4;  /* number of full 16-ch blocks */

    int32_t base_y = -pad_ht;
    for (int32_t out_y = 0; out_y < output_ht; out_y++, base_y += stride_ht) {
        int32_t base_x = -pad_wd;
        for (int32_t out_x = 0; out_x < output_wd; out_x++, base_x += stride_wd) {
            int32_t filter_y_start = max(0, -base_y);
            int32_t filter_x_start = max(0, -base_x);
            int32_t filter_y_end = min(filter_ht, input_ht - base_y);
            int32_t filter_x_end = min(filter_wd, input_wd - base_x);

            int8_t *out_ptr = output + (out_y * output_wd + out_x) * channels;

            /* Process channels in blocks of 16 */
            int32_t ch_offset = 0;
            for (int32_t ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_offset += 16) {
                /* Initialize running max to INT8_MIN (copy q6 -> q0) */
                asm volatile ("esp.vmax.s8 q0, q6, q6 \n\t");

                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_y = base_y + fy;
                        int32_t in_x = base_x + fx;
                        const int8_t *in_ptr = input + (in_y * input_wd + in_x) * channels + ch_offset;

                        asm volatile (
                            "mv              x30, %0     \n\t"
                            "esp.vld.128.ip  q1, x30, 0  \n\t"  /* load 16 channels */
                            "esp.vmax.s8     q0, q0, q1  \n\t"  /* running max */
                            :
                            : "r"(in_ptr)
                            : "x30"
                        );
                    }
                }

                /* Apply activation: max(act_min, min(act_max, result)) and store */
                {
                    int8_t *store_ptr = out_ptr + ch_offset;
                    asm volatile (
                        "esp.vmax.s8     q0, q0, q4       \n\t"  /* max(result, act_min) */
                        "esp.vmin.s8     q0, q0, q5       \n\t"  /* min(result, act_max) */
                        "mv              x30, %0          \n\t"
                        "esp.vst.128.ip  q0, x30, 0       \n\t"  /* store 16 channels */
                        :
                        : "r"(store_ptr)
                        : "x30", "memory"
                    );
                }
            }

            /* Handle remaining channels scalar */
            for (int32_t ch_idx = ch_offset; ch_idx < channels; ch_idx++) {
                int8_t result = INT8_MIN;
                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_y = base_y + fy;
                        int32_t in_x = base_x + fx;
                        int32_t input_index = (in_y * input_wd + in_x) * channels + ch_idx;
                        result = max(input[input_index], result);
                    }
                }
                result = max(result, (int8_t) activation_min);
                result = min(result, (int8_t) activation_max);
                out_ptr[ch_idx] = result;
            }
        }
    }
}
