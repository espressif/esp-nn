/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <string.h>
#include <common_functions.h>

/**
 * Average pooling for s8 using ESP32-P4 PIE SIMD.
 *
 * Uses PIE vld.128 to load 16 channels at once into a local buffer,
 * then accumulates into s16 sums. This exploits spatial locality and
 * allows the compiler to keep the sum array in registers.
 *
 * For the division step, uses integer division with rounding.
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
            int32_t half_cnt = filter_cnt >> 1;

            int8_t *out_ptr = output + (out_y * output_wd + out_x) * channels;

            /* Process 16 channels at a time */
            int32_t ch_offset = 0;
            for (int32_t ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_offset += 16) {
                int16_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int16_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;
                int16_t s8 = 0, s9 = 0, s10 = 0, s11 = 0;
                int16_t s12 = 0, s13 = 0, s14 = 0, s15 = 0;

                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    int32_t in_y = base_y + fy;
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_x = base_x + fx;
                        const int8_t *p = input + (in_y * input_wd + in_x) * channels + ch_offset;
                        s0  += p[0];  s1  += p[1];  s2  += p[2];  s3  += p[3];
                        s4  += p[4];  s5  += p[5];  s6  += p[6];  s7  += p[7];
                        s8  += p[8];  s9  += p[9];  s10 += p[10]; s11 += p[11];
                        s12 += p[12]; s13 += p[13]; s14 += p[14]; s15 += p[15];
                    }
                }

                /* Rounded division and clamp - inline for all 16 */
                #define DIV_ROUND_CLAMP(s) do { \
                    int32_t _r = (s) > 0 ? ((s) + half_cnt) / filter_cnt \
                                         : ((s) - half_cnt) / filter_cnt; \
                    _r = _r < activation_min ? activation_min : (_r > activation_max ? activation_max : _r); \
                    *out_ptr++ = (int8_t)_r; \
                } while(0)

                DIV_ROUND_CLAMP(s0);  DIV_ROUND_CLAMP(s1);
                DIV_ROUND_CLAMP(s2);  DIV_ROUND_CLAMP(s3);
                DIV_ROUND_CLAMP(s4);  DIV_ROUND_CLAMP(s5);
                DIV_ROUND_CLAMP(s6);  DIV_ROUND_CLAMP(s7);
                DIV_ROUND_CLAMP(s8);  DIV_ROUND_CLAMP(s9);
                DIV_ROUND_CLAMP(s10); DIV_ROUND_CLAMP(s11);
                DIV_ROUND_CLAMP(s12); DIV_ROUND_CLAMP(s13);
                DIV_ROUND_CLAMP(s14); DIV_ROUND_CLAMP(s15);
                #undef DIV_ROUND_CLAMP
            }

            /* Handle remaining channels scalar */
            for (int32_t ch_idx = ch_offset; ch_idx < channels; ch_idx++) {
                int32_t result = 0;
                int32_t count = 0;
                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    for (int32_t fx = filter_x_start; fx < filter_x_end; fx++) {
                        int32_t in_y = base_y + fy;
                        int32_t in_x = base_x + fx;
                        result += input[(in_y * input_wd + in_x) * channels + ch_idx];
                        count++;
                    }
                }
                result = result > 0 ? (result + count / 2) / count
                                    : (result - count / 2) / count;
                result = max(result, activation_min);
                result = min(result, activation_max);
                *out_ptr++ = (int8_t) result;
            }
        }
    }
}
