/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * Average pooling for s8 using ESP32-P4 PIE SIMD.
 *
 * Uses QACC per-lane accumulation: multiply 16 input channels by a
 * vector of 1s, accumulate per-lane across filter window.
 * Extract 16 × int32 sums via esp.st.qacc.{l,h}.{l,h}.128.ip.
 * Then divide, clamp, and store.
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
    /* Enable PIE */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    /* Broadcast 1 into q7 for "multiply by 1" accumulation trick */
    const int8_t one_val = 1;
    asm volatile (
        "mv     x30, %0             \n\t"
        "esp.vldbc.8.ip q7, x30, 0  \n\t"
        :: "r"(&one_val) : "x30"
    );

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

            /* Process 16 channels at a time using QACC per-lane accumulation */
            int32_t ch_offset = 0;
            for (int32_t ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_offset += 16) {

                /* Clear per-lane accumulators */
                asm volatile ("esp.zero.qacc \n\t");

                /* Accumulate via QACC with stride-based fx loop */
                for (int32_t fy = filter_y_start; fy < filter_y_end; fy++) {
                    int32_t in_y = base_y + fy;
                    const int8_t *row_ptr = input + (in_y * input_wd + base_x + filter_x_start) * channels + ch_offset;
                    int32_t fx_count = filter_x_end - filter_x_start;

                    asm volatile (
                        "mv     x30, %[ptr]              \n\t"
                        "mv     s7,  %[cnt]              \n\t"
                        "1:                              \n\t"
                        "esp.vld.128.ip  q0, x30, 0      \n\t"
                        "esp.vmulas.s8.qacc q0, q7       \n\t"
                        "add    x30, x30, %[stride]      \n\t"
                        "addi   s7, s7, -1               \n\t"
                        "bnez   s7, 1b                   \n\t"
                        :
                        : [ptr] "r"(row_ptr), [cnt] "r"(fx_count),
                          [stride] "r"((int32_t)channels)
                        : "x30", "s7"
                    );
                }

                /* Extract 16 per-lane int32 sums from QACC:
                 * qacc has 4 quadrants, each 128 bits = 4 × int32 */
                int32_t sums[16] __attribute__((aligned(16)));
                asm volatile (
                    "mv                      x30, %0     \n\t"
                    "esp.st.qacc.l.l.128.ip  x30, 16     \n\t"  /* lanes 0-3 */
                    "esp.st.qacc.l.h.128.ip  x30, 16     \n\t"  /* lanes 4-7 */
                    "esp.st.qacc.h.l.128.ip  x30, 16     \n\t"  /* lanes 8-11 */
                    "esp.st.qacc.h.h.128.ip  x30, 0      \n\t"  /* lanes 12-15 */
                    :: "r"(sums)
                    : "x30", "memory"
                );

                /* Rounded division and activation clamp */
                for (int k = 0; k < 16; k++) {
                    int32_t s = sums[k];
                    int32_t result = s > 0 ? (s + half_cnt) / filter_cnt
                                           : (s - half_cnt) / filter_cnt;
                    result = max(result, activation_min);
                    result = min(result, activation_max);
                    out_ptr[ch_offset + k] = (int8_t) result;
                }
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
                out_ptr[ch_idx] = (int8_t) result;
            }
        }
    }
}
