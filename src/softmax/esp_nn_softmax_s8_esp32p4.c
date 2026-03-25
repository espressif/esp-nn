/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "softmax_common.h"
#include <stdio.h>
#include <limits.h>

static int32_t *p4_scratch_buf = NULL;

int32_t esp_nn_get_softmax_scratch_size_esp32p4(const int32_t width, const int32_t height)
{
    (void) height;
    return width * 4;
}

void esp_nn_set_softmax_scratch_buf_esp32p4(void *buffer)
{
    /* Enable PIE */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );
    p4_scratch_buf = (int32_t *) buffer;
}

/**
 * Softmax for s8 optimized for ESP32-P4.
 * Phase 1 (find-max) uses PIE esp.vmax.s8 for 16 elements at a time.
 * Phases 2-3 (exp + normalize) use cached exp values in scratch buffer.
 */
void esp_nn_softmax_s8_esp32p4(const int8_t *input_data,
                                const int32_t height,
                                const int32_t width,
                                const int32_t mult,
                                const int32_t shift,
                                const int32_t diff_min,
                                int8_t *output_data)
{
    if (p4_scratch_buf == NULL) {
        printf("%s error! scratch buffer not set\n", __FUNCTION__);
        return;
    }

#define ACCUM_BITS  12
#define DIFF_BITS   5

    const int32_t mask = (1 << shift);
    int32_t col = 0;
    const int8_t *in_ptr = input_data;
    int8_t *out_ptr = output_data;

    for (int row_idx = 0; row_idx < height; row_idx++) {
        /* Phase 1: Find max in row using PIE vectorization */
        int8_t max_in_row;
        if (width >= 16) {
            /* Load first 16 elements as running max */
            asm volatile (
                "mv   x30, %0           \n\t"
                "esp.vld.128.ip q0, x30, 0 \n\t"
                :: "r"(in_ptr) : "x30"
            );

            int32_t i = 16;
            for (; i <= width - 16; i += 16) {
                asm volatile (
                    "mv   x30, %0            \n\t"
                    "esp.vld.128.ip q1, x30, 0 \n\t"
                    "esp.vmax.s8    q0, q0, q1 \n\t"
                    :: "r"(in_ptr + i) : "x30"
                );
            }

            /* Reduce q0 to scalar max */
            /* Extract all 16 bytes and find max scalar */
            int8_t tmp[16] __attribute__((aligned(16)));
            asm volatile (
                "mv   x30, %0             \n\t"
                "esp.vst.128.ip q0, x30, 0 \n\t"
                :: "r"(tmp) : "x30", "memory"
            );

            max_in_row = tmp[0];
            for (int j = 1; j < 16; j++) {
                if (tmp[j] > max_in_row) max_in_row = tmp[j];
            }
            /* Check remaining elements */
            for (; i < width; i++) {
                if (in_ptr[i] > max_in_row) max_in_row = in_ptr[i];
            }
        } else {
            max_in_row = in_ptr[0];
            for (col = 1; col < width; col++) {
                max_in_row = max(max_in_row, in_ptr[col]);
            }
        }

        /* Phase 2: Compute exp values and sum */
        int32_t input_diff = 0;
        int32_t sum_of_exps = 0;

        for (col = 0; col < width; col++) {
            input_diff = in_ptr[col] - max_in_row;
            if (input_diff >= diff_min) {
                const int32_t input_diff_rescaled = SAT_HIGH_MUL(input_diff * mask, mult);
                const int32_t exp_raw = esp_nn_exp_on_negative_values(input_diff_rescaled);
                p4_scratch_buf[col] = exp_raw;
                sum_of_exps += DIV_POW2(exp_raw, ACCUM_BITS);
            }
        }

        /* Phase 3: Normalize */
        const int32_t headroom_plus1 = esp_nn_clz32((uint32_t) sum_of_exps);
        const int32_t shifted_scale = ONE_OVER_ONE_X((sum_of_exps << headroom_plus1) - (1 << 31));
        const int32_t bits_over_unit = ACCUM_BITS - headroom_plus1 + 31 - sizeof(int8_t) * 8;

        for (col = 0; col < width; col++) {
            input_diff = in_ptr[col] - max_in_row;
            if (input_diff >= diff_min) {
                int32_t exp_raw = p4_scratch_buf[col];
                const int32_t shifted_output = SAT_HIGH_MUL(shifted_scale, exp_raw);
                const int32_t result = DIV_POW2(shifted_output, bits_over_unit) - 128;
                out_ptr[col] = (int8_t) esp_nn_saturate8(result);
            } else {
                out_ptr[col] = -128;
            }
        }
        in_ptr  += width;
        out_ptr += width;
    }
}
