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
        /* Phase 1: Find max in row using PIE vectorization.
         * Use auto-incrementing loads to avoid redundant mv per iteration. */
        int8_t max_in_row;
        if (width >= 16) {
            int32_t vec_count = (width >> 4);  /* number of 16-element groups */
            int32_t vec_processed = vec_count << 4;

            int32_t max_scalar;
            asm volatile (
                "mv     x30, %[ptr]              \n\t"
                "esp.vld.128.ip q0, x30, 16      \n\t"  /* load first 16, advance */
                "addi   %[cnt], %[cnt], -1       \n\t"  /* one group already loaded */
                "beqz   %[cnt], 2f               \n\t"
                "1:                              \n\t"
                "esp.vld.128.ip q1, x30, 16      \n\t"  /* load next 16, advance */
                "esp.vmax.s8    q0, q0, q1       \n\t"  /* running max */
                "addi   %[cnt], %[cnt], -1       \n\t"
                "bnez   %[cnt], 1b               \n\t"
                "2:                              \n\t"
                "esp.max.s8.a   q0, %[max]       \n\t"  /* horizontal reduce */
                : [cnt] "+r"(vec_count), [max] "=r"(max_scalar)
                : [ptr] "r"(in_ptr)
                : "x30"
            );
            max_in_row = (int8_t) max_scalar;

            /* Check remaining elements (< 16) */
            for (int32_t i = vec_processed; i < width; i++) {
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
