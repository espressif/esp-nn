/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized softmax with SIMD find-max for width >= 16.
 */

#include <stdint.h>
#include "softmax_common.h"

static int32_t *scratch_buf_s3 = NULL;

int32_t esp_nn_get_softmax_scratch_size_esp32s3(const int32_t width, const int32_t height)
{
    (void) height;
    return width * 4;
}

void esp_nn_set_softmax_scratch_buf_esp32s3(void *buffer)
{
    scratch_buf_s3 = (int32_t *) buffer;
}

/* Find max of int8 array — SIMD for len >= 32, scalar for smaller */
static inline int8_t find_max_s8(const int8_t *data, int32_t len)
{
    int8_t m = -128;
    int32_t idx = 0;

#if defined(__XTENSA__)
    if (len >= 32) {
        /* Use ee.vmax.s8 for 16 elements/cycle — only for len >= 32
         * to avoid potential alignment issues with small buffers */
        int8_t tmp_buf[16] __attribute__((aligned(16)));
        const int8_t *ptr = data;
        int8_t *buf_ptr = tmp_buf;
        int32_t simd_len = len & ~15; /* round down to multiple of 16 */

        asm volatile (
            "ee.vld.128.ip  q0, %[ptr], 16          \n\t" /* q0 = running max */
            "movi.n %[idx], 16                       \n\t"
            "j      2f                               \n\t"
            "1:                                      \n\t"
            "ee.vld.128.ip  q1, %[ptr], 16           \n\t"
            "ee.vmax.s8     q0, q0, q1               \n\t"
            "addi   %[idx], %[idx], 16               \n\t"
            "2:                                      \n\t"
            "blt    %[idx], %[slen], 1b              \n\t"
            /* Store vector max to tmp_buf for horizontal reduction */
            "ee.vst.128.ip  q0, %[buf], 16           \n\t"
            : [idx] "+r"(idx), [ptr] "+r"(ptr), [buf] "+r"(buf_ptr)
            : [slen] "r"(simd_len)
            : "memory"
        );

        /* Horizontal reduction of 16 max values */
        for (int i = 0; i < 16; i++) {
            if (tmp_buf[i] > m) m = tmp_buf[i];
        }
        idx = simd_len;
    }
#endif

    /* Scalar for remainder or small arrays */
    for (; idx < len; idx++) {
        if (data[idx] > m) m = data[idx];
    }
    return m;
}

void esp_nn_softmax_s8_esp32s3(const int8_t *input_data,
                                const int32_t height,
                                const int32_t width,
                                const int32_t mult,
                                const int32_t shift,
                                const int32_t diff_min,
                                int8_t *output_data)
{
    if (scratch_buf_s3 == NULL) {
        /* Fall through to opt version if scratch not set */
        return;
    }

#define ACCUM_BITS  12

    const int32_t mask = (1 << shift);
    const int8_t *in_ptr = input_data;
    int8_t *out_ptr = output_data;

    for (int row_idx = 0; row_idx < height; row_idx++) {
        /* Phase 1: Find max */
        int8_t max_in_row = find_max_s8(in_ptr, width);

        /* Phase 2: Compute exp and accumulate sum */
        int32_t sum_of_exps = 0;
        for (int col = 0; col < width; col++) {
            int32_t input_diff = in_ptr[col] - max_in_row;
            if (input_diff >= diff_min) {
                const int32_t input_diff_rescaled = SAT_HIGH_MUL(input_diff * mask, mult);
                const int32_t exp_raw = esp_nn_exp_on_negative_values(input_diff_rescaled);
                scratch_buf_s3[col] = exp_raw;
                sum_of_exps += DIV_POW2(exp_raw, ACCUM_BITS);
            }
        }

        /* Phase 3: Compute normalization scale */
        const int32_t headroom_plus1 = esp_nn_clz32((uint32_t) sum_of_exps);
        const int32_t shifted_scale = ONE_OVER_ONE_X((sum_of_exps << headroom_plus1) - (1 << 31));
        const int32_t bits_over_unit = ACCUM_BITS - headroom_plus1 + 31 - 8;

        /* Phase 4: Normalize and output — unrolled 4x for reduced loop overhead */
        int col = 0;
        for (; col + 3 < width; col += 4) {
            for (int k = 0; k < 4; k++) {
                int32_t input_diff = in_ptr[col + k] - max_in_row;
                if (input_diff >= diff_min) {
                    int32_t exp_raw = scratch_buf_s3[col + k];
                    const int32_t shifted_output = SAT_HIGH_MUL(shifted_scale, exp_raw);
                    const int32_t result = DIV_POW2(shifted_output, bits_over_unit) - 128;
                    out_ptr[col + k] = (int8_t) esp_nn_saturate8(result);
                } else {
                    out_ptr[col + k] = -128;
                }
            }
        }
        /* Remainder */
        for (; col < width; col++) {
            int32_t input_diff = in_ptr[col] - max_in_row;
            if (input_diff >= diff_min) {
                int32_t exp_raw = scratch_buf_s3[col];
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
#undef ACCUM_BITS
}
