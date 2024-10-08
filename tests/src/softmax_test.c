/*
 * SPDX-FileCopyrightText: 2022-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"

void esp_nn_softmax_s8_test()
{
    const int32_t height = 8;
    const int32_t width = 32;
    const int32_t diff_min = -128;
    const int32_t mult = INT32_MAX / 2;
    const int32_t shift = 7;
    void *scratch_buf = NULL, *scratch_buf_orig = NULL;
    const int size = width * height;
    int8_t *input, *out_ansi, *out_opt;

    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(size + 16);
    int8_t *out_opt_orig = malloc(size + 16);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        printf(ANSI_COLOR_RED"%s buffer allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto softmax_s8_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    out_ansi = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    out_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    /* Generate input data between -128 -> +127 */
    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 255 - 128;
    }

    /* enable profiler */
    profile_c_start();

    /* C function */
    esp_nn_softmax_s8_ansi(input, height, width, mult, shift, diff_min, out_ansi);

    profile_c_end();

    int32_t scratch_buf_size = esp_nn_get_softmax_scratch_size(width, height);
    if (scratch_buf_size) {
        scratch_buf_orig = malloc(scratch_buf_size * 4 + 16);
        scratch_buf = 16 + scratch_buf_orig - ((uint32_t) scratch_buf_orig & 0xf);
        if (scratch_buf == NULL) {
            printf(ANSI_COLOR_RED"%s scratch_buf alloc failed size %"PRIi32"\n"ANSI_COLOR_RESET,
                   __FUNCTION__, scratch_buf_size);
            goto softmax_s8_cleanup;
        }
        esp_nn_set_softmax_scratch_buf(scratch_buf);
    }

    profile_opt_start();

    /* Optimized function */
    esp_nn_softmax_s8(input, height, width, mult, shift, diff_min, out_opt);

    /* disable profiler */
    profile_opt_end();

    bool ret = CHECK_EQUAL(out_ansi, out_opt, size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"%s failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        printf("Output: \n");
        PRINT_ARRAY_HEX(out_opt, width, height);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(out_ansi, width, height);
        printf("Input:\n");
        PRINT_ARRAY_HEX(input, width, height);
        goto softmax_s8_cleanup;
    }
    printf(ANSI_COLOR_GREEN"%s passed\n"ANSI_COLOR_RESET, __FUNCTION__);

softmax_s8_cleanup:
    if (input_orig) {
        free (input_orig);
    }
    if (out_c_orig) {
        free (out_c_orig);
    }
    if (out_opt_orig) {
        free (out_opt_orig);
    }
    if (scratch_buf_orig) {
        free (scratch_buf_orig);
    }
}
