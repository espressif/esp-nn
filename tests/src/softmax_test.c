/*
 * SPDX-FileCopyrightText: 2022-2026 Espressif Systems (Shanghai) CO LTD
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

static void run_softmax_test(int32_t height, int32_t width, int32_t mult,
                             int32_t shift, int32_t diff_min, int iter)
{
    void *scratch_buf = NULL, *scratch_buf_orig = NULL;
    const int size = width * height;
    int8_t *input = NULL, *out_ansi = NULL, *out_opt = NULL;

    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(size + 16);
    int8_t *out_opt_orig = malloc(size + 16);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        printf(ANSI_COLOR_RED"softmax [%d] allocations failed\n"ANSI_COLOR_RESET, iter);
        goto softmax_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    out_ansi = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    out_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 255 - 128;
    }

    profile_c_start();
    esp_nn_softmax_s8_ansi(input, height, width, mult, shift, diff_min, out_ansi);
    profile_c_end();

    int32_t scratch_buf_size = esp_nn_get_softmax_scratch_size(width, height);
    if (scratch_buf_size) {
        scratch_buf_orig = malloc(scratch_buf_size * 4 + 16);
        if (scratch_buf_orig == NULL) {
            printf(ANSI_COLOR_RED"softmax [%d] scratch alloc failed size %"PRIi32"\n"ANSI_COLOR_RESET,
                   iter, scratch_buf_size);
            goto softmax_cleanup;
        }
        scratch_buf = (void *)(((uint32_t) scratch_buf_orig + 15) & ~15);
        esp_nn_set_softmax_scratch_buf(scratch_buf);
    }

    profile_opt_start();
    esp_nn_softmax_s8(input, height, width, mult, shift, diff_min, out_opt);
    profile_opt_end();

    bool ret = CHECK_EQUAL(out_ansi, out_opt, size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"softmax [%d] failed [h %"PRIi32", w %"PRIi32", mult %"PRIi32", shift %"PRIi32", diff_min %"PRIi32"]\n"ANSI_COLOR_RESET,
               iter, height, width, mult, shift, diff_min);
        printf("Output: \n");
        PRINT_ARRAY_HEX(out_opt, width, height);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(out_ansi, width, height);
        goto softmax_cleanup;
    }
    printf(ANSI_COLOR_GREEN"softmax [%2d] passed [h %"PRIi32", w %"PRIi32", mult %"PRIi32", shift %"PRIi32"]\n"ANSI_COLOR_RESET,
           iter, height, width, mult, shift);

softmax_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
    if (scratch_buf_orig) free(scratch_buf_orig);
}

void esp_nn_softmax_s8_test()
{
    int iter = 0;
    /* Original test case */
    run_softmax_test(8, 32, INT32_MAX / 2, 7, -128, iter++);
    /* Small output classes (person_detection: 2, micro_speech: 4) */
    run_softmax_test(1, 2, INT32_MAX / 2, 7, -128, iter++);
    run_softmax_test(1, 4, INT32_MAX / 2, 7, -128, iter++);
    /* Single element (degenerate) */
    run_softmax_test(1, 1, INT32_MAX / 2, 7, -128, iter++);
    /* Medium width */
    run_softmax_test(1, 10, INT32_MAX / 2, 7, -128, iter++);
    run_softmax_test(4, 10, INT32_MAX / 2, 7, -128, iter++);
    /* Large width (ImageNet-class) */
    run_softmax_test(1, 1000, INT32_MAX / 2, 7, -128, iter++);
    /* Large height */
    run_softmax_test(64, 32, INT32_MAX / 2, 7, -128, iter++);
    /* Varying diff_min */
    run_softmax_test(8, 32, INT32_MAX / 2, 7, -64, iter++);
    run_softmax_test(8, 32, INT32_MAX / 2, 7, -32, iter++);
    run_softmax_test(8, 32, INT32_MAX / 2, 7, 0, iter++);
    /* Varying multiplier and shift */
    run_softmax_test(8, 32, INT32_MAX / 4, 5, -128, iter++);
    run_softmax_test(8, 32, INT32_MAX, 10, -128, iter++);
    /* Odd width (non-aligned) */
    run_softmax_test(8, 17, INT32_MAX / 2, 7, -128, iter++);
    run_softmax_test(8, 3, INT32_MAX / 2, 7, -128, iter++);
}
