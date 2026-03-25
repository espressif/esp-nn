/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include <esp_nn.h>
#include "test_utils.h"

static void run_relu6_test(int size, int iter)
{
    int8_t *input = NULL, *inout_ansi = NULL, *inout_opt = NULL;

    int8_t *input_orig = malloc(size + 16);
    int8_t *inout_c_orig = malloc(size + 16);
    int8_t *inout_opt_orig = malloc(size + 16);

    if (input_orig == NULL || inout_c_orig == NULL || inout_opt_orig == NULL) {
        printf(ANSI_COLOR_RED"relu6 [%d] allocations failed\n"ANSI_COLOR_RESET, iter);
        goto relu6_cleanup;
    }
    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    inout_ansi = (int8_t *) (((uint32_t) inout_c_orig + 15) & ~15);
    inout_opt = (int8_t *) (((uint32_t) inout_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 255 - 128;
        inout_ansi[i] = input[i];
        inout_opt[i] = input[i];
    }

    profile_c_start();
    esp_nn_relu6_s8_ansi(inout_ansi, size);
    profile_c_end();

    profile_opt_start();
    esp_nn_relu6_s8(inout_opt, size);
    profile_opt_end();

    bool ret = CHECK_EQUAL(inout_ansi, inout_opt, size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"relu6 [%d] failed [size %d]\n"ANSI_COLOR_RESET, iter, size);
        goto relu6_cleanup;
    }
    printf(ANSI_COLOR_GREEN"relu6 [%2d] passed [size %d]\n"ANSI_COLOR_RESET, iter, size);

relu6_cleanup:
    if (input_orig) free(input_orig);
    if (inout_c_orig) free(inout_c_orig);
    if (inout_opt_orig) free(inout_opt_orig);
}

void esp_nn_relu6_s8_test()
{
    int iter = 0;
    /* Original test case: odd size with leftover */
    run_relu6_test(1600 + 8 + 7, iter++);
    /* Very small sizes (< 8 elements, below SIMD width) */
    run_relu6_test(1, iter++);
    run_relu6_test(3, iter++);
    run_relu6_test(7, iter++);
    /* Between 8 and 16 (partial SIMD) */
    run_relu6_test(8, iter++);
    run_relu6_test(12, iter++);
    run_relu6_test(15, iter++);
    /* Exact multiple of 16 (full SIMD, no leftover) */
    run_relu6_test(16, iter++);
    run_relu6_test(32, iter++);
    run_relu6_test(256, iter++);
    /* Non-aligned sizes */
    run_relu6_test(17, iter++);
    run_relu6_test(33, iter++);
    run_relu6_test(100, iter++);
}
