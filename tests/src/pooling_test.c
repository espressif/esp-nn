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
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"

static void run_avg_pool_test(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                              uint16_t filter_wd, uint16_t filter_ht,
                              uint16_t stride_wd, uint16_t stride_ht,
                              uint16_t pad_wd, uint16_t pad_ht,
                              int iter)
{
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const uint16_t out_wd = (input_wd + 2 * pad_wd - filter_wd) / stride_wd + 1;
    const uint16_t out_ht = (input_ht + 2 * pad_ht - filter_ht) / stride_ht + 1;
    const int size = input_wd * input_ht * channels;
    const int out_size = out_wd * out_ht * channels;

    int8_t *input = NULL, *output_c = NULL, *output_opt = NULL;
    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(out_size + 16);
    int8_t *out_opt_orig = malloc(out_size + 16);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        printf(ANSI_COLOR_RED"avg_pool [%d] allocations failed\n"ANSI_COLOR_RESET, iter);
        goto avg_pool_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    output_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    output_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
    }

    profile_c_start();
    esp_nn_avg_pool_s8_ansi(input, input_wd, input_ht, output_c, out_wd, out_ht,
                            stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                            activation_min, activation_max, channels);
    profile_c_end();

    profile_opt_start();
    esp_nn_avg_pool_s8(input, input_wd, input_ht, output_opt, out_wd, out_ht,
                       stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                       activation_min, activation_max, channels);
    profile_opt_end();

    bool ret = CHECK_EQUAL(output_c, output_opt, out_size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"avg_pool [%d] failed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d]\n"ANSI_COLOR_RESET,
               iter, input_wd, input_ht, channels, filter_wd, filter_ht,
               stride_wd, stride_ht, pad_wd, pad_ht);
        goto avg_pool_cleanup;
    }
    printf(ANSI_COLOR_GREEN"avg_pool [%2d] passed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d]\n"ANSI_COLOR_RESET,
           iter, input_wd, input_ht, channels, filter_wd, filter_ht,
           stride_wd, stride_ht, pad_wd, pad_ht);

avg_pool_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
}

void esp_nn_avg_pool_s8_test()
{
    int iter = 0;
    /* Original test case */
    run_avg_pool_test(16, 16, 16, 3, 3, 1, 1, 1, 1, iter++);
    /* Varying channel counts */
    run_avg_pool_test(16, 16, 4, 3, 3, 1, 1, 1, 1, iter++);
    run_avg_pool_test(16, 16, 8, 3, 3, 1, 1, 1, 1, iter++);
    run_avg_pool_test(16, 16, 32, 3, 3, 1, 1, 1, 1, iter++);
    run_avg_pool_test(16, 16, 64, 3, 3, 1, 1, 1, 1, iter++);
    /* Note: non-multiple-of-4 channels not supported by S3 optimized path */
    /* Different filter sizes */
    run_avg_pool_test(16, 16, 16, 1, 1, 1, 1, 0, 0, iter++);
    run_avg_pool_test(16, 16, 16, 2, 2, 1, 1, 0, 0, iter++);
    run_avg_pool_test(16, 16, 16, 5, 5, 1, 1, 2, 2, iter++);
    /* Stride > 1 */
    run_avg_pool_test(16, 16, 16, 3, 3, 2, 2, 1, 1, iter++);
    run_avg_pool_test(24, 24, 32, 3, 3, 2, 2, 1, 1, iter++);
    /* Person detection final pooling: 6x6x128, filter 6x6 */
    run_avg_pool_test(6, 6, 128, 6, 6, 1, 1, 0, 0, iter++);
    /* No padding */
    run_avg_pool_test(16, 16, 16, 3, 3, 1, 1, 0, 0, iter++);
}

static void run_max_pool_test(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                              uint16_t filter_wd, uint16_t filter_ht,
                              uint16_t stride_wd, uint16_t stride_ht,
                              uint16_t pad_wd, uint16_t pad_ht,
                              int iter)
{
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const uint16_t out_wd = (input_wd + 2 * pad_wd - filter_wd) / stride_wd + 1;
    const uint16_t out_ht = (input_ht + 2 * pad_ht - filter_ht) / stride_ht + 1;
    const int size = input_wd * input_ht * channels;
    const int out_size = out_wd * out_ht * channels;

    int8_t *input = NULL, *output_c = NULL, *output_opt = NULL;
    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(out_size + 16);
    int8_t *out_opt_orig = malloc(out_size + 16);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        printf(ANSI_COLOR_RED"max_pool [%d] allocations failed\n"ANSI_COLOR_RESET, iter);
        goto max_pool_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    output_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    output_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
    }

    profile_c_start();
    esp_nn_max_pool_s8_ansi(input, input_wd, input_ht, output_c, out_wd, out_ht,
                            stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                            activation_min, activation_max, channels);
    profile_c_end();

    profile_opt_start();
    esp_nn_max_pool_s8(input, input_wd, input_ht, output_opt, out_wd, out_ht,
                       stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                       activation_min, activation_max, channels);
    profile_opt_end();

    bool ret = CHECK_EQUAL(output_c, output_opt, out_size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"max_pool [%d] failed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d]\n"ANSI_COLOR_RESET,
               iter, input_wd, input_ht, channels, filter_wd, filter_ht,
               stride_wd, stride_ht, pad_wd, pad_ht);
        goto max_pool_cleanup;
    }
    printf(ANSI_COLOR_GREEN"max_pool [%2d] passed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d]\n"ANSI_COLOR_RESET,
           iter, input_wd, input_ht, channels, filter_wd, filter_ht,
           stride_wd, stride_ht, pad_wd, pad_ht);

max_pool_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
}

void esp_nn_max_pool_s8_test()
{
    int iter = 0;
    /* Original test case */
    run_max_pool_test(16, 16, 16, 3, 3, 1, 1, 1, 1, iter++);
    /* Varying channel counts */
    run_max_pool_test(16, 16, 4, 3, 3, 1, 1, 1, 1, iter++);
    run_max_pool_test(16, 16, 8, 3, 3, 1, 1, 1, 1, iter++);
    run_max_pool_test(16, 16, 32, 3, 3, 1, 1, 1, 1, iter++);
    run_max_pool_test(16, 16, 64, 3, 3, 1, 1, 1, 1, iter++);
    /* Note: non-multiple-of-4 channels not supported by S3 optimized path */
    /* Different filter sizes */
    run_max_pool_test(16, 16, 16, 1, 1, 1, 1, 0, 0, iter++);
    run_max_pool_test(16, 16, 16, 2, 2, 1, 1, 0, 0, iter++);
    run_max_pool_test(16, 16, 16, 5, 5, 1, 1, 2, 2, iter++);
    /* Stride > 1 */
    run_max_pool_test(16, 16, 16, 3, 3, 2, 2, 1, 1, iter++);
    run_max_pool_test(24, 24, 32, 3, 3, 2, 2, 1, 1, iter++);
    /* Person detection final pooling-like: 6x6x128 */
    run_max_pool_test(6, 6, 128, 6, 6, 1, 1, 0, 0, iter++);
    /* No padding */
    run_max_pool_test(16, 16, 16, 3, 3, 1, 1, 0, 0, iter++);
}
