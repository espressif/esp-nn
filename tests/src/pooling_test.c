/*
 * SPDX-FileCopyrightText: 2020-2023 Espressif Systems (Shanghai) CO LTD
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


void esp_nn_avg_pool_s8_test()
{
    /* prepare data */
    const uint16_t input_wd = 16;
    const uint16_t input_ht = 16;
    const uint16_t channels = 16; /* With TFLite example, I have seen it 256 */
    const int size = input_wd * input_ht * channels;
    int8_t *input, *output_c, *output_opt;
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const uint16_t pad_wd = 1;
    const uint16_t pad_ht = 1;
    const uint16_t stride_wd = 1;
    const uint16_t stride_ht = 1;
    const uint16_t filter_ht = 3;
    const uint16_t filter_wd = 3;
    const uint16_t out_wd = input_wd / stride_wd;
    const uint16_t out_ht = input_ht / stride_ht;
    const int out_size = out_wd * out_ht * channels;

    int8_t *input_orig = malloc(size + 32);
    int8_t *out_c_orig = malloc(out_size + 32);
    int8_t *out_opt_orig = malloc(out_size + 32);
    input = 16 + input_orig - ((uint32_t) input_orig & 0xf);
    output_c = 16 + out_c_orig - ((uint32_t) out_c_orig & 0xf);
    output_opt = 16 + out_opt_orig - ((uint32_t) out_opt_orig & 0xf);

    if (input == NULL || output_c == NULL || output_opt == NULL) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto avg_pool_s8_cleanup;
    }
    /**
     * width/height, channels etc look suspicious but it it true.
     * It actually depends upon where in model this is actually placed.
     * If at the end wd/ht tends to be smaller and depth larger.
     */

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
    }

    /* enable profiler */
    profile_c_start();

    /* C function */
    esp_nn_avg_pool_s8_ansi(input, input_wd, input_ht, output_c, out_wd, out_ht,
                              stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                              activation_min, activation_max, channels);

    profile_c_end();
    profile_opt_start();

    /* Optimized function */
    esp_nn_avg_pool_s8(input, input_wd, input_ht, output_opt, out_wd, out_ht,
                         stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                         activation_min, activation_max, channels);

    /* disable profiler */
    profile_opt_end();


    bool ret = CHECK_EQUAL(output_c, output_opt, out_size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"%s failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        printf("Output: \n");
        PRINT_ARRAY_HEX(output_opt, out_wd * channels, out_ht);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(output_c, out_wd * channels, out_ht);
        printf("Input:\n");
        PRINT_ARRAY_HEX(input, input_wd * channels, input_ht);
        goto avg_pool_s8_cleanup;
    }
    printf(ANSI_COLOR_GREEN"%s passed\n"ANSI_COLOR_RESET, __FUNCTION__);

avg_pool_s8_cleanup:
    if (input) {
        free(input_orig);
    }
    if (output_c) {
        free(out_c_orig);
    }
    if (output_opt) {
        free(out_opt_orig);
    }
}

void esp_nn_max_pool_s8_test()
{
    /* prepare data */
    const uint16_t input_wd = 16;
    const uint16_t input_ht = 16;
    const uint16_t channels = 16; /* With TFLite example, I have seen it 256 */
    int8_t *input, *output_c, *output_opt;
    const int size = input_wd * input_ht * channels;
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const uint16_t pad_wd = 1;
    const uint16_t pad_ht = 1;
    const uint16_t stride_wd = 1;
    const uint16_t stride_ht = 1;
    const uint16_t filter_ht = 3;
    const uint16_t filter_wd = 3;
    const uint16_t out_wd = input_wd / stride_wd;
    const uint16_t out_ht = input_ht / stride_ht;
    const int out_size = out_wd * out_ht * channels;

    int8_t *input_orig = malloc(size + 32);
    int8_t *out_c_orig = malloc(out_size + 32);
    int8_t *out_opt_orig = malloc(out_size + 32);
    input = 16 + input_orig - ((uint32_t) input_orig & 0xf);
    output_c = 16 + out_c_orig - ((uint32_t) out_c_orig & 0xf);
    output_opt = 16 + out_opt_orig - ((uint32_t) out_opt_orig & 0xf);

    if (input == NULL || output_c == NULL || output_opt == NULL) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto max_pool_s8_cleanup;
    }

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
    }

    /* enable profiler */
    profile_c_start();

    /* C function */
    esp_nn_max_pool_s8_ansi(input, input_wd, input_ht, output_c, out_wd, out_ht,
                            stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                            activation_min, activation_max, channels);

    profile_c_end();
    profile_opt_start();

    /* Optimized function */
    esp_nn_max_pool_s8(input, input_wd, input_ht, output_opt, out_wd, out_ht,
                       stride_wd, stride_ht, filter_wd, filter_ht, pad_wd, pad_ht,
                       activation_min, activation_max, channels);

    /* disable profiler */
    profile_opt_end();


    bool ret = CHECK_EQUAL(output_c, output_opt, out_wd * out_ht * channels);
    if (ret == false) {
        printf(ANSI_COLOR_RED"%s failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        printf("Output: \n");
        PRINT_ARRAY_HEX(output_opt, out_wd * out_ht * channels, 1);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(output_c, out_wd * out_ht * channels, 1);
        printf("Input:\n");
        PRINT_ARRAY_HEX(input, 8, size / 8);
        goto max_pool_s8_cleanup;
    }
    printf(ANSI_COLOR_GREEN"%s passed\n"ANSI_COLOR_RESET, __FUNCTION__);

max_pool_s8_cleanup:
    if (input) {
        free(input_orig);
    }
    if (output_c) {
        free(out_c_orig);
    }
    if (output_opt) {
        free(out_opt_orig);
    }
}
