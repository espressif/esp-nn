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

void esp_nn_relu6_s8_test()
{
    const int size = 1600 + 8 + 7;
    int8_t *input, *inout_ansi, *inout_opt;

    int8_t *input_orig = malloc(size + 32);
    int8_t *inout_c_orig = malloc(size + 32);
    int8_t *inout_opt_orig = malloc(size + 32);
    input = 16 + input_orig - ((uint32_t) input_orig & 0xf);
    inout_ansi = 16 + inout_c_orig - ((uint32_t) inout_c_orig & 0xf);
    inout_opt = 16 + inout_opt_orig - ((uint32_t) inout_opt_orig & 0xf);

    if (input == NULL || inout_ansi == NULL || inout_opt == NULL) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto relu6_s8_cleanup;
    }
    /* Generate filter data between -128 -> +127 */
    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 255 - 128;
        inout_ansi[i] = input[i];
        inout_opt[i] = input[i];
    }

    /* enable profiler */
    profile_c_start();

    /* C function */
    esp_nn_relu6_s8_ansi(inout_ansi, size);

    profile_c_end();
    profile_opt_start();

    /* Optimized function */
    esp_nn_relu6_s8(inout_opt, size);

    /* disable profiler */
    profile_opt_end();

    bool ret = CHECK_EQUAL(inout_ansi, inout_opt, size);
    if (ret == false) {
        printf(ANSI_COLOR_RED"%s failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        printf("Output: \n");
        PRINT_ARRAY_HEX(inout_opt, size, 1);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(inout_ansi, size, 1);
        printf("Input:\n");
        PRINT_ARRAY_HEX(input, size, 1);
        goto relu6_s8_cleanup;
    }
    printf(ANSI_COLOR_GREEN"%s passed\n"ANSI_COLOR_RESET, __FUNCTION__);

relu6_s8_cleanup:
    if (input) {
        free (input_orig);
    }
    if (inout_ansi) {
        free (inout_c_orig);
    }
    if (inout_opt) {
        free (inout_opt_orig);
    }
}
