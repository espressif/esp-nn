/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
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

void esp_nn_mean_nhwc_s8_test()
{
    /* Test dimensions matching MobileNetV3 SE blocks */
    struct {
        int height, width, channels;
    } test_cases[] = {
        {7, 7, 16},      /* small SE block */
        {7, 7, 72},      /* medium SE block */
        {14, 14, 40},    /* larger spatial */
        {14, 14, 120},   /* larger channels */
        {28, 28, 24},    /* early layer SE */
        {1, 1, 576},     /* degenerate 1x1 */
        {3, 3, 96},      /* small spatial */
    };
    const int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    const int32_t input_zp = -128;
    const int32_t output_zp = -128;
    const int32_t multiplier = 1073741824; /* typical */
    const int32_t shift = -1;

    printf("\n######## Running %s ##########\n", __FUNCTION__);

    for (int t = 0; t < num_tests; t++) {
        int h = test_cases[t].height;
        int w = test_cases[t].width;
        int c = test_cases[t].channels;
        int input_size = h * w * c;

        int8_t *input_orig = malloc(input_size + 16);
        int8_t *out_c_orig = malloc(c + 16);
        int8_t *out_opt_orig = malloc(c + 16);

        if (!input_orig || !out_c_orig || !out_opt_orig) {
            printf(ANSI_COLOR_RED"mean [%d] alloc failed\n"ANSI_COLOR_RESET, t);
            goto cleanup;
        }

        int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
        int8_t *out_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
        int8_t *out_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);

        for (int i = 0; i < input_size; i++) {
            input[i] = rand() % 256 - 128;
        }

        /* ANSI C reference */
        profile_c_start();
        esp_nn_mean_nhwc_s8_ansi(input, out_c, h, w, c,
                                  input_zp, output_zp, multiplier, shift);
        profile_c_end();

        /* Optimized */
        profile_opt_start();
        esp_nn_mean_nhwc_s8(input, out_opt, h, w, c,
                             input_zp, output_zp, multiplier, shift);
        profile_opt_end();

        bool ret = CHECK_EQUAL(out_c, out_opt, c);
        if (!ret) {
            printf(ANSI_COLOR_RED"mean [%d] failed [%dx%dx%d]\n"ANSI_COLOR_RESET,
                   t, h, w, c);
            goto cleanup;
        }
        printf(ANSI_COLOR_GREEN"mean [%2d] passed [%dx%dx%d]\n"ANSI_COLOR_RESET,
               t, h, w, c);

    cleanup:
        if (input_orig) free(input_orig);
        if (out_c_orig) free(out_c_orig);
        if (out_opt_orig) free(out_opt_orig);
    }
}
