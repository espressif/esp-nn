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

#include <esp_nn.h>
#include "test_utils.h"

void esp_nn_hard_swish_s8_test()
{
    /* Test with representative MobileNetV3 parameters */
    const int test_sizes[] = {1, 8, 16, 32, 100, 1024, 12544};
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    /* Typical quantization params from MobileNetV3 layers */
    const int16_t input_zp = -128;
    const int16_t output_mult_fxp = 19661;  /* typical value */
    const int16_t reluish_mult_fxp = 22938; /* typical value */
    const int16_t output_zp = -128;

    /* Test all three branches: exp > 0, exp < 0, exp == 0 */
    int32_t reluish_exps[] = {2, -1, 0};
    int32_t output_exps[] = {-1, -2, -1};

    printf("\n######## Running %s ##########\n", __FUNCTION__);

    /* Set up scratch buffer for LUT-based optimization */
    int32_t scratch_size = esp_nn_get_hard_swish_scratch_size();
    void *scratch_buf = NULL;
    if (scratch_size > 0) {
        scratch_buf = malloc(scratch_size);
        if (scratch_buf) {
            esp_nn_set_hard_swish_scratch_buf(scratch_buf);
        }
    }

    for (int t = 0; t < num_tests; t++) {
        int size = test_sizes[t];
        int8_t *input_orig = malloc(size + 16);
        int8_t *out_c_orig = malloc(size + 16);
        int8_t *out_opt_orig = malloc(size + 16);

        if (!input_orig || !out_c_orig || !out_opt_orig) {
            printf(ANSI_COLOR_RED"hard_swish [%d] alloc failed\n"ANSI_COLOR_RESET, t);
            goto cleanup;
        }

        int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
        int8_t *out_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
        int8_t *out_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);

        for (int i = 0; i < size; i++) {
            input[i] = rand() % 256 - 128;
        }

        for (int exp_idx = 0; exp_idx < 3; exp_idx++) {
            /* ANSI C reference */
            profile_c_start();
            esp_nn_hard_swish_s8_ansi(input, out_c, size,
                                       input_zp, output_mult_fxp, reluish_mult_fxp,
                                       reluish_exps[exp_idx], output_exps[exp_idx], output_zp);
            profile_c_end();

            /* Optimized */
            profile_opt_start();
            esp_nn_hard_swish_s8(input, out_opt, size,
                                  input_zp, output_mult_fxp, reluish_mult_fxp,
                                  reluish_exps[exp_idx], output_exps[exp_idx], output_zp);
            profile_opt_end();

            bool ret = CHECK_EQUAL(out_c, out_opt, size);
            if (!ret) {
                printf(ANSI_COLOR_RED"hard_swish [size=%d, exp=%d] failed\n"ANSI_COLOR_RESET,
                       size, (int)reluish_exps[exp_idx]);
                goto cleanup;
            }
        }
        printf(ANSI_COLOR_GREEN"hard_swish [%2d] passed [size %d]\n"ANSI_COLOR_RESET, t, size);

    cleanup:
        if (input_orig) free(input_orig);
        if (out_c_orig) free(out_c_orig);
        if (out_opt_orig) free(out_opt_orig);
    }
    if (scratch_buf) free(scratch_buf);
}
