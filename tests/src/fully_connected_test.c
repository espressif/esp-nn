/*
 * SPDX-FileCopyrightText: 2020-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"


void esp_nn_fully_connected_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    /* prepare data */
    uint16_t row_len = 256 + 8 + 7; /* odd len to test unaligned+left-over */
    uint16_t out_channels = 3;
    int8_t input[row_len];
    int8_t filter_data[row_len * out_channels];
    int8_t output_c[out_channels], output_opt[out_channels];
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int32_t input_offset = 0;
    int32_t filter_offset = 0;
    int32_t out_shift = -10;
    int32_t out_offset = 5;
    int32_t out_mult = 0x59e492c4;
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0; itr < 15; itr++) {
        out_mult = INT32_MAX / row_len + rand() % INT16_MAX;
        switch (itr) {
        case 0:
            out_shift = -10;
            break;
        case 1:
            out_shift = SHIFT_MIN;
            break;
        case 2:
            out_shift = SHIFT_MAX;
            break;
        case 3:
            out_shift = 0;
            break;
        case 4:
            row_len = 1;
            out_channels = 16;
            out_shift = -10 + rand() % 5;
            break;
        case 5:
            row_len = 16;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 6:
            row_len = 8;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 7:
            row_len = 8;
            out_channels = 15;
            out_shift = -10 + rand() % 5;
            break;
        case 8:
            row_len = 8;
            out_channels = 1;
            out_shift = -10 + rand() % 5;
            break;
        default:
            row_len = rand() % 7 + 1;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        }
        if (itr == 0) {
            out_shift = SHIFT_MAX;
        }
        /* Generate input and filter data */
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }

        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter_data, filter_offset,
                                    NULL, output_c, out_channels, out_offset, out_shift, out_mult,
                                    activation_min, activation_max);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter_data, filter_offset,
                                NULL, output_opt, out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(output_c, output_opt, out_channels);
        if (ret == false) {
            printf(ANSI_COLOR_RED"[%3d] failed\n"ANSI_COLOR_RESET, itr);
#if 0
            printf("Output: \n");
            PRINT_ARRAY_HEX(output_opt, out_channels, 1);
            printf("Expected: \n");
            PRINT_ARRAY_HEX(output_c, out_channels, 1);
            printf("Input:\n");
            PRINT_ARRAY_HEX(input, row_len, 1);
            printf("Filter data:\n");
            PRINT_ARRAY_HEX(filter_data, row_len, out_channels);
            printf("Out shift: %d\n", out_shift);
            printf("Out mult: %x\n", out_mult);
#endif
            return;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [row_len %"PRIu16", out_ch %"PRIu16"]"ANSI_COLOR_RESET,
               itr, row_len, out_channels);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);
    }
}
