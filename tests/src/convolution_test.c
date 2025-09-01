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
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"

void esp_nn_depthwise_conv_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    int8_t *input = NULL, *filter_data = NULL;
    int8_t *out_data_c = NULL, *out_data_opt = NULL;
    int32_t *bias = NULL;
    int32_t input_offset = 5; /* some number in [-128, 127] */
    int32_t out_offset = 7;
    int32_t activation_min = -125;
    int32_t activation_max = 120;
    void *scratch_buf = NULL;

    /* independent variables */
    int input_wd, input_ht, channels;
    uint16_t filter_ht, filter_wd, ch_mult, out_wd, out_ht;
    uint16_t pad_wd, pad_ht, stride_wd, stride_ht;

    printf("\n######## Running %s ##########\n", __FUNCTION__);
    // run for 15 iterations
    for (int itr = 0; itr < 15; itr++) {
        /* prepare data */
        switch (itr) {
        case 0: // (ch_mult 1, (channels % 16) = 0), filter (3,3), pad (0,0)
            input_wd = 18;
            input_ht = 18;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 16;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 1: // (ch_mult 1, (channels % 16) = 0), filter (3,3), pad (1,1)
            input_wd = 10;
            input_ht = 10;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 16;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 2: // (ch_mult 1, (channels % 8) = 0), filter (3,3), pad (1,1)
            input_wd = 10;
            input_ht = 10;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 24;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 3: // other filter sizes (ch_mult 1, (channels % 8) = 0)
            input_wd = 10;
            input_ht = 10;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 24;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 4: // other filter sizes (ch_mult 8 = 0)
            input_wd = 6;
            input_ht = 6;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 8;
            channels = 4;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 5: // other filter sizes (ch_mult 8 = 0)
            input_wd = 12;
            input_ht = 12;
            filter_ht = 5;
            filter_wd = 5;
            ch_mult = 8;
            channels = 4;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 6: // other filter sizes (ch_mult 4 = 0)
            input_wd = 6;
            input_ht = 6;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 4;
            channels = 4;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 7: // (ch_mult 1, (channels % 16) = 0), filter (3,3), pad (0,0)  stride (2,2)
            input_wd = 6;
            input_ht = 6;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 16;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 8: // same as case 7, with large parameters
            input_wd = 58;
            input_ht = 58;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 128;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 9: // (ch_mult 1, (channels % 16) = 0), filter (3,3), pad (0,0)  stride (2,2)
            input_wd = 6;
            input_ht = 6;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 16;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        default:
            input_wd = 6;
            input_ht = 6;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 16;
            stride_wd = rand() % 2 + 1;
            stride_ht = stride_wd;
            pad_wd = stride_wd == 1 ? 0 : rand() % 2;
            pad_ht = pad_wd;
            break;
        }

        /* prepare data */
        if (pad_wd) {
            out_wd = (input_wd + stride_wd - 1) / stride_wd;
        } else {
            out_wd = (input_wd + stride_wd - filter_wd) / stride_wd;
        }
        if (pad_ht) {
            out_ht = (input_ht + stride_ht - 1) / stride_ht;
        } else {
            out_ht = (input_ht + stride_ht - filter_ht) / stride_ht;
        }

        // if (itr == 9) {
            // expect the function to handle this gracefully
            // out_wd += 1;
            // out_ht += 1;
        // }
        int in_size = input_wd * input_ht * channels;
        int out_size = out_wd * out_ht * channels * ch_mult;
        int filter_size = filter_wd * filter_ht * channels * ch_mult + 4;
        int bias_size = channels * ch_mult + 1;
        int32_t out_shift[channels * ch_mult];
        int32_t out_mult[channels * ch_mult];

        int8_t *input_orig = ESP_NN_TEST_ALLOC(in_size + 16);
        int8_t *out_c_orig = ESP_NN_TEST_ALLOC(out_size + 16);
        int8_t *out_opt_orig = ESP_NN_TEST_ALLOC(out_size + 16);
        filter_data = ESP_NN_TEST_ALLOC(filter_size);
        bias = ESP_NN_TEST_ALLOC(bias_size * 4);

        if (bias == NULL || input_orig == NULL || filter_data == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL) {
            printf(ANSI_COLOR_RED"[%d] allocations failed\n"ANSI_COLOR_RESET, itr);
            goto dc_s8_cleanup;
        }

        input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
        out_data_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
        out_data_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

        /* Generate input data */
        for (int i = 0; i < in_size; ++i) {
            input[i] = rand() % 128;
        }

        /* Generate filter data */
        for (int i = 0; i < filter_size; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }

        /* Generate bias data */
        for (int i = 0; i < channels * ch_mult; ++i) {
            bias[i + 1] = rand() % INT16_MAX; //0th index left for unalignment
            out_shift[i] = -8 + rand() % 3;
            out_mult[i] = 0x7eb0e200 + rand() % 50;
        }

        data_dims_t input_dims = {.width = input_wd, .height = input_ht, .channels = channels, 1};
        data_dims_t output_dims = {.width = out_wd, .height = out_ht, .channels = channels * ch_mult, 1};
        data_dims_t filter_dims = {.width = filter_wd, .height = filter_ht, 0, 0};
        dw_conv_params_t conv_params = {.in_offset = input_offset, .out_offset = out_offset, .ch_mult = ch_mult,
                                        .stride = {stride_wd, stride_ht}, .padding = {pad_wd, pad_ht},
                                        .dilation = {0, 0}, .activation = {activation_min, activation_max}};
        quant_data_t quant_data = {.shift = out_shift, .mult = out_mult};

        int scratch_buf_size = esp_nn_get_depthwise_conv_scratch_size(&input_dims, &filter_dims,
                                                                      &output_dims, &conv_params);
        if (scratch_buf_size > 0) {
            scratch_buf = ESP_NN_TEST_ALLOC(scratch_buf_size + 16);
            if (scratch_buf == NULL) {
                printf(ANSI_COLOR_RED"[%d] scratch_buf alloc failed size %d\n"ANSI_COLOR_RESET,
                       itr, scratch_buf_size);
                goto dc_s8_cleanup;
            }
            int align_sz = 16 - (((int32_t) scratch_buf) & 0xf);
            esp_nn_set_depthwise_conv_scratch_buf(scratch_buf + align_sz);
        }

        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_depthwise_conv_s8_ansi(&input_dims, input, &filter_dims, filter_data + 4,
                                      bias + 1, &output_dims, out_data_c, &conv_params, &quant_data);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_depthwise_conv_s8(&input_dims, input, &filter_dims, filter_data + 4,
                                 bias + 1, &output_dims, out_data_opt, &conv_params, &quant_data);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(out_data_c, out_data_opt, out_size);
        if (ret == false) {
        printf(ANSI_COLOR_RED"[%3d] failed [pad: (%d, %d), stride: (%d, %d)"
               " out: (%3d,%3d), filter: (%d, %d,%3d), ch_mult %d]\n"ANSI_COLOR_RESET,
               itr, pad_wd, pad_ht, stride_wd, stride_ht, out_wd, out_ht,
               filter_wd, filter_ht, channels, ch_mult);
#if 0
            printf("Output: \n");
            PRINT_ARRAY_HEX(out_data_opt, out_size / out_ht, out_ht);
            printf("Expected: \n");
            PRINT_ARRAY_HEX(out_data_c, out_size / out_ht, out_ht);
            printf("Input:\n");
            PRINT_ARRAY_HEX(input, in_size / input_ht, input_ht);
            printf("Filter data:\n");
            PRINT_ARRAY_HEX(filter_data + 4, (filter_size - 4) / filter_ht, filter_ht);
            printf("bias data:\n");
            PRINT_ARRAY_INT(bias + 1, ch_mult * channels, 1);
#endif
            goto dc_s8_cleanup;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [pad: (%d, %d), stride: (%d, %d)"
               " out: (%3d,%3d), filter: (%d, %d,%3d), ch_mult %d]"ANSI_COLOR_RESET,
               itr, pad_wd, pad_ht, stride_wd, stride_ht, out_wd,
               out_ht, filter_wd, filter_ht, channels, ch_mult);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);

    dc_s8_cleanup:
        if (input_orig) {
            free(input_orig);
        }
        if (filter_data) {
            free(filter_data);
        }
        if (out_c_orig) {
            free(out_c_orig);
        }
        if (out_opt_orig) {
            free(out_opt_orig);
        }
        if (bias) {
            free(bias);
        }
        if (scratch_buf) {
            free(scratch_buf);
        }
    }
}

void esp_nn_conv_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    const int32_t input_offset = 5; /* some number in [-128, 127] */
    const int32_t activation_min = -125;
    const int32_t activation_max = 122;
    const int32_t out_offset = 3;

    void *scratch_buf = NULL;
    int8_t *input_orig = NULL;
    int8_t *out_c_orig = NULL;
    int8_t *out_opt_orig = NULL;
    int8_t *filter_data = NULL;
    int32_t *bias = NULL;

    /* independent variable */
    int in_wd, in_ht, in_channels, out_channels;
    uint16_t filter_ht, filter_wd, out_wd, out_ht;
    uint16_t pad_wd, pad_ht, stride_wd, stride_ht;

    printf("\n######## Running %s ##########\n", __FUNCTION__);
    // run for 10 iterations
    for (int itr = 0; itr < 15; itr++) {
        switch (itr) {
        case 0: // ch % 8 == 0 && filter (1,1), padding (0,0)
            in_wd = 10;
            in_ht = 10;
            in_channels = 64;
            out_channels = 64;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 1: // ch % 4 == 0 && (in_wd * in_ht) % 16 == 0
            in_wd = 4;
            in_ht = 4;
            in_channels = 20;
            out_channels = 8;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 2: // ch, filter (3x3x3)
            in_wd = 10;
            in_ht = 10;
            in_channels = 3;
            out_channels = 64;
            filter_ht = 3;
            filter_wd = 3;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 3: // remaining pad (0, 0)
            in_wd = 10;
            in_ht = 10;
            in_channels = 3;
            out_channels = 64;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 4: // unopt case
            in_wd = 10;
            in_ht = 10;
            in_channels = 12;
            out_channels = 64;
            filter_ht = 3;
            filter_wd = 3;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 5: // ch % 8 == 0 & stride (2,2)
            in_wd = 16;
            in_ht = 16;
            in_channels = 16;
            out_channels = 16;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 6: // ch % 8 == 0 && filter (1,1), padding (0,0)
            in_wd = 2;
            in_ht = 2;
            in_channels = 8;
            out_channels = 8;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 7: // ch == 3, pad (0, 0)
            in_wd = 112;
            in_ht = 112;
            in_channels = 3;
            out_channels = 16;
            filter_ht = 6;
            filter_wd = 6;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 8: // ch == 5, remaining pad (0, 0)
            in_wd = 8;
            in_ht = 8;
            in_channels = 5;
            out_channels = 16;
            filter_ht = 6;
            filter_wd = 6;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 9: //
            in_wd = 3;
            in_ht = 3;
            in_channels = 32;
            out_channels = 1;
            filter_ht = 3;
            filter_wd = 3;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 10: // needs right and bottom padding
            in_wd = 4;
            in_ht = 8;
            in_channels = 1;
            out_channels = 3;
            filter_ht = 3;
            filter_wd = 3;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        case 11: // needs right and bottom padding
            in_wd = 4;
            in_ht = 8;
            in_channels = 3;
            out_channels = 4;
            filter_ht = 3;
            filter_wd = 3;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 2;
            stride_ht = 2;
            break;
        default: // ch % 8 == 0
            in_wd = 8;
            in_ht = 8;
            in_channels = 16;
            out_channels = 16;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            break;
        }

        /* prepare data */
        if (pad_wd) {
            out_wd = (in_wd + stride_wd - 1) / stride_wd;
        } else {
            out_wd = (in_wd + stride_wd - filter_wd) / stride_wd;
        }
        if (pad_ht) {
            out_ht = (in_ht + stride_ht - 1) / stride_ht;
        } else {
            out_ht = (in_ht + stride_ht - filter_ht) / stride_ht;
        }

        int in_size = in_wd * in_ht * in_channels;
        int filter_size = filter_wd * filter_ht * in_channels * out_channels + 2;
        int out_size = out_wd * out_ht * out_channels;

        input_orig = ESP_NN_TEST_ALLOC(in_size + 16);
        out_c_orig = ESP_NN_TEST_ALLOC(out_size + 16);
        out_opt_orig = ESP_NN_TEST_ALLOC(out_size + 16);
        filter_data = ESP_NN_TEST_ALLOC(filter_size + 16);
        bias = ESP_NN_TEST_ALLOC(128 + sizeof (int32_t) * out_channels);

        int32_t *out_shift = ESP_NN_TEST_ALLOC(128 + sizeof (int32_t) * out_channels);
        int32_t *out_mult = ESP_NN_TEST_ALLOC(128 + sizeof (int32_t) * out_channels);

        if (input_orig == NULL || filter_data == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL) {
            printf(ANSI_COLOR_RED"input/filter/out_data/bias allocations failed\n"ANSI_COLOR_RESET);
            goto conv_s8_cleanup;
        }

        if (bias == NULL || out_shift == NULL || out_mult == NULL) {
            printf(ANSI_COLOR_RED"bias/out_shift/out_mult allocations failed\n"ANSI_COLOR_RESET);
            goto conv_s8_cleanup;
        }

        int8_t *input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
        int8_t *out_data_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
        int8_t *out_data_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

        /* Generate input data between -128 -> +127 */
        for (int i = 0; i < in_size; ++i) {
            input[i] = rand() % 255 - 128;
        }

        /* Generate filter data between -128 -> +127 */
        for (int i = 0; i < filter_size; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }

        /* Generate bias data */
        for (int i = 0; i < out_channels; ++i) {
            bias[i] = (int32_t)rand() % UINT16_MAX + UINT8_MAX;
        }

        /* Shift and multiplier */
        for (int i = 0; i < out_channels; ++i) {
            out_shift[i] = -10 + rand() % 2;
            out_mult[i] = 0x7f67f4f8 + rand() % 50;
        }

        data_dims_t input_dims = {.width = in_wd, .height = in_ht, .channels = in_channels, 1};
        data_dims_t output_dims = {.width = out_wd, .height = out_ht, .channels = out_channels, 1};
        data_dims_t filter_dims = {.width = filter_wd, .height = filter_ht, 0, 0};
        conv_params_t conv_params = {.in_offset = input_offset, .out_offset = out_offset,
                                    .stride = {stride_wd, stride_ht}, .padding = {pad_wd, pad_ht},
                                    .dilation = {0, 0}, .activation = {activation_min, activation_max}};
        quant_data_t quant_data = {.shift = out_shift, .mult = out_mult};

        int scratch_buf_size = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                                            &output_dims, &conv_params);
        if (scratch_buf_size > 0) {
#if IDF_HEAP_CAPS
            void *scratch_buf = heap_caps_malloc(scratch_buf_size + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
            void *scratch_buf = malloc(scratch_buf_size + 16);
#endif
            if (scratch_buf == NULL) {
                printf(ANSI_COLOR_RED"scratch_buf alloc failed size %d\n"ANSI_COLOR_RESET, scratch_buf_size);
                goto conv_s8_cleanup;
            }
            int align_sz = 16 - (((int32_t) scratch_buf) & 0xf);
            esp_nn_set_conv_scratch_buf(scratch_buf + align_sz);
        }

        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_conv_s8_ansi(&input_dims, input, &filter_dims, filter_data,
                            bias, &output_dims, out_data_c, &conv_params, &quant_data);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_conv_s8(&input_dims, input, &filter_dims, filter_data,
                       bias, &output_dims, out_data_opt, &conv_params, &quant_data);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(out_data_c, out_data_opt, out_size);
        if (ret == false) {
            printf(ANSI_COLOR_RED"[%3d] failed [pad: (%d, %d), stride: (%d, %d)"
                   " out: (%3d,%3d,%3d), filter: (%d, %d,%3d)]\n"ANSI_COLOR_RESET,
                   itr, pad_wd, pad_ht, stride_wd, stride_ht, out_wd, out_ht,
                   out_channels, filter_wd, filter_ht, in_channels);
#if 0
            printf("Output: \n");
            PRINT_ARRAY_INT8(out_data_opt, out_size / out_ht, out_ht);
            printf("Expected: \n");
            PRINT_ARRAY_INT8(out_data_c, out_size / out_ht, out_ht);
            printf("Input:\n");
            PRINT_ARRAY_INT8(input, in_size / in_ht, in_ht);
            printf("Filter data:\n");
            PRINT_ARRAY_INT8(filter_data, (filter_size - 2) / filter_ht, filter_ht);
            printf("bias data:\n");
            PRINT_ARRAY_INT(bias, out_channels, 1);
#endif
            goto conv_s8_cleanup;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [pad: (%d, %d), stride: (%d, %d)"
               " out: (%3d,%3d,%3d), filter: (%d, %d,%3d)]"ANSI_COLOR_RESET,
               itr, pad_wd, pad_ht, stride_wd, stride_ht, out_wd, out_ht,
               out_channels, filter_wd, filter_ht, in_channels);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);

    conv_s8_cleanup:
        if (input_orig) {
            free(input_orig);
        }
        if (filter_data) {
            free(filter_data);
        }
        if (out_c_orig) {
            free(out_c_orig);
        }
        if (out_opt_orig) {
            free(out_opt_orig);
        }
        if (bias) {
            free(bias);
        }
        if (out_shift) {
            free(out_shift);
        }
        if (out_mult) {
            free(out_mult);
        }
        if (scratch_buf) {
            free(scratch_buf);
        }
    }
}
