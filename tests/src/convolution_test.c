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
    // run for 17 iterations
    for (int itr = 0; itr < 17; itr++) {
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
        case 8: // same as case 7, with large parameters (reduced for non-PSRAM boards)
            input_wd = 28;
            input_ht = 28;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 64;
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
        case 15: // ch=8, 3x3, pad=1 (person_detection model layer, ch<12 path)
            input_wd = 48;
            input_ht = 48;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 8;
            pad_wd = 1;
            pad_ht = 1;
            stride_wd = 1;
            stride_ht = 1;
            break;
        case 16: // ch=8, 3x3, pad=0, stride=2 (another ch<12 variant)
            input_wd = 12;
            input_ht = 12;
            filter_ht = 3;
            filter_wd = 3;
            ch_mult = 1;
            channels = 8;
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
    int32_t input_offset = 5; /* some number in [-128, 127] */
    int32_t activation_min = -125;
    int32_t activation_max = 122;
    int32_t out_offset = 3;

    void *scratch_buf = NULL;
    int8_t *input_orig = NULL;
    int8_t *out_c_orig = NULL;
    int8_t *out_opt_orig = NULL;
    int8_t *filter_data = NULL;
    int32_t *bias = NULL;
    int32_t *out_shift = NULL;
    int32_t *out_mult = NULL;

    /* independent variable */
    int in_wd, in_ht, in_channels, out_channels;
    uint16_t filter_ht, filter_wd, out_wd, out_ht;
    uint16_t pad_wd, pad_ht, stride_wd, stride_ht;

    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0; itr < 18; itr++) {
        /* Reset quant params to defaults each iteration */
        input_offset = 5;
        out_offset = 3;
        activation_min = -125;
        activation_max = 122;

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
        case 15: // 1x1 conv, large spatial, YOLO-like quant params
            in_wd = 48;
            in_ht = 48;
            in_channels = 32;
            out_channels = 32;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            // Override quant params to match YOLO Op[8]
            input_offset = 127;
            out_offset = 39;
            break;
        case 16: // 1x1, YOLO exact data: 48x48x32->32 with real filter/bias/quant
            in_wd = 48;
            in_ht = 48;
            in_channels = 32;
            out_channels = 32;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            input_offset = 127;
            out_offset = 39;
            activation_min = -128;
            activation_max = 127;
            break;
        case 17: // 1x1 conv with DELIBERATELY UNALIGNED filter + small out_shift
            // Tests both alignment (filter+5) AND transpose correctness (shift=-6 won't mask 8x error)
            in_wd = 24;
            in_ht = 24;
            in_channels = 32;
            out_channels = 32;
            filter_ht = 1;
            filter_wd = 1;
            pad_wd = 0;
            pad_ht = 0;
            stride_wd = 1;
            stride_ht = 1;
            input_offset = 110; /* typical YOLO value that exposed the bug */
            out_offset = 39;
            activation_min = -128;
            activation_max = 127;
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

        int8_t *filter_data_orig_save = NULL; /* for case 17 unaligned filter restore */

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
        out_shift = ESP_NN_TEST_ALLOC(128 + sizeof (int32_t) * out_channels);
        out_mult = ESP_NN_TEST_ALLOC(128 + sizeof (int32_t) * out_channels);

        if (input_orig == NULL || filter_data == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL ||
                bias == NULL || out_shift == NULL || out_mult == NULL) {
            printf(ANSI_COLOR_RED"[%3d] alloc failed (in=%d filter=%d out=%d)\n"ANSI_COLOR_RESET,
                   itr, in_size, filter_size, out_size);
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

        /* Case 17: deliberately misalign filter by 5 bytes to test alignment handling.
         * This reproduces the bug where ee.vld.l.64.ip ignores lower address bits. */
        filter_data_orig_save = filter_data;
        if (itr == 17) {
            filter_data = filter_data + 5; /* misalign by 5 bytes (like YOLO's 0x3c05fe55) */
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

        /* Case 17: use small out_shift to expose transpose cross-position contamination.
         * out_shift=-6 (÷64) won't mask an 8x error like -10 (÷1024) would. */
        if (itr == 17) {
            for (int i = 0; i < out_channels; ++i) {
                out_shift[i] = -6;
            }
        }

        /* Case 16: override ALL data with exact YOLO Op[8] values */
        if (itr == 16) {
            static const int8_t yolo_filter[] = {
                6,127,57,21,23,8,5,109,2,15,-1,-99,14,7,-67,-59,-12,40,-90,16,-1,-3,25,7,17,-16,14,24,-53,-2,-110,-10,
                -6,-5,5,5,55,3,2,-6,-4,-17,0,17,-10,7,-3,-13,56,-3,-13,-83,-1,-4,-49,6,-127,1,5,1,8,-10,7,-2,
                3,-1,-2,0,-29,-1,-5,-14,-2,-22,-1,-1,-9,1,-12,-18,-127,-1,-14,71,-1,0,-3,-2,-5,3,0,-4,0,-21,-1,-1,
                -13,-9,-20,-77,-2,-77,-20,59,127,-7,120,-51,-9,-47,50,45,11,8,17,8,112,-20,2,-50,-12,-34,-88,-14,-59,8,-29,2,
                -4,11,-32,-32,3,-4,5,-113,-11,2,-18,-13,-2,-7,127,8,8,7,2,6,-16,-3,1,-15,1,-5,-20,-2,13,1,24,3,
                -8,-7,1,-1,54,1,1,-1,7,-6,5,3,-4,-5,-2,11,-68,-3,-10,27,5,4,-61,4,-127,3,0,2,1,1,2,3,
                0,-2,0,2,11,2,-3,3,1,-61,1,-5,127,2,-2,-5,8,0,27,9,2,-2,4,1,2,-1,2,-2,0,101,0,0,
                1,2,-51,-1,6,-6,2,10,-7,-2,2,-19,2,-3,-115,127,12,12,6,1,0,-6,2,-22,5,-4,-18,3,1,-2,51,-2,
                -20,-21,-60,123,-4,127,-17,25,-80,-7,-95,-45,-7,11,49,51,14,0,-4,8,-73,-52,-5,-47,-11,-33,119,-21,-31,4,21,1,
                -1,4,-1,1,-2,1,-1,1,1,71,1,-3,-127,1,2,4,-1,3,46,1,2,-2,-7,-1,2,-2,1,0,-1,85,2,0,
                -2,-127,81,81,-5,71,18,22,80,14,83,58,14,-2,-14,36,6,29,-106,7,71,-46,-27,88,-19,-66,79,-13,77,7,66,-18,
                46,17,-9,-24,3,17,-22,-4,-9,-1,-36,15,-1,-49,11,-3,5,-29,2,2,0,64,-1,-19,4,12,-4,32,9,-5,-9,-127,
                -30,6,105,-67,-16,-61,-45,110,-56,-15,-50,-54,-18,-37,14,36,19,1,21,22,-66,-13,2,127,-4,-52,-60,-22,92,-6,45,-7,
                -14,12,18,13,5,10,12,29,6,-10,2,-29,-3,-28,-3,-15,-2,39,127,14,3,-43,13,6,1,-103,9,11,4,-10,-5,-27,
                -88,-35,-15,7,4,-6,64,-2,-48,-3,-18,-8,-3,-71,-8,0,7,63,12,3,-7,74,0,16,1,-67,-10,-78,-9,-7,5,127,
                -3,7,2,4,15,2,0,9,2,127,2,-16,74,2,3,-5,9,1,29,13,2,-5,-16,1,8,-4,3,2,-2,-122,-2,-1,
                8,-15,-2,3,11,-1,0,57,1,-7,2,19,-4,-2,-127,54,0,17,48,0,0,2,-2,-4,5,5,4,-5,-8,-7,-20,5,
                11,39,-91,-65,11,-67,3,4,-56,-4,-66,-3,-3,5,4,8,-3,6,28,-8,-51,11,15,-106,10,23,-73,9,-127,-3,-78,8,
                8,-3,1,0,-127,0,-7,-5,-2,-17,-2,0,5,-3,-7,-5,-3,11,-10,-3,-2,-2,3,1,70,-3,2,-7,2,-17,0,-1,
                5,-127,13,6,2,3,12,25,0,-3,-2,-45,0,-6,4,-6,9,-11,-19,6,0,4,-14,9,2,25,3,0,3,5,-5,3,
                -9,36,18,-4,-4,2,-19,-101,0,10,-9,-127,-4,-5,-37,82,-1,-20,6,-13,-2,-1,4,4,1,-11,3,-10,-27,5,-45,-3,
                -6,-13,6,-3,4,-1,-5,5,2,-4,-2,-5,-1,-16,-5,-1,0,-1,0,-1,-15,-127,-3,-2,0,23,-3,0,-1,2,0,16,
                3,45,17,24,8,27,-5,42,25,-9,21,-47,-14,18,27,23,-4,-15,127,19,24,14,19,21,-2,39,23,9,14,-7,15,15,
                -127,-30,2,-10,3,-5,71,11,-16,0,-15,-18,3,-3,14,6,-1,73,-3,-1,-12,27,-2,-2,0,8,-7,-108,9,3,-6,8,
                63,-47,0,-37,11,-20,-48,6,-19,-1,-18,13,-3,76,-18,15,3,-48,16,2,-4,-34,-6,3,7,-127,-7,58,1,-3,-23,108,
                102,-1,0,3,11,1,-127,-7,-4,0,-2,-8,-13,-6,-6,-22,5,115,18,7,-1,-6,-4,3,-5,10,-1,-88,0,4,-1,7,
                127,-5,12,-6,10,-13,-89,16,-20,1,-24,-12,-6,4,-4,-15,-3,-110,3,-6,-17,89,-10,9,13,-80,-18,105,-3,-4,3,-85,
                2,7,2,-1,-25,-2,-5,-5,0,-25,-1,-6,-5,0,-14,-24,74,0,-13,-127,-1,0,-1,-1,-4,0,-1,-4,-2,-19,-8,1,
                -84,-1,-6,-2,-19,-4,105,-3,-2,8,-2,-32,2,-3,2,-21,1,-127,-10,5,-3,8,0,2,-5,-8,-4,85,1,10,4,2,
                19,-15,0,1,95,2,-15,-2,0,-56,-3,-4,-24,-5,-2,3,-16,-6,-37,-6,-3,1,127,-1,-119,4,2,-13,0,-41,3,3,
                -10,-29,-13,-4,5,-9,-7,71,-6,7,-3,113,-2,0,51,-127,-10,-11,26,-3,-4,-1,0,-23,1,5,-7,-9,-20,8,-2,7,
                -66,-1,-1,-10,3,-31,43,9,-18,-9,-2,-22,-2,75,22,-1,5,39,-14,4,-5,-62,-2,3,-1,69,-19,-61,-17,-2,-8,-127
            };
            static const int8_t yolo_input[] = {
                -127,-65,-96,-127,-124,-100,-122,-127,-93,-122,-127,-127,-114,-91,-126,-105,
                -127,-127,-128,-118,-102,-127,-127,-93,-127,-126,-127,-103,-127,-124,-127,-127,
                -126,-63,-128,-128,-127,-127,-122,-118,-127,-126,-128,-114,-112,-122,-120,-122,
                -114,-127,-127,-114,-126,-118,-127,-127,-127,-124,-128,-100,-128,-124,-127,-107,
                -126,-63,-128,-128,-128,-126,-120,-118,-124,-126,-128,-112,-112,-122,-120,-122,
                -114,-127,-127,-114,-128,-120,-127,-124,-127,-124,-127,-98,-128,-124,-127,-105,
                -127,-62,-127,-127,-127,-128,-118,-114,-128,-126,-126,-112,-112,-124,-122,-124,
                -114,-127,-127,-114,-127,-120,-127,-128,-127,-122,-128,-100,-128,-124,-128,-105,
                -126,-63,-128,-127,-127,-128,-120,-116,-128,-124,-128,-112,-114,-122,-120,-124,
                -114,-127,-127,-112,-126,-118,-127,-127,-127,-124,-127,-98,-128,-124,-128,-105,
                -127,-63,-128,-128,-127,-128,-120,-114,-127,-124,-120,-112,-114,-122,-122,-124,
                -114,-127,-127,-114,-127,-120,-127,-127,-127,-124,-127,-98,-124,-124,-128,-107,
                -128,-67,-127,-126,-127,-127,-118,-112,-127,-124,-122,-111,-114,-128,-118,-127,
                -114,-127,-127,-114,-128,-118,-127,-127,-127,-122,-127,-102,-127,-124,-128,-102,
                -126,-69,-128,-128,-127,-127,-120,-112,-127,-124,-118,-111,-114,-124,-124,-126,
                -112,-127,-127,-116,-128,-120,-127,-127,-127,-124,-126,-105,-128,-124,-122,-107
            };
            static const int32_t yolo_bias[] = {
                2420,1649,1302,1816,-446,1562,685,32,2503,-74,3143,463,1507,1883,-932,525,
                1205,162,540,1680,1846,388,338,274,-433,502,817,1021,812,1371,-30,1525
            };
            static const int32_t yolo_shifts[] = {
                -8,-7,-6,-8,-6,-7,-8,-7,-8,-7,-9,-6,-8,-8,-7,-8,-8,-8,-7,-7,-7,-6,-8,-7,-7,-8,-8,-7,-8,-8,-8,-7
            };
            static const int32_t yolo_mults[] = {
                0x52a119c9,0x53a7fce0,0x4430a104,0x5afd73fd,0x4a9394b6,0x5e2b6940,0x7c02c5c9,0x509cb64d,
                0x5941a055,0x5d50f6be,0x60b9e0ad,0x41e9ef39,0x67d9347b,0x6b36dcc7,0x5406c784,0x70ae9dd9,
                0x6a183a7f,0x78f48e0e,0x53e7df22,0x63cc6072,0x448b1623,0x4cd5d08c,0x6175e8be,0x5cd03362,
                0x4de1312d,0x6c5bd16d,0x6e89094f,0x64a1947e,0x78e1060e,0x63d8179b,0x791c8d51,0x532420c2
            };
            memcpy(input, yolo_input, sizeof(yolo_input));
            memcpy(filter_data, yolo_filter, sizeof(yolo_filter));
            memcpy(bias, yolo_bias, sizeof(yolo_bias));
            memcpy(out_shift, yolo_shifts, sizeof(yolo_shifts));
            memcpy(out_mult, yolo_mults, sizeof(yolo_mults));
        }

        data_dims_t input_dims = {.width = in_wd, .height = in_ht, .channels = in_channels, 1};
        data_dims_t output_dims = {.width = out_wd, .height = out_ht, .channels = out_channels, 1};
        data_dims_t filter_dims = {.width = filter_wd, .height = filter_ht, .channels = in_channels, 1};
        conv_params_t conv_params = {.in_offset = input_offset, .out_offset = out_offset,
                                    .stride = {stride_wd, stride_ht}, .padding = {pad_wd, pad_ht},
                                    .dilation = {0, 0}, .activation = {activation_min, activation_max}};
        quant_data_t quant_data = {.shift = out_shift, .mult = out_mult};

        int scratch_buf_size = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                                            &output_dims, &conv_params);
        if (scratch_buf_size > 0) {
            scratch_buf = ESP_NN_TEST_ALLOC(scratch_buf_size + 16);
            if (scratch_buf == NULL) {
                printf(ANSI_COLOR_RED"[%3d] scratch_buf alloc failed size %d\n"ANSI_COLOR_RESET,
                       itr, scratch_buf_size);
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
            goto conv_s8_cleanup;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [pad: (%d, %d), stride: (%d, %d)"
               " out: (%3d,%3d,%3d), filter: (%d, %d,%3d)]"ANSI_COLOR_RESET,
               itr, pad_wd, pad_ht, stride_wd, stride_ht, out_wd, out_ht,
               out_channels, filter_wd, filter_ht, in_channels);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);

    conv_s8_cleanup:
        /* Restore original filter pointer (may have been offset for alignment test) */
        filter_data = filter_data_orig_save;
        if (input_orig) {
            free(input_orig);
            input_orig = NULL;
        }
        if (filter_data) {
            free(filter_data);
            filter_data = NULL;
        }
        if (out_c_orig) {
            free(out_c_orig);
            out_c_orig = NULL;
        }
        if (out_opt_orig) {
            free(out_opt_orig);
            out_opt_orig = NULL;
        }
        if (bias) {
            free(bias);
            bias = NULL;
        }
        if (out_shift) {
            free(out_shift);
            out_shift = NULL;
        }
        if (out_mult) {
            free(out_mult);
            out_mult = NULL;
        }
        if (scratch_buf) {
            free(scratch_buf);
            scratch_buf = NULL;
        }
    }
}
