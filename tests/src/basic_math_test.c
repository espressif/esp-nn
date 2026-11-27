/*
 * SPDX-FileCopyrightText: 2020-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>

#include <common_functions.h>
#include <esp_nn.h>
#include "test_utils.h"

const int8_t test_add_in1[] = {
       13,   26,  -26,   26,  -13,   13,  -13,   13,  -13,   13,  -13,   13,  -26,  -51,  -26,  -51,
      -26,  -39,  -26,  -39,  -39,  -39,  -26,  -51,  -13,  -13,  -13,  -13,  -26,  -13,  -13,  -13,
      -13,  -13,    0,  -26,    0,  -13,    0,  -26,  -13,  -26,  -26,  -26,  -26,  -26,  -26,  -26,
      -13,  -13,    0,  -26,  -13,  -26,  -26,  -26,    0,    0,  -26,  -13,   13,    0,   26,    0,
       13,    0,   13,    0,    0,    0,   13,    0,   13,   26,  -26,   13,  -26,   13,  -13,   13,
      -13,   13,  -13,   13,  -26,  -26,  -13,  -26,  -26,  -26,  -26,  -26,  -39,  -26,  -26,  -26,
      -13,    0,  -13,    0,  -26,    0,  -13,    0,  -13,    0,  -13,  -13,    0,    0,    0,  -13,
      -13,  -13,  -26,  -13,  -26,  -13,  -13,  -13,  -13,    0,    0,  -13,  -13,  -13,  -13,  -13,
        0,    0,  -13,    0,   13,   13,   13,    0,    0,    0,   13,   13,    0,    0,   13,   13,
        0,   26,    0,   13,    0,   13,    0,   13,    0,   13,    0,   13,    0,   13,    0,    0,
        0,    0,    0,   13,    0,    0,    0,    0,   13,   13,    0,   13,    0,   13,    0,   13,
        0,   13,    0,   13,   13,   13,    0,   13,    0,   13,    0,   13,    0,   13,   13,   13,
       13,   13,    0,   13,    0,   13,    0,   13,   13,   13,   13,   13,   13,   13,   13,   13,
        0,   13,   13,   13,   13,   13,   13,   13,
};

const int8_t test_add_in2[] = {
     -128, -128, -103, -128,  -77, -128,  -52, -128,  -26, -128,   -1, -128, -128, -103, -103, -103,
      -77, -103,  -52, -103,  -26, -103,   -1, -103, -128,  -77, -103,  -77,  -77,  -77,  -52,  -77,
      -26,  -77,   -1,  -77, -128,  -52, -103,  -52,  -77,  -52,  -52,  -52,  -26,  -52,   -1,  -52,
     -128,  -26, -103,  -26,  -77,  -26,  -52,  -26,  -26,  -26,   -1,  -26, -128,   -1, -103,   -1,
      -77,   -1,  -52,   -1,  -26,   -1,   -1,   -1, -128, -128, -103, -128,  -77, -128,  -52, -128,
      -26, -128,   -1, -128, -128, -103, -103, -103,  -77, -103,  -52, -103,  -26, -103,   -1, -103,
     -128,  -77, -103,  -77,  -77,  -77,  -52,  -77,  -26,  -77,   -1,  -77, -128,  -52, -103,  -52,
      -77,  -52,  -52,  -52,  -26,  -52,   -1,  -52, -128,  -26, -103,  -26,  -77,  -26,  -52,  -26,
      -26,  -26,   -1,  -26, -128,   -1, -103,   -1,  -77,   -1,  -52,   -1,  -26,   -1,   -1,   -1,
     -128, -128, -103, -128,  -77, -128,  -52, -128,  -26, -128,   -1, -128, -128, -103, -103, -103,
      -77, -103,  -52, -103,  -26, -103,   -1, -103, -128,  -77, -103,  -77,  -77,  -77,  -52,  -77,
      -26,  -77,   -1,  -77, -128,  -52, -103,  -52,  -77,  -52,  -52,  -52,  -26,  -52,   -1,  -52,
     -128,  -26, -103,  -26,  -77,  -26,  -52,  -26,  -26,  -26,   -1,  -26, -128,   -1, -103,   -1,
      -77,   -1,  -52,   -1,  -26,   -1,   -1,   -1,
};

void esp_nn_add_elementwise_s8_test()
{
    /* prepare data */
    int size = 1600 + 8 + 7; /* odd len to test leftover */
    int8_t *input1;
    int8_t *input2;
    int8_t *out_data_c;
    int8_t *out_data_opt;
    int8_t *input1_orig = NULL;
    int8_t *input2_orig = NULL;
    int8_t *out_c_orig = NULL;
    int8_t *out_opt_orig = NULL;
    int32_t input1_offset = 34;
    int32_t input2_offset = 35;
    int32_t output_offset = 36;
    int32_t input1_shift = -8; // right_shift amt always <= 0
    int32_t input2_shift = -8; // right_shift amt always <= 0
    int32_t output_shift = -9; // right_shift amt always <= 0
    int32_t left_shift = 15; // always +ve
    int32_t input1_mult = INT32_MAX;
    int32_t input2_mult = INT32_MAX;
    int32_t output_mult = INT32_MAX;
    int32_t activation_min = -128;
    int32_t activation_max = 127;

    /* Cases 10-21 reach the P4/S31 PIE fast path: left_shift 14..20 with
     * TFLite-like multipliers and shifts. Case 18 and cases 22-28 each put
     * one parameter just outside that path, so the reference code runs.
     * Cases 29-35 keep both inputs aligned and offset the output by 1..7:
     * the S3 vector loop must not store through a misaligned output. */
    for (int itr = 0; itr < 36; itr++) {
        int in1_skew = 0, in2_skew = 0, out_skew = 0;
        bool in_place = false;
        activation_min = -128;
        activation_max = 127;
        if (itr >= 10) {
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            input1_mult = 0x40000000 + rand() % 0x3fff0000;
            input2_mult = 0x40000000 + rand() % 0x3fff0000;
            output_mult = 0x40000000 + rand() % 0x3fff0000;
            input1_shift = 0;
            input2_shift = -1 - rand() % 3;
            output_shift = -17 - rand() % 4;
            left_shift = 20;
            size = 4096 + 5;
        }
        switch (itr) {
        case 10: // real-model parameters, tail of 5
            break;
        case 11: // exactly one 8-element step
            size = 8;
            break;
        case 12: // one step plus a tail of 5
            size = 13;
            break;
        case 13: // unaligned inputs and output
            size = 1000 + 3;
            in1_skew = 1;
            in2_skew = 3;
            out_skew = 5;
            break;
        case 14: // output aliases input 1, as for an in-place add
            size = 1000 + 7;
            in_place = true;
            break;
        case 15: // int32 activation limits: the clamp is a no-op, the store wraps
            activation_min = INT32_MIN;
            activation_max = INT32_MAX;
            break;
        case 16: // all shifts 0, no rounding shift anywhere
            input2_shift = 0;
            output_shift = 0;
            left_shift = 14;
            break;
        case 17: // largest shifts and multipliers the fast path takes
            input1_shift = -20;
            input2_shift = -20;
            output_shift = -20;
            input1_mult = 0x7fff7fff;
            input2_mult = 0x7fff7fff;
            output_mult = 0x7fff7fff;
            break;
        case 18: // input 1 multiplier at the fast-path limit
            input1_mult = 0x7fff8000;
            break;
        case 22: // left_shift above the fast path
            left_shift = 21;
            break;
        case 23: // left_shift below the fast path
            left_shift = 13;
            break;
        case 24: // input offset below the fast path
            input1_offset = -128;
            break;
        case 25: // input 2 shift past the fast path
            input2_shift = -21;
            break;
        case 26: // output multiplier at the fast-path limit
            output_mult = 0x7fff8000;
            break;
        case 27: // output shift past the fast path
            output_shift = -21;
            break;
        case 28: // output offset past the fast path
            output_offset = 128;
            break;
        case 29: case 30: case 31: case 32: case 33: case 34: case 35:
            // aligned inputs, misaligned output
            size = 1000 + 3;
            out_skew = itr - 28;
            break;
        case 19: // narrowed activation range
            activation_min = -30;
            activation_max = 60;
            break;
        case 20: // offset edges
            input1_offset = -127;
            input2_offset = 128;
            output_offset = 127;
            break;
        case 21: // MobileNetV2 56x56x24 residual add
            size = 56 * 56 * 24;
            output_offset = -128;
            break;
        case 0: // all zeros
            input1_offset = 0;
            input2_offset = 0;
            output_offset = 0;
            input1_mult = 0;
            input2_mult = 0;
            output_mult = 0;
            input1_shift = 0;
            input2_shift = 0;
            output_shift = 0;
            left_shift = 0;
        break;
        case 1: // hit min
            input1_offset = -127;
            input2_offset = -127;
            output_offset = -128;
            input1_mult = MULT_MIN;
            input2_mult = MULT_MIN;
            output_mult = MULT_MIN;
            input1_shift = 0;
            input2_shift = 0;
            output_shift = 0;
            left_shift = 0;
        break;
        case 2: // hit max
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -127;
            input1_mult = MULT_MAX;
            input2_mult = MULT_MAX;
            output_mult = MULT_MAX;
            input1_shift = SHIFT_MIN;
            input2_shift = SHIFT_MIN;
            output_shift = SHIFT_MIN;
            left_shift = 30 - 8; // since input is 8 bits
        break;
        case 3: // hit extreme max
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -127;
            input1_mult = MULT_MAX;
            input2_mult = MULT_MAX;
            output_mult = MULT_MAX;
            input1_shift = 0;
            input2_shift = 0;
            output_shift = 0;
            left_shift = 30 - 8; // -8 since input is 8 bit
        break;
        case 4: // from yolo model
            input1_offset = 64;
            input2_offset = 128;
            output_offset = -128;
            input1_mult = 1705397815;
            input2_mult = 1073741824;
            output_mult = 1756091225;
            input1_shift = -3;
            input2_shift = 0;
            output_shift = -19;
            left_shift = 20;
            size = 216;
        break;
        case 5: // S3 dual-core split threshold (>= 8192) with an odd SIMD tail
            size = 8192 + 15;
            __attribute__((fallthrough));
        case 6: // exactly the threshold; case 5 leaves size at 8207 otherwise
            if (itr == 6) {
                size = 8192;
            }
            __attribute__((fallthrough));
        default:  // practical random input
            input1_offset = rand() % 256 - 127; // range [-127, 128]
            input2_offset = rand() % 256 - 127; // range [-127, 128]
            output_offset = rand() % 256 - 128; // range [-128, 127]
            input1_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            input2_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            input1_shift = -8 + rand() % 4;
            input2_shift = -8 + rand() % 4;
            output_shift = -8 + rand() % 4;
            left_shift = rand() % 15;
        }

        input1_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 32);
        input2_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 32);
        out_c_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 32);
        out_opt_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 32);

        if (input1_orig == NULL || input2_orig == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL) {
            TEST_SKIP("%s error allocating buffers\n", __FUNCTION__);
            goto elementwise_add_test_cleanup;
        }

        input1 = (int8_t *) ((((uint32_t) input1_orig + 15) & ~15) + in1_skew);
        input2 = (int8_t *) ((((uint32_t) input2_orig + 15) & ~15) + in2_skew);
        if (itr == 4) {
            input2 = input2_orig; // unaligned input
        }
        out_data_c = (int8_t *) (((uint32_t)out_c_orig + 15) & ~15);
        out_data_opt = (int8_t *) ((((uint32_t)out_opt_orig + 15) & ~15) + out_skew);


        if (itr == 4) {
            memcpy(input1, test_add_in1, size);
            memcpy(input2, test_add_in2, size);
        } else {
            for (int i = 0; i < size; ++i) {
                input1[i] = rand() % 256 - 128;
                input2[i] = rand() % 256 - 128;
            }
        }

        if (itr == 0) {
            /* enable profiler */
            profile_c_start();
        }
        /* C function */
        esp_nn_add_elementwise_s8_ansi(input1, input2, input1_offset, input2_offset,
                                       input1_mult, input2_mult, input1_shift, input2_shift,
                                       left_shift, out_data_c, output_offset, output_mult,
                                       output_shift, activation_min, activation_max, size);

        if (itr == 0) {
            profile_c_end();
            profile_opt_start();
        }

        if (in_place) {
            out_data_opt = input1;
        }
        /* Optimized function */
        esp_nn_add_elementwise_s8(input1, input2, input1_offset, input2_offset,
                                  input1_mult, input2_mult, input1_shift, input2_shift,
                                  left_shift, out_data_opt, output_offset, output_mult,
                                  output_shift, activation_min, activation_max, size);
        if (itr == 0) {
            /* disable profiler */
            profile_opt_end();
        }

        bool ret = CHECK_EQUAL(out_data_c, out_data_opt, size);
        if (ret == false) {
            TEST_FAIL("%s[%d] failed\n", __FUNCTION__, itr);
            printf("Output: \n");
            PRINT_ARRAY_INT8(out_data_opt, size, 1);
            printf("Expected: \n");
            PRINT_ARRAY_INT8(out_data_c, size, 1);
            printf("Input1:\n");
            PRINT_ARRAY_INT8(input1, size, 1);
            printf("Input2:\n");
            PRINT_ARRAY_INT8(input2, size, 1);
            printf("in1_shift %"PRIi32", in2_shift %"PRIi32", left_shift %"PRIi32", out_shift %"PRIi32"\n",
                   input1_shift, input2_shift, left_shift, output_shift);
            printf("in1_mult %"PRIi32", in2_mult %"PRIi32", out_mult %"PRIi32"\n",
                   input1_mult, input2_mult, output_mult);
            goto elementwise_add_test_cleanup;
        }
        TEST_PASS("%s[%d] passed\n", __FUNCTION__, itr);

elementwise_add_test_cleanup:
        if (input1_orig) {
            free(input1_orig);
        }
        if (input2_orig) {
            free(input2_orig);
        }
        if (out_c_orig) {
            free(out_c_orig);
        }
        if (out_opt_orig) {
            free(out_opt_orig);
        }
    }
}

void esp_nn_mul_elementwise_s8_test()
{
    /* prepare data */
    int size = 1600 + 8 + 7; /* odd len to test leftover */
    int8_t *input1;
    int8_t *input2;
    int8_t *out_data_c;
    int8_t *out_data_opt;
    int32_t input1_offset = 34;
    int32_t input2_offset = 35;
    int32_t output_offset = 36;
    int32_t output_shift = -7;
    int32_t output_mult = MULT_MAX; // max out_mult
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int8_t *input1_orig = NULL;
    int8_t *input2_orig = NULL;
    int8_t *out_c_orig = NULL;
    int8_t *out_opt_orig = NULL;

    for (int itr = 0; itr < 16; itr++) {
        switch (itr) {
        case 0: // all zeros
            input1_offset = 0;
            input2_offset = 0;
            output_offset = 0;
            output_mult = 0;
            output_shift = 0;
        break;
        case 1: // hit min
            input1_offset = -127;
            input2_offset = -127;
            output_offset = -128;
            output_mult = MULT_MIN;
            output_shift = 0;
        break;
        case 2: // hit max
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -127;
            output_mult = MULT_MAX;
            output_shift = SHIFT_MIN;
        break;
        case 3: // hit extreme max
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -127;
            output_mult = MULT_MAX;
            output_shift = 0;
        break;
        case 14: // full-range activation limits, right shift 5: the 16-bit
                 // clamps cannot hold them, so the reference code runs
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = -5;
            activation_min = INT32_MIN;
            activation_max = INT32_MAX;
            size = 4096 + 5;
        break;
        case 15: // 4400 steps on each core of the two-core split, 4376 of them
                 // with prefetch: past the ESP32-P4 rev < v3 hardware loop limit
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = -8 + rand() % 4;
            activation_min = -128;
            activation_max = 127;
            size = 2 * 16 * 4400 + 37;
        break;
        case 12: // narrowed activation (fused ReLU-like), right shift 0: s32 clamp
        case 13: // narrowed activation, right shift 5: s16 clamp
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = (itr == 12) ? 0 : -5;
            activation_min = -20;
            activation_max = 60;
            size = 4096 + 5;
        break;
        case 10: // left shift: the S3 vector path takes it for 8-byte aligned buffers
        case 11: // right shift above 16
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = (itr == 10) ? 3 : -20;
            size = 4096;
        break;
        default:  // practical random input
            input1_offset = rand() % 256 - 127; // range [-127, 128]
            input2_offset = rand() % 256 - 127; // range [-127, 128]
            output_offset = rand() % 256 - 128; // range [-128, 127]
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = -8 + rand() % 4;
            size = 4 + rand() % 64;
            if (itr == 5) {
                /* Unaligned input2 (see below) sends the whole array through
                 * the S3 kernel's scalar path; at this length a requant
                 * rounding tie is a statistical certainty, which is what
                 * caught the scalar path using the wrong nudge register. */
                size = 4096 + 3;
            } else if (itr == 6) {
                size = 8192 + 15; // dual-core split threshold, odd SIMD tail
            } else if (itr == 7) {
                size = 8192;      // exactly the threshold
            }
        }

        input1_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);
        input2_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);
        out_c_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);
        out_opt_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);

        if (input1_orig == NULL || input2_orig == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL) {
            TEST_SKIP("%s error allocating buffers\n", __FUNCTION__);
            goto elementwise_mult_test_cleanup;
        }

        input1 = (int8_t *) (((uint32_t) input1_orig + 15) & ~15);
        input2 = (int8_t *) (((uint32_t) input2_orig + 15) & ~15);
        if (itr == 4 || itr == 5) {
            input2 = input2_orig; // unaligned input
        }

        out_data_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
        out_data_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

        for (int i = 0; i < size; ++i) {
            input1[i] = rand() % 256 - 128;
            input2[i] = rand() % 256 - 128;
        }

        if (itr == 0) {
            /* enable profiler */
            profile_c_start();
        }
        /* C function */
        esp_nn_mul_elementwise_s8_ansi(input1, input2, input1_offset, input2_offset,
                                       out_data_c, output_offset, output_mult, output_shift,
                                       activation_min, activation_max, size);

        if (itr == 0) {
            profile_c_end();
            profile_opt_start();
        }
        /* Optimized function */
        esp_nn_mul_elementwise_s8(input1, input2, input1_offset, input2_offset,
                                  out_data_opt, output_offset, output_mult, output_shift,
                                  activation_min, activation_max, size);

        if (itr == 0) {
            /* disable profiler */
            profile_opt_end();
        }

        bool ret = CHECK_EQUAL(out_data_c, out_data_opt, size);
        if (ret == false) {
            TEST_FAIL("%s[%d] failed\n", __FUNCTION__, itr);
            printf("Output: \n");
            PRINT_ARRAY_HEX(out_data_opt, size, 1);
            printf("Expected: \n");
            PRINT_ARRAY_HEX(out_data_c, size, 1);
            printf("Input1:\n");
            PRINT_ARRAY_HEX(input1, size, 1);
            printf("Input2:\n");
            PRINT_ARRAY_HEX(input2, size, 1);
            goto elementwise_mult_test_cleanup;
        }
        TEST_PASS("%s[%d] passed\n", __FUNCTION__, itr);

elementwise_mult_test_cleanup:
        if (input1_orig) {
            free(input1_orig);
        }
        if (input2_orig) {
            free(input2_orig);
        }
        if (out_c_orig) {
            free(out_c_orig);
        }
        if (out_opt_orig) {
            free(out_opt_orig);
        }
    }
}

void esp_nn_mul_broadcast_channel_s8_test()
{
    int total_spatial = 49; /* 7x7 feature map */
    int channels = 64;
    int8_t *input1;
    int8_t *input2_per_ch;
    int8_t *out_data_c;
    int8_t *out_data_opt;
    int8_t *input1_orig = NULL;
    int8_t *input2_orig = NULL;
    int8_t *out_c_orig = NULL;
    int8_t *out_opt_orig = NULL;
    int32_t input1_offset = 34;
    int32_t input2_offset = 35;
    int32_t output_offset = 36;
    int32_t output_shift = -7;
    int32_t output_mult = MULT_MAX;
    int32_t activation_min = -128;
    int32_t activation_max = 127;

    for (int itr = 0; itr < 15; itr++) {
        switch (itr) {
        case 0: // all zeros
            input1_offset = 0;
            input2_offset = 0;
            output_offset = 0;
            output_mult = 0;
            output_shift = 0;
            total_spatial = 49;
            channels = 64;
        break;
        case 1: // hit min
            input1_offset = -127;
            input2_offset = -127;
            output_offset = -128;
            output_mult = MULT_MIN;
            output_shift = 0;
        break;
        case 2: // hit max
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -127;
            output_mult = MULT_MAX;
            output_shift = SHIFT_MIN;
        break;
        case 3: // small channels (leftover only, no SIMD)
            input1_offset = 64;
            input2_offset = 32;
            output_offset = -10;
            output_mult = MULT_MAX / 2;
            output_shift = -5;
            total_spatial = 16;
            channels = 5;
        break;
        case 4: // unaligned channels (SIMD + leftover)
            total_spatial = 14;
            channels = 19;
        break;
        case 5: // typical SE-block: 7x7 spatial, 96 channels
            input1_offset = 128;
            input2_offset = 128;
            output_offset = -128;
            output_mult = 1705397815;
            output_shift = -3;
            total_spatial = 49;
            channels = 96;
        break;
        case 10: // left shift: the S3 vector path takes it for channels % 8 == 0
        case 11: // right shift above 16
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = (itr == 10) ? 3 : -20;
            total_spatial = 64;
            channels = 96;
        break;
        case 14: // full-range activation limits, right shift 5: the 16-bit
                 // clamps cannot hold them, so the reference code runs
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = -5;
            activation_min = INT32_MIN;
            activation_max = INT32_MAX;
            total_spatial = 37;
            channels = 40;
        break;
        case 12: // narrowed activation, right shift 0, PIE rows
        case 13: // narrowed activation, right shift 5, PIE rows
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = (itr == 12) ? 0 : -5;
            activation_min = -20;
            activation_max = 60;
            total_spatial = 37;
            channels = 40;
        break;
        default: // random
            input1_offset = rand() % 256 - 127;
            input2_offset = rand() % 256 - 127;
            output_offset = rand() % 256 - 128;
            output_mult = MULT_MAX / 2 + rand() % INT16_MAX;
            output_shift = -8 + rand() % 4;
            total_spatial = 4 + rand() % 64;
            channels = 8 + rand() % 128;
        }

        int size = total_spatial * channels;
        input1_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);
        input2_orig = (int8_t *) ESP_NN_TEST_ALLOC(channels + 16);
        out_c_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);
        out_opt_orig = (int8_t *) ESP_NN_TEST_ALLOC(size + 16);

        if (input1_orig == NULL || input2_orig == NULL ||
                out_c_orig == NULL || out_opt_orig == NULL) {
            TEST_SKIP("%s error allocating buffers\n", __FUNCTION__);
            goto broadcast_mul_test_cleanup;
        }

        input1 = (int8_t *) (((uint32_t) input1_orig + 15) & ~15);
        input2_per_ch = (int8_t *) (((uint32_t) input2_orig + 15) & ~15);
        out_data_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
        out_data_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

        if (itr == 4) {
            input1 = input1_orig; // unaligned input
        }

        for (int i = 0; i < size; ++i) {
            input1[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < channels; ++i) {
            input2_per_ch[i] = rand() % 256 - 128;
        }

        if (itr == 0) {
            profile_c_start();
        }
        /* C reference */
        esp_nn_mul_broadcast_channel_s8_ansi(input1, input2_per_ch,
                                             input1_offset, input2_offset,
                                             out_data_c, output_offset,
                                             output_mult, output_shift,
                                             activation_min, activation_max,
                                             total_spatial, channels);
        if (itr == 0) {
            profile_c_end();
            profile_opt_start();
        }
        /* Optimized function */
        esp_nn_mul_broadcast_channel_s8(input1, input2_per_ch,
                                        input1_offset, input2_offset,
                                        out_data_opt, output_offset,
                                        output_mult, output_shift,
                                        activation_min, activation_max,
                                        total_spatial, channels);
        if (itr == 0) {
            profile_opt_end();
        }

        bool ret = CHECK_EQUAL(out_data_c, out_data_opt, size);
        if (ret == false) {
            TEST_FAIL("%s[%d] failed\n", __FUNCTION__, itr);
            printf("spatial=%d channels=%d size=%d\n", total_spatial, channels, size);
            for (int idx = 0; idx < size; idx++) {
                if (out_data_c[idx] != out_data_opt[idx]) {
                    printf("first mismatch at idx=%d (row=%d ch=%d): got %02x exp %02x\n",
                           idx, idx / channels, idx % channels,
                           (uint8_t)out_data_opt[idx], (uint8_t)out_data_c[idx]);
                    // print 8 more mismatches
                    int cnt = 0;
                    for (int j = idx + 1; j < size && cnt < 8; j++) {
                        if (out_data_c[j] != out_data_opt[j]) {
                            printf("  mismatch at idx=%d (row=%d ch=%d): got %02x exp %02x\n",
                                   j, j / channels, j % channels,
                                   (uint8_t)out_data_opt[j], (uint8_t)out_data_c[j]);
                            cnt++;
                        }
                    }
                    break;
                }
            }
            goto broadcast_mul_test_cleanup;
        }
        TEST_PASS("%s[%d] passed\n", __FUNCTION__, itr);

broadcast_mul_test_cleanup:
        if (input1_orig) {
            free(input1_orig);
        }
        if (input2_orig) {
            free(input2_orig);
        }
        if (out_c_orig) {
            free(out_c_orig);
        }
        if (out_opt_orig) {
            free(out_opt_orig);
        }
    }
}

/*
 * Multiply sweep: the elementwise and channel-broadcast kernels against the
 * ANSI C reference over lengths and channel counts around the 16-element
 * vector step, every buffer alignment, all output shifts, the multiplier and
 * offset limits, narrow activation ranges, and random parameters, all within
 * what TFLite produces (multiplier >= 0, input offsets -127..128, output
 * offset and range in int8). Guard bytes around the output catch writes past
 * either end.
 */
#define MUL_SWEEP_MAX       (3 * 96)
#define MUL_SWEEP_GUARD     32
#define MUL_SWEEP_GUARD_VAL 0x5a

typedef struct {
    int32_t in1_off, in2_off, out_off, mult, shift, act_min, act_max;
} mul_sweep_q_t;

static uint32_t mul_sweep_seed;

static uint32_t mul_sweep_rand(void)
{
    mul_sweep_seed ^= mul_sweep_seed << 13;
    mul_sweep_seed ^= mul_sweep_seed >> 17;
    mul_sweep_seed ^= mul_sweep_seed << 5;
    return mul_sweep_seed;
}

/* Run one case on buffers at the given misalignments and compare. */
static bool mul_sweep_case(int8_t *const bufs[4], bool broadcast, int spatial, int channels,
                           const mul_sweep_q_t *q, int al1, int al2, int alo, int fill)
{
    const int size = spatial * channels;
    const int n2 = broadcast ? channels : size;
    int8_t *in1 = bufs[0] + al1;
    int8_t *in2 = bufs[1] + al2;
    int8_t *out_c = bufs[2] + MUL_SWEEP_GUARD + alo;
    int8_t *out_opt = bufs[3] + MUL_SWEEP_GUARD + alo;

    for (int i = 0; i < size; i++) {
        in1[i] = fill == 1 ? -128 : fill == 2 ? 127 : (int8_t) mul_sweep_rand();
    }
    for (int i = 0; i < n2; i++) {
        in2[i] = fill == 1 ? -128 : fill == 2 ? 127 : (int8_t) mul_sweep_rand();
    }
    memset(bufs[3], MUL_SWEEP_GUARD_VAL, MUL_SWEEP_MAX + 2 * MUL_SWEEP_GUARD + 16);

    if (broadcast) {
        esp_nn_mul_broadcast_channel_s8_ansi(in1, in2, q->in1_off, q->in2_off, out_c, q->out_off,
                                             q->mult, q->shift, q->act_min, q->act_max,
                                             spatial, channels);
        esp_nn_mul_broadcast_channel_s8(in1, in2, q->in1_off, q->in2_off, out_opt, q->out_off,
                                        q->mult, q->shift, q->act_min, q->act_max,
                                        spatial, channels);
    } else {
        esp_nn_mul_elementwise_s8_ansi(in1, in2, q->in1_off, q->in2_off, out_c, q->out_off,
                                       q->mult, q->shift, q->act_min, q->act_max, size);
        esp_nn_mul_elementwise_s8(in1, in2, q->in1_off, q->in2_off, out_opt, q->out_off,
                                  q->mult, q->shift, q->act_min, q->act_max, size);
    }

    bool ok = CHECK_EQUAL(out_c, out_opt, size);
    for (int i = 0; ok && i < MUL_SWEEP_GUARD + alo; i++) {
        ok = bufs[3][i] == (int8_t) MUL_SWEEP_GUARD_VAL;
    }
    for (int i = 0; ok && i < MUL_SWEEP_GUARD; i++) {
        ok = out_opt[size + i] == (int8_t) MUL_SWEEP_GUARD_VAL;
    }
    if (!ok) {
        printf("%s: %s spatial=%d channels=%d align=%d/%d/%d fill=%d off=%"PRId32"/%"PRId32"/%"PRId32
               " mult=%"PRId32" shift=%"PRId32" act=[%"PRId32", %"PRId32"]\n",
               __FUNCTION__, broadcast ? "broadcast" : "elementwise", spatial, channels,
               al1, al2, alo, fill, q->in1_off, q->in2_off, q->out_off, q->mult, q->shift,
               q->act_min, q->act_max);
        for (int i = 0; i < size; i++) {
            if (out_c[i] != out_opt[i]) {
                printf("first mismatch at %d: got %d exp %d (in1 %d)\n",
                       i, out_opt[i], out_c[i], in1[i]);
                break;
            }
        }
    }
    return ok;
}

/* Both kernels for one quantization over a few shapes and alignments. */
static bool mul_sweep_both(int8_t *const bufs[4], const mul_sweep_q_t *q, int fill)
{
    static const int shapes[][2] = { {1, 1}, {1, 16}, {3, 17}, {2, 48}, {5, 31}, {1, 96} };
    bool ok = true;
    for (int s = 0; ok && s < (int) (sizeof(shapes) / sizeof(shapes[0])); s++) {
        const int al = (s * 5) % 16;
        ok = mul_sweep_case(bufs, false, shapes[s][0], shapes[s][1], q, al, (al + 3) % 16,
                            (al + 7) % 16, fill) &&
             mul_sweep_case(bufs, true, shapes[s][0], shapes[s][1], q, (al + 9) % 16, al,
                            (al + 1) % 16, fill);
    }
    return ok;
}

static mul_sweep_q_t mul_sweep_random_q(void)
{
    mul_sweep_q_t q;
    q.in1_off = (int32_t) (mul_sweep_rand() % 256) - 127;
    q.in2_off = (int32_t) (mul_sweep_rand() % 256) - 127;
    q.out_off = (int32_t) (mul_sweep_rand() % 256) - 128;
    q.mult = (int32_t) (mul_sweep_rand() & 0x7fffffff);
    q.shift = -(int32_t) (mul_sweep_rand() % 21);
    int32_t a = (int32_t) (mul_sweep_rand() % 256) - 128;
    int32_t b = (int32_t) (mul_sweep_rand() % 256) - 128;
    q.act_min = a < b ? a : b;
    q.act_max = a < b ? b : a;
    return q;
}

void esp_nn_mul_s8_sweep_test()
{
    int8_t *bufs[4] = { NULL };
    const int lens[] = { MUL_SWEEP_MAX + 16, MUL_SWEEP_MAX + 16,
                         MUL_SWEEP_MAX + 2 * MUL_SWEEP_GUARD + 16,
                         MUL_SWEEP_MAX + 2 * MUL_SWEEP_GUARD + 16 };
    for (int i = 0; i < 4; i++) {
        bufs[i] = (int8_t *) ESP_NN_TEST_ALLOC(lens[i] + 16);
        if (bufs[i] == NULL) {
            TEST_SKIP("%s error allocating buffers\n", __FUNCTION__);
            goto mul_sweep_cleanup;
        }
    }
    /* 16-byte aligned bases; each case adds its own misalignment */
    int8_t *base[4];
    for (int i = 0; i < 4; i++) {
        base[i] = (int8_t *) (((uintptr_t) bufs[i] + 15) & ~(uintptr_t) 15);
    }
    mul_sweep_seed = 0x9e3779b9u;
    const mul_sweep_q_t typical = { 128, 128, -5, 0x5a827999, -8, -128, 127 };
    bool ok;

    /* 1. Lengths and channel counts 1..96, spatial 1..3, all alignments */
    ok = true;
    for (int n = 1; ok && n <= 96; n++) {
        for (int sp = 1; ok && sp <= 3; sp++) {
            ok = mul_sweep_case(base, false, sp, n, &typical, n % 16, (n * 3) % 16,
                                (n * 7) % 16, 0) &&
                 mul_sweep_case(base, true, sp, n, &typical, (n * 5) % 16, (n * 11) % 16,
                                (n * 13) % 16, 0);
        }
    }
    if (ok) {
        for (int al = 0; ok && al < 16; al++) {
            for (int al2 = 0; ok && al2 < 16; al2 += 5) {
                ok = mul_sweep_case(base, false, 1, 80, &typical, al, al2, (al + al2) % 16, 0) &&
                     mul_sweep_case(base, true, 2, 64, &typical, al2, al, (al * 3) % 16, 0);
            }
        }
    }
    if (ok) {
        TEST_PASS("%s[lengths and alignments] passed\n", __FUNCTION__);
    } else {
        TEST_FAIL("%s[lengths and alignments] failed\n", __FUNCTION__);
    }

    /* 2. Every output shift down to -30 (the reference's mask for 31 overflows),
     * left shifts included */
    ok = true;
    for (int32_t shift = -30; ok && shift <= 4; shift++) {
        mul_sweep_q_t q = typical;
        q.shift = shift;
        q.mult = shift > 0 ? 0x40000000 + 12345 : 0x7ffffff0 + shift * 977;
        ok = mul_sweep_both(base, &q, 0);
    }
    if (ok) {
        TEST_PASS("%s[shifts] passed\n", __FUNCTION__);
    } else {
        TEST_FAIL("%s[shifts] failed\n", __FUNCTION__);
    }

    /* 3. Multipliers at the limits, with the shifts around the vector range */
    ok = true;
    {
        const int32_t mults[] = { 0, 1, 2, 0x3fffffff, 0x40000000, 0x7fff7fff, 0x7fff8000,
                                  0x7ffffffe, INT32_MAX };
        const int32_t shifts[] = { 0, -1, -7, -13, -16, -17, -18, -19 };
        for (int m = 0; ok && m < (int) (sizeof(mults) / sizeof(mults[0])); m++) {
            for (int s = 0; ok && s < (int) (sizeof(shifts) / sizeof(shifts[0])); s++) {
                for (int fill = 0; ok && fill < 3; fill++) {
                    mul_sweep_q_t q = typical;
                    q.mult = mults[m];
                    q.shift = shifts[s];
                    ok = mul_sweep_both(base, &q, fill);
                }
            }
        }
    }
    if (ok) {
        TEST_PASS("%s[multipliers] passed\n", __FUNCTION__);
    } else {
        TEST_FAIL("%s[multipliers] failed\n", __FUNCTION__);
    }

    /* 4. Offsets and activation ranges at the limits */
    ok = true;
    {
        const int32_t in_offs[] = { -127, -1, 0, 1, 127, 128 };
        const int32_t out_offs[] = { -128, -1, 0, 127 };
        const int32_t acts[][2] = { {-128, 127}, {-10, 10}, {0, 0}, {-128, -128}, {127, 127} };
        for (int a = 0; ok && a < (int) (sizeof(in_offs) / sizeof(in_offs[0])); a++) {
            for (int b = 0; ok && b < (int) (sizeof(in_offs) / sizeof(in_offs[0])); b++) {
                for (int o = 0; ok && o < (int) (sizeof(out_offs) / sizeof(out_offs[0])); o++) {
                    for (int r = 0; ok && r < (int) (sizeof(acts) / sizeof(acts[0])); r++) {
                        mul_sweep_q_t q = typical;
                        q.in1_off = in_offs[a];
                        q.in2_off = in_offs[b];
                        q.out_off = out_offs[o];
                        q.act_min = acts[r][0];
                        q.act_max = acts[r][1];
                        q.shift = -1 - (a + b + o + r) % 12;
                        ok = mul_sweep_both(base, &q, (a + r) % 3);
                    }
                }
            }
        }
    }
    if (ok) {
        TEST_PASS("%s[offsets and ranges] passed\n", __FUNCTION__);
    } else {
        TEST_FAIL("%s[offsets and ranges] failed\n", __FUNCTION__);
    }

    /* 5. Random parameters and shapes */
    ok = true;
    for (int i = 0; ok && i < 400; i++) {
        const mul_sweep_q_t q = mul_sweep_random_q();
        const int sp = 1 + (int) (mul_sweep_rand() % 3);
        const int ch = 1 + (int) (mul_sweep_rand() % 96);
        const int al1 = (int) (mul_sweep_rand() % 16);
        const int al2 = (int) (mul_sweep_rand() % 16);
        const int alo = (int) (mul_sweep_rand() % 16);
        ok = mul_sweep_case(base, false, sp, ch, &q, al1, al2, alo, 0) &&
             mul_sweep_case(base, true, sp, ch, &q, al2, alo, al1, 0);
    }
    if (ok) {
        TEST_PASS("%s[random] passed\n", __FUNCTION__);
    } else {
        TEST_FAIL("%s[random] failed\n", __FUNCTION__);
    }

mul_sweep_cleanup:
    for (int i = 0; i < 4; i++) {
        if (bufs[i]) {
            free(bufs[i]);
        }
    }
}
