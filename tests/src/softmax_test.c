/*
 * SPDX-FileCopyrightText: 2022-2026 Espressif Systems (Shanghai) CO LTD
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

static void run_softmax_test(int32_t height, int32_t width, int32_t mult,
                             int32_t shift, int32_t diff_min, int iter)
{
    void *scratch_buf = NULL, *scratch_buf_orig = NULL;
    const int size = width * height;
    int8_t *input = NULL, *out_ansi = NULL, *out_opt = NULL;

    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(size + 16);
    int8_t *out_opt_orig = malloc(size + 16);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        TEST_SKIP("softmax [%d] allocations failed\n", iter);
        goto softmax_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    out_ansi = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    out_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 255 - 128;
    }

    profile_c_start();
    esp_nn_softmax_s8_ansi(input, height, width, mult, shift, diff_min, out_ansi);
    profile_c_end();

    int32_t scratch_buf_size = esp_nn_get_softmax_scratch_size(width, height);
    if (scratch_buf_size) {
        scratch_buf_orig = malloc(scratch_buf_size * 4 + 16);
        if (scratch_buf_orig == NULL) {
            TEST_SKIP("softmax [%d] scratch alloc failed size %"PRIi32"\n",
                      iter, scratch_buf_size);
            goto softmax_cleanup;
        }
        scratch_buf = (void *)(((uint32_t) scratch_buf_orig + 15) & ~15);
        esp_nn_set_softmax_scratch_buf(scratch_buf);
    }

    profile_opt_start();
    esp_nn_softmax_s8(input, height, width, mult, shift, diff_min, out_opt);
    profile_opt_end();

    bool ret = CHECK_EQUAL(out_ansi, out_opt, size);
    if (ret == false) {
        TEST_FAIL("softmax [%d] failed [h %"PRIi32", w %"PRIi32", mult %"PRIi32", shift %"PRIi32", diff_min %"PRIi32"]\n",
                  iter, height, width, mult, shift, diff_min);
        printf("Output: \n");
        PRINT_ARRAY_HEX(out_opt, width, height);
        printf("Expected: \n");
        PRINT_ARRAY_HEX(out_ansi, width, height);
        goto softmax_cleanup;
    }
    TEST_PASS("softmax [%2d] passed [h %"PRIi32", w %"PRIi32", mult %"PRIi32", shift %"PRIi32"]\n",
              iter, height, width, mult, shift);

softmax_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
    if (scratch_buf_orig) free(scratch_buf_orig);
}

/* TFLite's prepare for input scale 1/16, beta 1: multiplier 2^30, left shift
 * 23, diff_min -248. The exp argument spans [-15.5, 0]. */
#define SM_MULT     (1 << 30)
#define SM_SHIFT    23
#define SM_DIFF_MIN -248

/* A row where every value is at the maximum or one step (1/16) below it, so
 * the sum of the row is large: the kernels divide by 2^31 when it is 256 or
 * more, and by 2^32 or more from 512. Unlike run_softmax_test(), this checks
 * both kernels against known outputs: `expected` for the values at the
 * maximum, `expected_below` for the others. */
static void run_softmax_wide_row_test(int32_t width, int32_t below_every,
                                      int8_t expected, int8_t expected_below, int iter)
{
    void *scratch_buf_orig = NULL;
    int8_t *input = malloc(width);
    int8_t *out_ansi = malloc(width);
    int8_t *out_opt = malloc(width);
    if (input == NULL || out_ansi == NULL || out_opt == NULL) {
        TEST_SKIP("softmax [%d] allocations failed\n", iter);
        goto softmax_wide_row_cleanup;
    }

    for (int i = 0; i < width; ++i) {
        input[i] = (below_every && (i % below_every) == below_every - 1) ? 99 : 100;
    }

    esp_nn_softmax_s8_ansi(input, 1, width, SM_MULT, SM_SHIFT, SM_DIFF_MIN, out_ansi);

    int32_t scratch_buf_size = esp_nn_get_softmax_scratch_size(width, 1);
    if (scratch_buf_size) {
        scratch_buf_orig = malloc(scratch_buf_size + 16);
        if (scratch_buf_orig == NULL) {
            TEST_SKIP("softmax [%d] scratch alloc failed size %"PRIi32"\n",
                      iter, scratch_buf_size);
            goto softmax_wide_row_cleanup;
        }
        esp_nn_set_softmax_scratch_buf((void *)(((uint32_t) scratch_buf_orig + 15) & ~15));
    }
    esp_nn_softmax_s8(input, 1, width, SM_MULT, SM_SHIFT, SM_DIFF_MIN, out_opt);

    for (int i = 0; i < width; ++i) {
        const int8_t want = input[i] == 100 ? expected : expected_below;
        if (out_ansi[i] != want || out_opt[i] != want) {
            TEST_FAIL("softmax [%d] failed [wide row, w %"PRIi32", below every %"PRIi32"]: "
                      "index %d, ansi %d, optimized %d, expected %d\n",
                      iter, width, below_every, i, out_ansi[i], out_opt[i], want);
            goto softmax_wide_row_cleanup;
        }
    }
    TEST_PASS("softmax [%2d] passed [wide row, w %"PRIi32", below every %"PRIi32"]\n",
              iter, width, below_every);

softmax_wide_row_cleanup:
    if (input) free(input);
    if (out_ansi) free(out_ansi);
    if (out_opt) free(out_opt);
    if (scratch_buf_orig) free(scratch_buf_orig);
}

void esp_nn_softmax_s8_test()
{
    int iter = 0;
    /* Original test case */
    run_softmax_test(8, 32, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Small output classes (person_detection: 2, micro_speech: 4) */
    run_softmax_test(1, 2, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    run_softmax_test(1, 4, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Single element (degenerate) */
    run_softmax_test(1, 1, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Medium width */
    run_softmax_test(1, 10, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    run_softmax_test(4, 10, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Large width (ImageNet-class) */
    run_softmax_test(1, 1000, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Large height */
    run_softmax_test(64, 32, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Varying diff_min */
    run_softmax_test(8, 32, SM_MULT, SM_SHIFT, -64, iter++);
    run_softmax_test(8, 32, SM_MULT, SM_SHIFT, -32, iter++);
    run_softmax_test(8, 32, SM_MULT, SM_SHIFT, 0, iter++);
    /* Other TFLite scalings: input scale 1/256 (every element included) and
     * 0.1 (a multiplier that is not a power of two). */
    run_softmax_test(8, 32, 1 << 30, 19, -3968, iter++);
    run_softmax_test(8, 32, 1717986918, 23, -248, iter++);
    /* Small multiplier and shift: exp arguments close to 0 */
    run_softmax_test(8, 32, INT32_MAX / 2, 7, -128, iter++);
    run_softmax_test(8, 32, INT32_MAX / 4, 5, -128, iter++);
    run_softmax_test(8, 32, INT32_MAX, 10, -128, iter++);
    /* Input scale 16 and above: left shift 31, diff_min 0 */
    run_softmax_test(8, 32, 1 << 30, 31, 0, iter++);
    run_softmax_test(8, 32, INT32_MAX, 31, 0, iter++);
    /* Odd width (non-aligned) */
    run_softmax_test(8, 17, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    run_softmax_test(8, 3, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* SIMD-width rows whose stride is not a multiple of 16: rows 1+ start
     * unaligned (yolo11n op140 is 36-wide) - regression for the S3
     * find-max, which must not issue unaligned ee.vld.128 loads. */
    run_softmax_test(8, 36, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    run_softmax_test(4, 1000, SM_MULT, SM_SHIFT, SM_DIFF_MIN, iter++);
    /* Row sums from 256 (division by 2^31) and from 512 (by 2^32 and 2^33,
     * where every probability is below half an output step). */
    run_softmax_wide_row_test(256, 0, -127, -127, iter++);
    run_softmax_wide_row_test(511, 0, -127, -127, iter++);
    run_softmax_wide_row_test(511, 2, -127, -128, iter++);
    run_softmax_wide_row_test(400, 2, -127, -127, iter++);
    run_softmax_wide_row_test(512, 0, -128, -128, iter++);
    run_softmax_wide_row_test(1000, 0, -128, -128, iter++);
    run_softmax_wide_row_test(1000, 2, -128, -128, iter++);
    run_softmax_wide_row_test(1024, 0, -128, -128, iter++);
    run_softmax_wide_row_test(2048, 3, -128, -128, iter++);
    /* 4096 values at the maximum and more: the row sum passes INT32_MAX,
     * and 2^32 from 8192. */
    run_softmax_wide_row_test(4096, 0, -128, -128, iter++);
    run_softmax_wide_row_test(8192, 0, -128, -128, iter++);
    run_softmax_wide_row_test(8193, 0, -128, -128, iter++);
}
