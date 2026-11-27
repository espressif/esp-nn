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
#include <string.h>

#include <esp_nn.h>
#include "test_utils.h"

static void run_avg_pool_test_act(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                                  uint16_t filter_wd, uint16_t filter_ht,
                                  uint16_t stride_wd, uint16_t stride_ht,
                                  uint16_t pad_wd, uint16_t pad_ht,
                                  int32_t activation_min, int32_t activation_max,
                                  int iter)
{
    const uint16_t out_wd = (input_wd + 2 * pad_wd - filter_wd) / stride_wd + 1;
    const uint16_t out_ht = (input_ht + 2 * pad_ht - filter_ht) / stride_ht + 1;
    const int size = input_wd * input_ht * channels;
    const int out_size = out_wd * out_ht * channels;
    /* Bytes checked after the output: enough to catch a row stride of up to
     * max(out_wd, out_ht) * channels without the overrun reaching the heap. */
    const int guard = out_ht * (out_wd > out_ht ? out_wd : out_ht) * channels - out_size + 512;

    int8_t *input = NULL, *output_c = NULL, *output_opt = NULL;
    int8_t *input_orig = malloc(size + 16);
    int8_t *out_c_orig = malloc(out_size + 16 + guard);
    int8_t *out_opt_orig = malloc(out_size + 16 + guard);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        TEST_SKIP("avg_pool [%d] allocations failed\n", iter);
        goto avg_pool_cleanup;
    }

    input = (int8_t *) (((uint32_t) input_orig + 15) & ~15);
    output_c = (int8_t *) (((uint32_t) out_c_orig + 15) & ~15);
    output_opt = (int8_t *) (((uint32_t) out_opt_orig + 15) & ~15);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
    }
    memset(output_opt + out_size, 0x5a, guard);

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

    for (int i = 0; i < guard; ++i) {
        if ((uint8_t) output_opt[out_size + i] != 0x5a) {
            TEST_FAIL("avg_pool [%d] wrote past its output (guard byte %d) [in %dx%dx%d, out %dx%d]\n",
                      iter, i, input_wd, input_ht, channels, out_wd, out_ht);
            goto avg_pool_cleanup;
        }
    }
    bool ret = CHECK_EQUAL(output_c, output_opt, out_size);
    if (ret == false) {
        TEST_FAIL("avg_pool [%d] failed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d, act %d..%d]\n",
                  iter, input_wd, input_ht, channels, filter_wd, filter_ht,
                  stride_wd, stride_ht, pad_wd, pad_ht, (int) activation_min, (int) activation_max);
        goto avg_pool_cleanup;
    }
    TEST_PASS("avg_pool [%2d] passed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d]\n",
              iter, input_wd, input_ht, channels, filter_wd, filter_ht,
              stride_wd, stride_ht, pad_wd, pad_ht);

avg_pool_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
}

static void run_avg_pool_test(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                              uint16_t filter_wd, uint16_t filter_ht,
                              uint16_t stride_wd, uint16_t stride_ht,
                              uint16_t pad_wd, uint16_t pad_ht,
                              int iter)
{
    run_avg_pool_test_act(input_wd, input_ht, channels, filter_wd, filter_ht,
                          stride_wd, stride_ht, pad_wd, pad_ht, -128, 127, iter);
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
    /* Non-square outputs: the output row stride must use the output width.
     * Taller than wide also checks nothing is written past the output. */
    run_avg_pool_test(16, 8, 16, 3, 3, 1, 1, 1, 1, iter++);   /* 16x8, ch % 8 == 0 */
    run_avg_pool_test(8, 16, 16, 3, 3, 1, 1, 1, 1, iter++);   /* 8x16, ch % 8 == 0 */
    run_avg_pool_test(16, 8, 12, 3, 3, 1, 1, 1, 1, iter++);   /* 16x8, ch % 8 == 4 */
    run_avg_pool_test(8, 16, 12, 3, 3, 1, 1, 1, 1, iter++);   /* 8x16, ch % 8 == 4 */
    run_avg_pool_test(5, 24, 32, 2, 2, 2, 2, 0, 0, iter++);   /* 2x12 */
    /* Padding as wide as the filter: the first and last windows of each row
     * and column miss the input, and the output is 0, clamped. */
    run_avg_pool_test(6, 6, 16, 2, 2, 1, 1, 2, 2, iter++);                /* ch % 16 == 0 */
    run_avg_pool_test(6, 5, 12, 2, 2, 1, 1, 2, 2, iter++);                /* ch % 8 == 4 */
    run_avg_pool_test(1, 8, 16, 1, 2, 3, 1, 1, 0, iter++);                /* one column */
    run_avg_pool_test_act(6, 6, 20, 2, 2, 1, 1, 2, 2, 5, 30, iter++);     /* clamped to 5 */
}

/* Bytes after the max_pool output that must stay untouched */
#define MAX_POOL_GUARD 64

/* When set, every max pool input is at most 0 except the last pixel, which
 * is 127, so a window loop that stops early gets the wrong max. */
static bool max_pool_peak_last;

/* in_off and out_off put the input and outputs that many bytes past a
 * 16-byte boundary. out_wd and out_ht are given, so a test can ask for the
 * extra right/bottom output of TFLite SAME padding (pad_wd and pad_ht are the
 * left and top padding). */
static void run_max_pool_test_out(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                                  uint16_t filter_wd, uint16_t filter_ht,
                                  uint16_t stride_wd, uint16_t stride_ht,
                                  uint16_t pad_wd, uint16_t pad_ht,
                                  uint16_t out_wd, uint16_t out_ht,
                                  int32_t activation_min, int32_t activation_max,
                                  int in_off, int out_off, int iter)
{
    const int size = input_wd * input_ht * channels;
    const int out_size = out_wd * out_ht * channels;

    int8_t *input = NULL, *output_c = NULL, *output_opt = NULL;
    int8_t *input_orig = malloc(size + 32);
    int8_t *out_c_orig = malloc(out_size + 32);
    int8_t *out_opt_orig = malloc(out_size + 32 + MAX_POOL_GUARD);
    if (input_orig == NULL || out_c_orig == NULL || out_opt_orig == NULL) {
        TEST_SKIP("max_pool [%d] allocations failed\n", iter);
        goto max_pool_cleanup;
    }

    input = (int8_t *) ((((uint32_t) input_orig + 15) & ~15) + in_off);
    output_c = (int8_t *) ((((uint32_t) out_c_orig + 15) & ~15) + out_off);
    output_opt = (int8_t *) ((((uint32_t) out_opt_orig + 15) & ~15) + out_off);

    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256 - 128;
        if (max_pool_peak_last) {
            input[i] = i >= size - channels ? 127 : -(rand() % 129);
        }
    }
    memset(output_opt + out_size, 0x5a, MAX_POOL_GUARD);

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

    for (int i = 0; i < MAX_POOL_GUARD; ++i) {
        if ((uint8_t) output_opt[out_size + i] != 0x5a) {
            TEST_FAIL("max_pool [%d] wrote past its output (guard byte %d) [in %dx%dx%d, out %dx%d]\n",
                      iter, i, input_wd, input_ht, channels, out_wd, out_ht);
            goto max_pool_cleanup;
        }
    }
    bool ret = CHECK_EQUAL(output_c, output_opt, out_size);
    if (ret == false) {
        TEST_FAIL("max_pool [%d] failed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d, out %dx%d, "
                  "act %d..%d, off %d/%d]\n",
                  iter, input_wd, input_ht, channels, filter_wd, filter_ht,
                  stride_wd, stride_ht, pad_wd, pad_ht, out_wd, out_ht,
                  (int) activation_min, (int) activation_max, in_off, out_off);
        goto max_pool_cleanup;
    }
    TEST_PASS("max_pool [%2d] passed [in %dx%dx%d, f %dx%d, s %dx%d, p %dx%d, out %dx%d]\n",
              iter, input_wd, input_ht, channels, filter_wd, filter_ht,
              stride_wd, stride_ht, pad_wd, pad_ht, out_wd, out_ht);

max_pool_cleanup:
    if (input_orig) free(input_orig);
    if (out_c_orig) free(out_c_orig);
    if (out_opt_orig) free(out_opt_orig);
}

static void run_max_pool_test_buf(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                                  uint16_t filter_wd, uint16_t filter_ht,
                                  uint16_t stride_wd, uint16_t stride_ht,
                                  uint16_t pad_wd, uint16_t pad_ht,
                                  int32_t activation_min, int32_t activation_max,
                                  int in_off, int out_off, int iter)
{
    const uint16_t out_wd = (input_wd + 2 * pad_wd - filter_wd) / stride_wd + 1;
    const uint16_t out_ht = (input_ht + 2 * pad_ht - filter_ht) / stride_ht + 1;
    run_max_pool_test_out(input_wd, input_ht, channels, filter_wd, filter_ht,
                          stride_wd, stride_ht, pad_wd, pad_ht, out_wd, out_ht,
                          activation_min, activation_max, in_off, out_off, iter);
}

/* TFLite SAME padding: out = ceil(in / stride). The extra padding goes on the right and bottom. */
static void run_max_pool_test_same(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                                   uint16_t filter_wd, uint16_t filter_ht,
                                   uint16_t stride_wd, uint16_t stride_ht,
                                   int in_off, int out_off, int iter)
{
    const int out_wd = (input_wd + stride_wd - 1) / stride_wd;
    const int out_ht = (input_ht + stride_ht - 1) / stride_ht;
    int pad_wd_total = (out_wd - 1) * stride_wd + filter_wd - input_wd;
    int pad_ht_total = (out_ht - 1) * stride_ht + filter_ht - input_ht;
    pad_wd_total = pad_wd_total > 0 ? pad_wd_total : 0;
    pad_ht_total = pad_ht_total > 0 ? pad_ht_total : 0;
    run_max_pool_test_out(input_wd, input_ht, channels, filter_wd, filter_ht,
                          stride_wd, stride_ht, pad_wd_total / 2, pad_ht_total / 2,
                          out_wd, out_ht, -128, 127, in_off, out_off, iter);
}

static void run_max_pool_test_act(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                                  uint16_t filter_wd, uint16_t filter_ht,
                                  uint16_t stride_wd, uint16_t stride_ht,
                                  uint16_t pad_wd, uint16_t pad_ht,
                                  int32_t activation_min, int32_t activation_max,
                                  int iter)
{
    run_max_pool_test_buf(input_wd, input_ht, channels, filter_wd, filter_ht,
                          stride_wd, stride_ht, pad_wd, pad_ht,
                          activation_min, activation_max, 0, 0, iter);
}

static void run_max_pool_test(uint16_t input_wd, uint16_t input_ht, uint16_t channels,
                              uint16_t filter_wd, uint16_t filter_ht,
                              uint16_t stride_wd, uint16_t stride_ht,
                              uint16_t pad_wd, uint16_t pad_ht,
                              int iter)
{
    run_max_pool_test_act(input_wd, input_ht, channels, filter_wd, filter_ht,
                          stride_wd, stride_ht, pad_wd, pad_ht, -128, 127, iter);
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
    /* Fused activation must be applied on every channel path */
    run_max_pool_test_act(16, 16, 16, 3, 3, 1, 1, 1, 1, -20, 30, iter++); /* ch % 8 == 0 */
    run_max_pool_test_act(16, 16, 12, 3, 3, 1, 1, 1, 1, -20, 30, iter++); /* ch % 8 == 4 */
    run_max_pool_test_act(16, 16, 4, 2, 2, 2, 2, 0, 0, 0, 127, iter++);   /* ReLU, ch 4 */
    run_max_pool_test_act(9, 7, 20, 3, 3, 2, 2, 1, 1, -128, 6, iter++);   /* ch % 8 == 4 */
    /* Padding as wide as the filter: the first and last windows of each row
     * and column miss the input, and the output is the clamped INT8_MIN. */
    run_max_pool_test(6, 6, 16, 2, 2, 1, 1, 2, 2, iter++);                /* banded rows */
    run_max_pool_test(6, 5, 12, 2, 2, 1, 1, 2, 2, iter++);                /* ch % 8 == 4 */
    run_max_pool_test(1, 8, 16, 1, 2, 3, 1, 1, 0, iter++);                /* one column */
    run_max_pool_test_act(6, 6, 16, 2, 2, 1, 1, 2, 2, -20, 30, iter++);
    /* 2x2 with 16-lane funnel-shift loads: channels % 8 == 4, >= 20 */
    run_max_pool_test(12, 12, 20, 2, 2, 2, 2, 0, 0, iter++);
    run_max_pool_test(9, 9, 28, 2, 2, 1, 1, 0, 0, iter++);
    /* 3x3 with 16-lane funnel-shift loads: one chunk and tail 3, two chunks and tail 1 or 3 */
    run_max_pool_test(9, 9, 28, 3, 3, 1, 1, 0, 0, iter++);
    run_max_pool_test(9, 9, 36, 3, 3, 2, 2, 0, 0, iter++);
    run_max_pool_test(9, 9, 44, 3, 3, 1, 1, 0, 0, iter++);
    /* channels % 16 == 0 below 16-byte alignment: 64-bit fallbacks */
    run_max_pool_test_buf(12, 12, 16, 2, 2, 2, 2, 0, 0, -128, 127, 8, 8, iter++);
    run_max_pool_test_buf(12, 12, 32, 2, 2, 2, 2, 0, 0, -128, 127, 0, 8, iter++);
    run_max_pool_test_buf(10, 10, 16, 3, 3, 1, 1, 1, 1, -128, 127, 8, 0, iter++);
    run_max_pool_test_buf(10, 10, 48, 3, 3, 2, 2, 0, 0, -128, 127, 8, 8, iter++);
    /* 2x2 with in and out at 8 mod 16 and aligned blocks between the halves */
    run_max_pool_test_buf(12, 12, 32, 2, 2, 2, 2, 0, 0, -128, 127, 8, 8, iter++);
    run_max_pool_test_buf(12, 12, 48, 2, 2, 2, 2, 0, 0, -128, 127, 8, 8, iter++);
    run_max_pool_test_buf(10, 10, 32, 3, 3, 1, 1, 1, 1, -128, 127, 8, 8, iter++); /* 2x2 corners */
    /* activation limits outside int8 take the reference */
    run_max_pool_test_act(8, 8, 16, 2, 2, 2, 2, 0, 0, -200, 200, iter++);

    /* 1xK line filters on real input rows (1D models) */
    run_max_pool_test(16, 4, 16, 3, 1, 1, 1, 0, 0, iter++);               /* 1x3, flat band */
    run_max_pool_test(17, 3, 20, 4, 1, 2, 1, 1, 0, iter++);               /* 1x4 s2, clipped left */
    run_max_pool_test(12, 3, 40, 5, 1, 1, 1, 2, 0, iter++);               /* 1x5, 3 blocks */
    run_max_pool_test(13, 2, 32, 3, 1, 3, 1, 1, 0, iter++);               /* 1x3 s3 */
    run_max_pool_test_buf(15, 2, 8, 3, 1, 2, 1, 1, 0, -128, 127, 4, 12, iter++); /* ch < 16 */
    run_max_pool_test_buf(10, 2, 4, 3, 1, 1, 1, 1, 0, -128, 127, 12, 4, iter++); /* ch < 16, flat */
    /* 1x2 line filters */
    run_max_pool_test(12, 3, 16, 2, 1, 2, 1, 0, 0, iter++);               /* 2-tap band */
    run_max_pool_test(12, 3, 12, 2, 1, 1, 1, 0, 0, iter++);               /* flat */
    run_max_pool_test_buf(9, 2, 8, 2, 1, 2, 1, 0, 0, -128, 127, 4, 8, iter++);
    run_max_pool_test_same(11, 3, 24, 2, 1, 2, 1, 0, 0, iter++);          /* 1-tap right edge */
    /* Width-1 inputs with real data: pooled as one row */
    run_max_pool_test(1, 12, 16, 1, 3, 1, 2, 0, 1, iter++);               /* 3x1 s2, pad 1 */
    run_max_pool_test(1, 9, 8, 1, 2, 1, 1, 0, 0, iter++);                 /* 2x1, ch 8 */
    run_max_pool_test(1, 10, 20, 3, 3, 1, 1, 1, 1, iter++);               /* 3x3 clipped to 1 col */
    run_max_pool_test_buf(1, 11, 4, 1, 3, 1, 1, 0, 1, -128, 127, 4, 4, iter++);
    run_max_pool_test_same(1, 7, 16, 1, 2, 1, 2, 0, 0, iter++);           /* extra bottom output */
    /* TFLite SAME padding with the extra column and row on the right and bottom only */
    run_max_pool_test_same(7, 7, 16, 2, 2, 2, 2, 0, 0, iter++);           /* 2x2 s2, out 4x4 */
    run_max_pool_test_same(7, 5, 12, 2, 2, 2, 2, 4, 8, iter++);
    run_max_pool_test_same(8, 8, 16, 3, 3, 2, 2, 0, 0, iter++);           /* 3x3 s2, pad 0 */
    run_max_pool_test_same(8, 6, 40, 3, 3, 2, 2, 0, 0, iter++);           /* 3 blocks */
    run_max_pool_test_same(8, 8, 4, 3, 3, 2, 2, 12, 4, iter++);
    run_max_pool_test_same(7, 7, 8, 2, 2, 1, 1, 0, 0, iter++);            /* 2x2 s1 */
    run_max_pool_test_same(10, 9, 24, 3, 3, 2, 2, 8, 0, iter++);
    /* channels < 16 with unaligned buffers: 16-byte spill store and input-end guard */
    run_max_pool_test_buf(6, 6, 4, 2, 2, 2, 2, 0, 0, -128, 127, 4, 12, iter++);
    run_max_pool_test_buf(7, 5, 8, 2, 2, 1, 1, 0, 0, -128, 127, 12, 4, iter++);
    run_max_pool_test_buf(5, 5, 12, 3, 3, 1, 1, 1, 1, -128, 127, 8, 4, iter++);
    run_max_pool_test_buf(9, 7, 8, 3, 3, 2, 2, 1, 1, -128, 127, 4, 8, iter++);
    run_max_pool_test_buf(3, 3, 4, 2, 2, 1, 1, 0, 0, -128, 127, 12, 12, iter++);
    run_max_pool_test_buf(6, 4, 3, 3, 3, 1, 1, 1, 1, -128, 127, 1, 3, iter++); /* ch 3 */
    run_max_pool_test_buf(8, 3, 1, 2, 2, 2, 1, 0, 0, -128, 127, 7, 5, iter++); /* ch 1 */
    run_max_pool_test_buf(5, 5, 5, 4, 4, 1, 1, 2, 2, -128, 127, 3, 9, iter++); /* generic window */
    /* Generic clipped window with three blocks at 0, 16, d and 16 < d < 32 */
    run_max_pool_test(6, 6, 40, 3, 3, 1, 1, 1, 1, iter++);                /* d = 24 */
    run_max_pool_test(6, 6, 33, 5, 5, 1, 1, 2, 2, iter++);                /* d = 17 */
    run_max_pool_test(7, 5, 47, 3, 3, 2, 2, 1, 1, iter++);                /* d = 31 */
    run_max_pool_test(5, 5, 56, 4, 4, 1, 1, 1, 1, iter++);                /* 4 blocks, d = 24 */
    run_max_pool_test_buf(6, 6, 44, 5, 5, 1, 1, 2, 2, -128, 127, 8, 4, iter++);
#if CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_ESP32S31
    /* Bands and windows of more than 4095 hardware loop iterations, the
     * ESP32-P4 rev < v3 limit */
    run_max_pool_test(4200, 1, 16, 2, 1, 1, 1, 0, 0, iter++);       /* 1x2 flat band */
    run_max_pool_test(4200, 2, 16, 2, 2, 1, 1, 0, 0, iter++);       /* 2x2 flat band */
    run_max_pool_test(8200, 2, 16, 2, 2, 2, 2, 0, 0, iter++);       /* 2x2, 4100 pixels */
    run_max_pool_test(4102, 3, 16, 3, 3, 1, 1, 0, 0, iter++);       /* 3x3 flat band of 4100 blocks */
    max_pool_peak_last = true;
    run_max_pool_test(4200, 1, 48, 4200, 1, 1, 1, 0, 0, iter++);    /* 1x4200 window */
    /* 40 channels: 3 blocks, the last one overlapping the one before */
    run_max_pool_test(4200, 1, 40, 4200, 1, 1, 1, 0, 0, iter++);    /* 1x4200 window */
    max_pool_peak_last = false;
#endif
}
