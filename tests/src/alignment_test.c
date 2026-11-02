/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Layout sweeps for the S3 kernels: same geometry, every buffer address offset
 * by a different number of bytes. The kernels load with ee.vld.64/ee.vld.128,
 * which mask low address bits, and several derive SAR/USAR from one operand, so
 * a path that assumes alignment silently reads the wrong bytes instead of
 * faulting (esp-nn#34). Every case is compared against the ANSI reference and
 * the output carries guard bands on both sides.
 *
 * Pooling is swept at every offset: its assembly has an 8-byte path (channels
 * % 8 == 0) and a 4-byte path (% 8 == 4) and defers to the scalar/reference
 * path below that, and the wrapper uses the reference for channels % 4 != 0
 * (where the assembly used to write nothing).
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"
#include "test_align.h"

/* Offsets applied after 16-byte rounding. 1 and 4 cover the "no usable vector
 * alignment" cases; 8 and 12 the 8-byte-only ones. */
static const int ag_aligns[] = { 0, 1, 4, 8, 12 };
#define AG_N_ALIGNS ((int)(sizeof(ag_aligns) / sizeof(ag_aligns[0])))

/* Pooling offsets: 0/8 take the 8-byte path (channels % 8 == 0), 4/12 the
 * 4-byte path (% 8 == 4), and the odd ones the scalar/reference fallback. */
static const int ag_pool_aligns[] = { 0, 1, 4, 8, 12 };
#define AG_N_POOL_ALIGNS ((int)(sizeof(ag_pool_aligns) / sizeof(ag_pool_aligns[0])))

static bool ag_equal(const int8_t *a, const int8_t *b, int size)
{
    return memcmp(a, b, (size_t)size) == 0;
}

static bool ag_alloc3(struct test_align_buf *a, struct test_align_buf *b,
                      struct test_align_buf *c, int a_size, int b_size, int c_size,
                      int align)
{
    return test_align_buf_init(a, a_size, align) == 0 &&
           test_align_buf_init(b, b_size, align) == 0 &&
           test_align_buf_init(c, c_size, align) == 0;
}

/* ------------------------------------------------------------------ conv -- */

static void ag_conv_case(int in_wd, int in_ht, int in_ch, int out_ch,
                         int f_wd, int f_ht, int stride, int pad)
{
    const int out_wd = (in_wd + 2 * pad - f_wd) / stride + 1;
    const int out_ht = (in_ht + 2 * pad - f_ht) / stride + 1;
    const int in_size = in_wd * in_ht * in_ch;
    const int f_size = f_wd * f_ht * in_ch * out_ch;
    const int out_size = out_wd * out_ht * out_ch;
    const int32_t in_offset = 3, out_offset = -5;
    const int32_t activation_min = -128, activation_max = 127;

    data_dims_t input_dims = { .width = in_wd, .height = in_ht, .channels = in_ch, 1 };
    data_dims_t filter_dims = { .width = f_wd, .height = f_ht, .channels = in_ch, 1 };
    data_dims_t output_dims = { .width = out_wd, .height = out_ht, .channels = out_ch, 1 };
    conv_params_t conv_params = {
        .in_offset = in_offset, .out_offset = out_offset,
        .stride = { stride, stride }, .padding = { pad, pad },
        .dilation = { 0, 0 },
        .activation = { activation_min, activation_max },
    };

    int32_t *bias = malloc(sizeof(int32_t) * out_ch);
    int32_t *mult = malloc(sizeof(int32_t) * out_ch);
    int32_t *shift = malloc(sizeof(int32_t) * out_ch);
    struct test_align_buf scratch = { 0 };
    if (bias == NULL || mult == NULL || shift == NULL) {
        TEST_SKIP("conv align: allocation failed\n");
        free(bias); free(mult); free(shift);
        return;
    }
    for (int i = 0; i < out_ch; i++) {
        bias[i] = (rand() % 2000) - 1000;
        mult[i] = 0x40000000 + i;
        shift[i] = -7;
    }
    quant_data_t quant_data = { .shift = shift, .mult = mult };

    const int scratch_size = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                                          &output_dims, &conv_params);
    if (scratch_size > 0 && test_align_buf_init(&scratch, scratch_size, 0) != 0) {
        TEST_SKIP("conv align: scratch allocation failed\n");
        free(bias); free(mult); free(shift);
        return;
    }
    if (scratch_size > 0) {
        esp_nn_set_conv_scratch_buf(scratch.data);
    }

    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_ALIGNS; ai++) {
        for (int fi = 0; fi < AG_N_ALIGNS; fi++) {
            struct test_align_buf in = { 0 }, flt = { 0 }, out_c = { 0 }, out_o = { 0 };
            bool ok = test_align_buf_init(&in, in_size, ag_aligns[ai]) == 0 &&
                      test_align_buf_init(&flt, f_size, ag_aligns[fi]) == 0 &&
                      test_align_buf_init(&out_c, out_size, 0) == 0 &&
                      test_align_buf_init(&out_o, out_size, ag_aligns[ai]) == 0;
            if (!ok) {
                TEST_SKIP("conv align: allocation failed\n");
                test_align_buf_free(&in); test_align_buf_free(&flt);
                test_align_buf_free(&out_c); test_align_buf_free(&out_o);
                goto conv_done;
            }
            test_align_fill_random(in.data, in_size);
            test_align_fill_random(flt.data, f_size);

            esp_nn_conv_s8_ansi(&input_dims, in.data, &filter_dims, flt.data, bias,
                                &output_dims, out_c.data, &conv_params, &quant_data);
            esp_nn_conv_s8(&input_dims, in.data, &filter_dims, flt.data, bias,
                           &output_dims, out_o.data, &conv_params, &quant_data);

            if (!ag_equal(out_c.data, out_o.data, out_size) || !test_align_guard_ok(&out_o)) {
                if (fail == 0) {
                    printf("conv align: in %dx%d ch %d/%d f %dx%d s %d p %d, "
                           "align %d/%d mismatch\n", in_wd, in_ht, in_ch, out_ch,
                           f_wd, f_ht, stride, pad, ag_aligns[ai], ag_aligns[fi]);
                }
                fail++;
            } else {
                pass++;
            }
            test_align_buf_free(&in); test_align_buf_free(&flt);
            test_align_buf_free(&out_c); test_align_buf_free(&out_o);
        }
    }
conv_done:
    esp_nn_set_conv_scratch_buf(NULL);
    test_align_buf_free(&scratch);
    free(bias); free(mult); free(shift);

    if (fail) {
        TEST_FAIL("conv align [%dx%d in %d out %d f %dx%d s %d p %d]: %d/%d failed\n",
                  in_wd, in_ht, in_ch, out_ch, f_wd, f_ht, stride, pad, fail,
                  pass + fail);
    } else {
        TEST_PASS("conv align [%dx%d in %d out %d f %dx%d s %d p %d]: %d passed\n",
                  in_wd, in_ht, in_ch, out_ch, f_wd, f_ht, stride, pad, pass);
    }
}

/* ------------------------------------------------------------- depthwise -- */

static void ag_dw_case(int in_wd, int in_ht, int channels, int ch_mult,
                       int f_wd, int f_ht, int stride, int pad)
{
    const int out_ch = channels * ch_mult;
    const int out_wd = (in_wd + 2 * pad - f_wd) / stride + 1;
    const int out_ht = (in_ht + 2 * pad - f_ht) / stride + 1;
    const int in_size = in_wd * in_ht * channels;
    const int f_size = f_wd * f_ht * channels * ch_mult;
    const int out_size = out_wd * out_ht * out_ch;
    const int32_t in_offset = 3, out_offset = -5;
    const int32_t activation_min = -128, activation_max = 127;

    data_dims_t input_dims = { .width = in_wd, .height = in_ht, .channels = channels, 1 };
    data_dims_t filter_dims = { .width = f_wd, .height = f_ht, .channels = 0, 0 };
    data_dims_t output_dims = { .width = out_wd, .height = out_ht, .channels = out_ch, 1 };
    dw_conv_params_t conv_params = {
        .in_offset = in_offset, .out_offset = out_offset, .ch_mult = ch_mult,
        .stride = { stride, stride }, .padding = { pad, pad },
        .dilation = { 0, 0 },
        .activation = { activation_min, activation_max },
    };

    int32_t *bias = malloc(sizeof(int32_t) * out_ch);
    int32_t *mult = malloc(sizeof(int32_t) * out_ch);
    int32_t *shift = malloc(sizeof(int32_t) * out_ch);
    struct test_align_buf scratch = { 0 };
    if (bias == NULL || mult == NULL || shift == NULL) {
        TEST_SKIP("depthwise align: allocation failed\n");
        free(bias); free(mult); free(shift);
        return;
    }
    for (int i = 0; i < out_ch; i++) {
        bias[i] = (rand() % 2000) - 1000;
        mult[i] = 0x40000000 + i;
        shift[i] = -7;
    }
    quant_data_t quant_data = { .shift = shift, .mult = mult };

    const int scratch_size = esp_nn_get_depthwise_conv_scratch_size(
            &input_dims, &filter_dims, &output_dims, &conv_params);
    if (scratch_size > 0 && test_align_buf_init(&scratch, scratch_size, 0) != 0) {
        TEST_SKIP("depthwise align: scratch allocation failed\n");
        free(bias); free(mult); free(shift);
        return;
    }
    if (scratch_size > 0) {
        esp_nn_set_depthwise_conv_scratch_buf(scratch.data);
    }

    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_ALIGNS; ai++) {
        for (int fi = 0; fi < AG_N_ALIGNS; fi++) {
            struct test_align_buf in = { 0 }, flt = { 0 }, out_c = { 0 }, out_o = { 0 };
            bool ok = test_align_buf_init(&in, in_size, ag_aligns[ai]) == 0 &&
                      test_align_buf_init(&flt, f_size, ag_aligns[fi]) == 0 &&
                      test_align_buf_init(&out_c, out_size, 0) == 0 &&
                      test_align_buf_init(&out_o, out_size, ag_aligns[ai]) == 0;
            if (!ok) {
                TEST_SKIP("depthwise align: allocation failed\n");
                test_align_buf_free(&in); test_align_buf_free(&flt);
                test_align_buf_free(&out_c); test_align_buf_free(&out_o);
                goto dw_done;
            }
            test_align_fill_random(in.data, in_size);
            test_align_fill_random(flt.data, f_size);

            esp_nn_depthwise_conv_s8_ansi(&input_dims, in.data, &filter_dims, flt.data,
                                          bias, &output_dims, out_c.data, &conv_params,
                                          &quant_data);
            esp_nn_depthwise_conv_s8(&input_dims, in.data, &filter_dims, flt.data,
                                     bias, &output_dims, out_o.data, &conv_params,
                                     &quant_data);

            if (!ag_equal(out_c.data, out_o.data, out_size) || !test_align_guard_ok(&out_o)) {
                if (fail == 0) {
                    printf("depthwise align: %dx%d ch %d x%d f %dx%d s %d p %d, "
                           "align %d/%d mismatch\n", in_wd, in_ht, channels, ch_mult,
                           f_wd, f_ht, stride, pad, ag_aligns[ai], ag_aligns[fi]);
                }
                fail++;
            } else {
                pass++;
            }
            test_align_buf_free(&in); test_align_buf_free(&flt);
            test_align_buf_free(&out_c); test_align_buf_free(&out_o);
        }
    }
dw_done:
    esp_nn_set_depthwise_conv_scratch_buf(NULL);
    test_align_buf_free(&scratch);
    free(bias); free(mult); free(shift);

    if (fail) {
        TEST_FAIL("depthwise align [%dx%d ch %d x%d f %dx%d s %d p %d]: %d/%d failed\n",
                  in_wd, in_ht, channels, ch_mult, f_wd, f_ht, stride, pad, fail,
                  pass + fail);
    } else {
        TEST_PASS("depthwise align [%dx%d ch %d x%d f %dx%d s %d p %d]: %d passed\n",
                  in_wd, in_ht, channels, ch_mult, f_wd, f_ht, stride, pad, pass);
    }
}

/* --------------------------------------------------------------- pooling -- */

static void ag_pool_case(bool avg, int in_wd, int in_ht, int channels,
                         int filter, int stride, int pad)
{
    const int out_wd = (in_wd + 2 * pad - filter) / stride + 1;
    const int out_ht = (in_ht + 2 * pad - filter) / stride + 1;
    const int in_size = in_wd * in_ht * channels;
    const int out_size = out_wd * out_ht * channels;
    const int32_t activation_min = -128, activation_max = 127;

    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_POOL_ALIGNS; ai++) {
        struct test_align_buf in = { 0 }, out_c = { 0 }, out_o = { 0 };
        bool ok = ag_alloc3(&in, &out_c, &out_o, in_size, out_size, out_size,
                            ag_pool_aligns[ai]);
        if (!ok) {
            TEST_SKIP("pool align: allocation failed\n");
            test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
            return;
        }
        test_align_fill_random(in.data, in_size);

        if (avg) {
            esp_nn_avg_pool_s8_ansi(in.data, in_wd, in_ht, out_c.data, out_wd, out_ht,
                                    stride, stride, filter, filter, pad, pad,
                                    activation_min, activation_max, channels);
            esp_nn_avg_pool_s8(in.data, in_wd, in_ht, out_o.data, out_wd, out_ht,
                               stride, stride, filter, filter, pad, pad,
                               activation_min, activation_max, channels);
        } else {
            esp_nn_max_pool_s8_ansi(in.data, in_wd, in_ht, out_c.data, out_wd, out_ht,
                                    stride, stride, filter, filter, pad, pad,
                                    activation_min, activation_max, channels);
            esp_nn_max_pool_s8(in.data, in_wd, in_ht, out_o.data, out_wd, out_ht,
                               stride, stride, filter, filter, pad, pad,
                               activation_min, activation_max, channels);
        }

        if (!ag_equal(out_c.data, out_o.data, out_size) || !test_align_guard_ok(&out_o)) {
            fail++;
            printf("%s align: %dx%d ch %d f %d s %d p %d, offset %d mismatch\n",
                   avg ? "avg_pool" : "max_pool", in_wd, in_ht, channels, filter,
                   stride, pad, ag_pool_aligns[ai]);
        } else {
            pass++;
        }
        test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
    }

    if (fail) {
        TEST_FAIL("%s align [%dx%d ch %d f %d s %d p %d]: %d/%d failed\n",
                  avg ? "avg_pool" : "max_pool", in_wd, in_ht, channels, filter,
                  stride, pad, fail, pass + fail);
    } else {
        TEST_PASS("%s align [%dx%d ch %d f %d s %d p %d]: %d passed\n",
                  avg ? "avg_pool" : "max_pool", in_wd, in_ht, channels, filter,
                  stride, pad, pass);
    }
}

/* ------------------------------------------------------------------ mean -- */

static void ag_mean_case(int height, int width, int channels)
{
    const int in_size = height * width * channels;
    const int out_size = channels;
    const int32_t in_zp = 3, out_zp = -5, mult = 0x40000000, shift = -10;

    struct test_align_buf scratch = { 0 };
    const int scratch_size = esp_nn_get_mean_scratch_size(height, width, channels);
    if (scratch_size > 0 && test_align_buf_init(&scratch, scratch_size, 0) != 0) {
        TEST_SKIP("mean align: scratch allocation failed\n");
        return;
    }

    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_ALIGNS; ai++) {
        if (scratch_size > 0) {
            esp_nn_set_mean_scratch_buf(scratch.data);
        }
        struct test_align_buf in = { 0 }, out_c = { 0 }, out_o = { 0 };
        bool ok = ag_alloc3(&in, &out_c, &out_o, in_size, out_size, out_size,
                            ag_aligns[ai]);
        if (!ok) {
            TEST_SKIP("mean align: allocation failed\n");
            test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
            goto mean_done;
        }
        test_align_fill_random(in.data, in_size);

        esp_nn_mean_nhwc_s8_ansi(in.data, out_c.data, height, width, channels,
                                 in_zp, out_zp, mult, shift);
        esp_nn_mean_nhwc_s8(in.data, out_o.data, height, width, channels,
                            in_zp, out_zp, mult, shift);

        if (!ag_equal(out_c.data, out_o.data, out_size) || !test_align_guard_ok(&out_o)) {
            fail++;
            printf("mean align: %dx%d ch %d, offset %d mismatch\n", height, width,
                   channels, ag_aligns[ai]);
        } else {
            pass++;
        }
        test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
    }
mean_done:
    esp_nn_set_mean_scratch_buf(NULL);
    test_align_buf_free(&scratch);

    if (fail) {
        TEST_FAIL("mean align [%dx%d ch %d]: %d/%d failed\n", height, width, channels,
                  fail, pass + fail);
    } else {
        TEST_PASS("mean align [%dx%d ch %d]: %d passed\n", height, width, channels, pass);
    }
}

/* --------------------------------------------------------------- softmax -- */

static void ag_softmax_case(int height, int width)
{
    const int size = height * width;
    const int32_t mult = INT32_MAX / 2, shift = 7, diff_min = -64;

    struct test_align_buf scratch = { 0 };
    /* The getter returns elements, as in the existing softmax test. */
    const int scratch_elems = esp_nn_get_softmax_scratch_size(width, height);
    if (scratch_elems > 0 &&
        test_align_buf_init(&scratch, scratch_elems * (int)sizeof(int32_t) + 16, 0) != 0) {
        TEST_SKIP("softmax align: scratch allocation failed\n");
        return;
    }

    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_ALIGNS; ai++) {
        if (scratch_elems > 0) {
            esp_nn_set_softmax_scratch_buf(scratch.data);
        }
        struct test_align_buf in = { 0 }, out_c = { 0 }, out_o = { 0 };
        bool ok = ag_alloc3(&in, &out_c, &out_o, size, size, size, ag_aligns[ai]);
        if (!ok) {
            TEST_SKIP("softmax align: allocation failed\n");
            test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
            goto sm_done;
        }
        test_align_fill_random(in.data, size);

        esp_nn_softmax_s8_ansi(in.data, height, width, mult, shift, diff_min, out_c.data);
        esp_nn_softmax_s8(in.data, height, width, mult, shift, diff_min, out_o.data);

        if (!ag_equal(out_c.data, out_o.data, size) || !test_align_guard_ok(&out_o)) {
            fail++;
            printf("softmax align: %dx%d, offset %d mismatch\n", height, width,
                   ag_aligns[ai]);
        } else {
            pass++;
        }
        test_align_buf_free(&in); test_align_buf_free(&out_c); test_align_buf_free(&out_o);
    }
sm_done:
    esp_nn_set_softmax_scratch_buf(NULL);
    test_align_buf_free(&scratch);

    if (fail) {
        TEST_FAIL("softmax align [%dx%d]: %d/%d failed\n", height, width, fail, pass + fail);
    } else {
        TEST_PASS("softmax align [%dx%d]: %d passed\n", height, width, pass);
    }
}

/* ---------------------------------------------------------------- relu6 -- */

static void ag_relu6_case(int size)
{
    int pass = 0, fail = 0;
    for (int ai = 0; ai < AG_N_ALIGNS; ai++) {
        struct test_align_buf a = { 0 }, b = { 0 };
        bool ok = test_align_buf_init(&a, size, ag_aligns[ai]) == 0 &&
                  test_align_buf_init(&b, size, ag_aligns[ai]) == 0;
        if (!ok) {
            TEST_SKIP("relu6 align: allocation failed\n");
            test_align_buf_free(&a); test_align_buf_free(&b);
            return;
        }
        test_align_fill_random(a.data, size);
        memcpy(b.data, a.data, (size_t)size);

        esp_nn_relu6_s8_ansi(a.data, (uint16_t)size);
        esp_nn_relu6_s8(b.data, (uint16_t)size);

        if (!ag_equal(a.data, b.data, size) || !test_align_guard_ok(&b)) {
            fail++;
            printf("relu6 align: size %d, offset %d mismatch\n", size, ag_aligns[ai]);
        } else {
            pass++;
        }
        test_align_buf_free(&a); test_align_buf_free(&b);
    }
    if (fail) {
        TEST_FAIL("relu6 align [%d]: %d/%d failed\n", size, fail, pass + fail);
    } else {
        TEST_PASS("relu6 align [%d]: %d passed\n", size, pass);
    }
}

/* -------------------------------------------------- elementwise add/mul -- */

static void ag_elementwise_case(bool mul, int size)
{
    static const int pairs[][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 4, 12 }, { 15, 8 } };
    const int n_pairs = (int)(sizeof(pairs) / sizeof(pairs[0]));
    const int32_t in1_offset = 2, in2_offset = -3, out_offset = 5;
    const int32_t in1_mult = 0x40000000, in2_mult = 0x40000000;
    const int32_t in1_shift = -10, in2_shift = -10, left_shift = 0;
    const int32_t out_mult = 0x40000000, out_shift = -10;
    const int32_t activation_min = -128, activation_max = 127;

    int pass = 0, fail = 0;
    for (int i = 0; i < n_pairs; i++) {
        struct test_align_buf in1 = { 0 }, in2 = { 0 }, out_c = { 0 }, out_o = { 0 };
        bool ok = test_align_buf_init(&in1, size, pairs[i][0]) == 0 &&
                  test_align_buf_init(&in2, size, pairs[i][1]) == 0 &&
                  test_align_buf_init(&out_c, size, pairs[i][0]) == 0 &&
                  test_align_buf_init(&out_o, size, pairs[i][0]) == 0;
        if (!ok) {
            TEST_SKIP("elementwise align: allocation failed\n");
            test_align_buf_free(&in1); test_align_buf_free(&in2);
            test_align_buf_free(&out_c); test_align_buf_free(&out_o);
            return;
        }
        test_align_fill_random(in1.data, size);
        test_align_fill_random(in2.data, size);

        if (mul) {
            esp_nn_mul_elementwise_s8_ansi(in1.data, in2.data, in1_offset, in2_offset,
                                           out_c.data, out_offset, out_mult, out_shift,
                                           activation_min, activation_max, size);
            esp_nn_mul_elementwise_s8(in1.data, in2.data, in1_offset, in2_offset,
                                      out_o.data, out_offset, out_mult, out_shift,
                                      activation_min, activation_max, size);
        } else {
            esp_nn_add_elementwise_s8_ansi(in1.data, in2.data, in1_offset, in2_offset,
                                           in1_mult, in2_mult, in1_shift, in2_shift,
                                           left_shift, out_c.data, out_offset, out_mult,
                                           out_shift, activation_min, activation_max, size);
            esp_nn_add_elementwise_s8(in1.data, in2.data, in1_offset, in2_offset,
                                      in1_mult, in2_mult, in1_shift, in2_shift,
                                      left_shift, out_o.data, out_offset, out_mult,
                                      out_shift, activation_min, activation_max, size);
        }

        if (!ag_equal(out_c.data, out_o.data, size) || !test_align_guard_ok(&out_o)) {
            fail++;
            printf("%s align: size %d, offsets %d/%d mismatch\n", mul ? "mul" : "add",
                   size, pairs[i][0], pairs[i][1]);
        } else {
            pass++;
        }
        test_align_buf_free(&in1); test_align_buf_free(&in2);
        test_align_buf_free(&out_c); test_align_buf_free(&out_o);
    }
    if (fail) {
        TEST_FAIL("%s align [%d]: %d/%d failed\n", mul ? "mul" : "add", size, fail,
                  pass + fail);
    } else {
        TEST_PASS("%s align [%d]: %d passed\n", mul ? "mul" : "add", size, pass);
    }
}

/* --------------------------------------------------------------- driver -- */

void esp_nn_alignment_s8_test(void)
{
    printf("\n######## Running %s ##########\n", __FUNCTION__);

    /* conv: 1x1, 3x3 SAME, 3x3 stride 2 */
    ag_conv_case(5, 5, 8, 8, 1, 1, 1, 0);
    ag_conv_case(5, 5, 16, 8, 1, 1, 1, 0);
    ag_conv_case(5, 5, 8, 8, 3, 3, 1, 1);
    ag_conv_case(6, 5, 8, 16, 3, 3, 2, 1);

    /* depthwise: multiplier 1 (s8 and s16 paths) and 8 */
    ag_dw_case(5, 5, 8, 1, 3, 3, 1, 1);
    ag_dw_case(5, 5, 16, 1, 3, 3, 1, 1);
    ag_dw_case(5, 5, 16, 1, 3, 3, 1, 0);
    ag_dw_case(5, 5, 8, 8, 3, 3, 1, 1);
    ag_dw_case(5, 5, 6, 2, 3, 3, 1, 1);

    /* pooling: 8-byte path (ch % 8 == 0), 4-byte path (ch % 8 == 4), the
     * scalar avg_pool path for channels % 4 != 0, and max_pool with
     * channels % 4 != 0, which the wrapper now sends to the reference
     * (the assembly used to write nothing) */
    ag_pool_case(true, 5, 5, 8, 3, 1, 0);
    ag_pool_case(true, 5, 5, 12, 3, 2, 1);
    ag_pool_case(true, 5, 5, 6, 3, 1, 0);
    ag_pool_case(false, 5, 5, 8, 3, 1, 0);
    ag_pool_case(false, 5, 5, 12, 3, 2, 1);
    ag_pool_case(false, 5, 5, 6, 3, 1, 0);

    /* scalar-reference kernels */
    ag_mean_case(4, 4, 8);
    ag_softmax_case(1, 16);
    ag_softmax_case(2, 36);
    ag_relu6_case(16);
    ag_relu6_case(33);

    /* elementwise */
    ag_elementwise_case(false, 16);
    ag_elementwise_case(false, 33);
    ag_elementwise_case(true, 16);
    ag_elementwise_case(true, 33);
}
