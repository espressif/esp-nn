/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdio.h>

#include <esp_nn.h>

#include "bench_ops.h"

/* --------------------------------------------------------------- Softmax -- */

typedef struct {
    int height, width;
    const int8_t *input[2];
} softmax_ctx_t;

/* TFLite's prepare for input scale 1/16, beta 1: multiplier 2^30, left
 * shift 23, diff_min -248. The exp argument spans [-15.5, 0]. */
#define SOFTMAX_MULT     (1 << 30)
#define SOFTMAX_SHIFT    23
#define SOFTMAX_DIFF_MIN -248

static void softmax_ref(void *p, int v, int8_t *out)
{
    const softmax_ctx_t *c = p;
    esp_nn_softmax_s8_ansi(c->input[v], c->height, c->width, SOFTMAX_MULT, SOFTMAX_SHIFT,
                           SOFTMAX_DIFF_MIN, out);
}

static void softmax_opt(void *p, int v, int8_t *out)
{
    const softmax_ctx_t *c = p;
    esp_nn_softmax_s8(c->input[v], c->height, c->width, SOFTMAX_MULT, SOFTMAX_SHIFT,
                      SOFTMAX_DIFF_MIN, out);
}

static const bench_variant_t s_softmax_variants[] = {
    { "esp_nn", softmax_opt },
};

/* rows x classes */
static const struct { int height, width; } s_softmax_shapes[] = {
    { 8, 32 },
    { 1, 256 },
    { 1, 1000 },
};

static void run_softmax(int height, int width)
{
    const size_t size = (size_t) height * width;
    int8_t *in[2] = { bench_alloc(size, BENCH_MEM_INT), bench_alloc(size, BENCH_MEM_INT) };
    const int scratch_size = esp_nn_get_softmax_scratch_size(width, height);
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, BENCH_MEM_INT) : NULL;
    char shape[32];
    snprintf(shape, sizeof(shape), "height %d, width %d", height, width);

    if (in[0] && in[1] && (scratch_size <= 0 || scratch)) {
        bench_seed_case("softmax", shape);
        bench_fill(in[0], size);
        bench_fill(in[1], size);
        esp_nn_set_softmax_scratch_buf(scratch);
        softmax_ctx_t c = { .height = height, .width = width, .input = { in[0], in[1] } };
        const bench_case_t bc = {
            .op = "softmax", .shape = shape, .mem = BENCH_MEM_INT, .out_size = size,
            .ref = softmax_ref, .variants = s_softmax_variants,
            .n_variants = sizeof(s_softmax_variants) / sizeof(s_softmax_variants[0]), .ctx = &c,
        };
        bench_run(&bc);
        esp_nn_set_softmax_scratch_buf(NULL);
    } else {
        bench_skip("softmax", shape, BENCH_MEM_INT);
    }
    bench_free(in[0]);
    bench_free(in[1]);
    bench_free(scratch);
}

void bench_softmax(void)
{
    for (int i = 0; i < (int) (sizeof(s_softmax_shapes) / sizeof(s_softmax_shapes[0])); i++) {
        run_softmax(s_softmax_shapes[i].height, s_softmax_shapes[i].width);
    }
}

/* ------------------------------------------------------------------ Mean -- */

typedef struct {
    int height, width, channels;
    int32_t mult, shift;
    const int8_t *input[2];
} mean_ctx_t;

/* Zero points -128; the multiplier and shift scale the sum by 1 / (H * W). */
#define MEAN_ARGS(c, in, out) (in), (out), (c)->height, (c)->width, (c)->channels, \
    -128, -128, (c)->mult, (c)->shift

/* 1 / n as a Q31 multiplier in [2^30, 2^31) and a shift, as TFLite's
 * QuantizeMultiplier computes it. */
static void mean_quant(int n, int32_t *mult, int32_t *shift)
{
    int e;
    const double q = frexp(1.0 / n, &e);
    int64_t m = llround(q * 2147483648.0);
    if (m == (1LL << 31)) {
        m /= 2;
        e++;
    }
    *mult = (int32_t) m;
    *shift = e;
}

static void mean_ref(void *p, int v, int8_t *out)
{
    const mean_ctx_t *c = p;
    esp_nn_mean_nhwc_s8_ansi(MEAN_ARGS(c, c->input[v], out));
}

static void mean_opt(void *p, int v, int8_t *out)
{
    const mean_ctx_t *c = p;
    esp_nn_mean_nhwc_s8(MEAN_ARGS(c, c->input[v], out));
}

static const bench_variant_t s_mean_variants[] = {
    { "esp_nn", mean_opt },
};

/* height x width x channels: squeeze-and-excite blocks of MobileNetV3. */
static const struct { int height, width, channels; } s_mean_shapes[] = {
    {  7,  7, 16 },
    { 14, 14, 120 },
};

static void run_mean(int height, int width, int channels)
{
    const size_t size = (size_t) height * width * channels;
    int8_t *in[2] = { bench_alloc(size, BENCH_MEM_INT), bench_alloc(size, BENCH_MEM_INT) };
    const int scratch_size = esp_nn_get_mean_scratch_size(height, width, channels);
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, BENCH_MEM_INT) : NULL;
    char shape[40];
    snprintf(shape, sizeof(shape), "input %dx%dx%d", height, width, channels);

    if (in[0] && in[1] && (scratch_size <= 0 || scratch)) {
        bench_seed_case("mean", shape);
        bench_fill(in[0], size);
        bench_fill(in[1], size);
        esp_nn_set_mean_scratch_buf(scratch);
        mean_ctx_t c = { .height = height, .width = width, .channels = channels,
                         .input = { in[0], in[1] } };
        mean_quant(height * width, &c.mult, &c.shift);
        const bench_case_t bc = {
            .op = "mean", .shape = shape, .mem = BENCH_MEM_INT, .out_size = channels,
            .ref = mean_ref, .variants = s_mean_variants,
            .n_variants = sizeof(s_mean_variants) / sizeof(s_mean_variants[0]), .ctx = &c,
        };
        bench_run(&bc);
        esp_nn_set_mean_scratch_buf(NULL);
    } else {
        bench_skip("mean", shape, BENCH_MEM_INT);
    }
    bench_free(in[0]);
    bench_free(in[1]);
    bench_free(scratch);
}

void bench_mean(void)
{
    for (int i = 0; i < (int) (sizeof(s_mean_shapes) / sizeof(s_mean_shapes[0])); i++) {
        run_mean(s_mean_shapes[i].height, s_mean_shapes[i].width, s_mean_shapes[i].channels);
    }
}
