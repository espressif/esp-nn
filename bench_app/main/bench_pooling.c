/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>

#include <esp_nn.h>

#include "bench_ops.h"

typedef struct {
    int in_wd, in_ht, ch, f_wd, f_ht, stride, pad;
    bench_mem_t mem;
} pool_shape_t;

/* The shapes of the README tables. */
static const pool_shape_t s_shapes[] = {
    { 16, 16, 16, 3, 3, 1, 1, BENCH_MEM_INT },
    /* MaxPool 5x5 of YOLO11n's SPPF block. */
    { 12, 12, 128, 5, 5, 1, 2, BENCH_MEM_EXT },
};

typedef struct {
    const pool_shape_t *s;
    int out_wd, out_ht;
    const int8_t *input[2];
} pool_ctx_t;

#define POOL_ARGS(c, in, out) (in), (c)->s->in_wd, (c)->s->in_ht, (out), (c)->out_wd, \
    (c)->out_ht, (c)->s->stride, (c)->s->stride, (c)->s->f_wd, (c)->s->f_ht, (c)->s->pad, \
    (c)->s->pad, -128, 127, (c)->s->ch

static void max_ref(void *p, int v, int8_t *out)
{
    const pool_ctx_t *c = p;
    esp_nn_max_pool_s8_ansi(POOL_ARGS(c, c->input[v], out));
}

static void max_opt(void *p, int v, int8_t *out)
{
    const pool_ctx_t *c = p;
    esp_nn_max_pool_s8(POOL_ARGS(c, c->input[v], out));
}

static void avg_ref(void *p, int v, int8_t *out)
{
    const pool_ctx_t *c = p;
    esp_nn_avg_pool_s8_ansi(POOL_ARGS(c, c->input[v], out));
}

static void avg_opt(void *p, int v, int8_t *out)
{
    const pool_ctx_t *c = p;
    esp_nn_avg_pool_s8(POOL_ARGS(c, c->input[v], out));
}

static const bench_variant_t s_max_variants[] = {
    { "esp_nn", max_opt },
};

static const bench_variant_t s_avg_variants[] = {
    { "esp_nn", avg_opt },
};

static void run_shape(const pool_shape_t *s)
{
    const size_t in_size = (size_t) s->in_wd * s->in_ht * s->ch;
    pool_ctx_t c = {
        .s = s,
        .out_wd = (s->in_wd + 2 * s->pad - s->f_wd) / s->stride + 1,
        .out_ht = (s->in_ht + 2 * s->pad - s->f_ht) / s->stride + 1,
    };
    int8_t *in[2] = { bench_alloc(in_size, s->mem), bench_alloc(in_size, s->mem) };
    char shape[80];
    snprintf(shape, sizeof(shape), "input %dx%dx%d, filter %dx%d, pad %d, stride %d",
             s->in_wd, s->in_ht, s->ch, s->f_wd, s->f_ht, s->pad, s->stride);

    if (in[0] && in[1]) {
        bench_seed_case("pool", shape);
        bench_fill(in[0], in_size);
        bench_fill(in[1], in_size);
        c.input[0] = in[0];
        c.input[1] = in[1];
        const size_t out_size = (size_t) c.out_wd * c.out_ht * s->ch;
        const bench_case_t max_case = {
            .op = "max_pool", .shape = shape, .mem = s->mem, .out_size = out_size,
            .ref = max_ref, .variants = s_max_variants,
            .n_variants = sizeof(s_max_variants) / sizeof(s_max_variants[0]), .ctx = &c,
        };
        const bench_case_t avg_case = {
            .op = "avg_pool", .shape = shape, .mem = s->mem, .out_size = out_size,
            .ref = avg_ref, .variants = s_avg_variants,
            .n_variants = sizeof(s_avg_variants) / sizeof(s_avg_variants[0]), .ctx = &c,
        };
        bench_run(&max_case);
        bench_run(&avg_case);
    } else {
        bench_skip("max_pool", shape, s->mem);
        bench_skip("avg_pool", shape, s->mem);
    }
    bench_free(in[0]);
    bench_free(in[1]);
}

void bench_pooling(void)
{
    for (int i = 0; i < (int) (sizeof(s_shapes) / sizeof(s_shapes[0])); i++) {
        run_shape(&s_shapes[i]);
    }
}
