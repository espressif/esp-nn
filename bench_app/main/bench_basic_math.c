/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>

#include <esp_nn.h>

#include "bench_ops.h"

/* Two-input elementwise ops share one context: input1 has `size` elements;
 * input2 has `size` elements, or `channels` for the channel broadcast. */
typedef struct {
    int size, positions, channels;
    const int8_t *in1[2], *in2[2];
} math_ctx_t;

/* ------------------------------------------------------------------- Add -- */

#define ADD_ARGS(in1, in2, out, n) (in1), (in2), 34, 35, INT32_MAX, INT32_MAX, -8, -8, 15, \
                                   (out), 36, INT32_MAX, -9, -128, 127, (n)

static void add_ref(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_add_elementwise_s8_ansi(ADD_ARGS(c->in1[v], c->in2[v], out, c->size));
}

static void add_opt(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_add_elementwise_s8(ADD_ARGS(c->in1[v], c->in2[v], out, c->size));
}

static const bench_variant_t s_add_variants[] = {
    { "esp_nn", add_opt },
};

/* ------------------------------------------------------------------- Mul -- */

/* out_offset, out_mult, out_shift, activation_min, activation_max */
#define MUL_QUANT -5, 0x5a827999, -8, -128, 127

static void mul_ref(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_mul_elementwise_s8_ansi(c->in1[v], c->in2[v], 128, 128, out, MUL_QUANT, c->size);
}

static void mul_opt(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_mul_elementwise_s8(c->in1[v], c->in2[v], 128, 128, out, MUL_QUANT, c->size);
}

static const bench_variant_t s_mul_variants[] = {
    { "esp_nn", mul_opt },
};

static void mul_bc_ref(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_mul_broadcast_channel_s8_ansi(c->in1[v], c->in2[v], 128, 128, out, MUL_QUANT,
                                         c->positions, c->channels);
}

static void mul_bc_opt(void *p, int v, int8_t *out)
{
    const math_ctx_t *c = p;
    esp_nn_mul_broadcast_channel_s8(c->in1[v], c->in2[v], 128, 128, out, MUL_QUANT,
                                    c->positions, c->channels);
}

static const bench_variant_t s_mul_bc_variants[] = {
    { "esp_nn", mul_bc_opt },
};

/* ------------------------------------------------------------------ Cases -- */

typedef struct {
    const char *op;
    int positions, channels;    /* elementwise: positions * channels elements */
    bool broadcast;             /* input2 is one [channels] vector */
    bench_mem_t mem;
    bench_fn ref;
    const bench_variant_t *variants;
    int n_variants;
} math_shape_t;

#define VARIANTS(v) (v), (int) (sizeof(v) / sizeof((v)[0]))

static const math_shape_t s_shapes[] = {
    { "add", 1, 1615, false, BENCH_MEM_EXT, add_ref, VARIANTS(s_add_variants) },
    { "mul", 1, 1615, false, BENCH_MEM_EXT, mul_ref, VARIANTS(s_mul_variants) },
    /* Squeeze-and-excite gating, MobileNetV3-Small. */
    { "mul_broadcast", 49, 576, true, BENCH_MEM_EXT, mul_bc_ref, VARIANTS(s_mul_bc_variants) },
};

static void run_shape(const math_shape_t *s)
{
    const size_t n1 = (size_t) s->positions * s->channels;
    const size_t n2 = s->broadcast ? (size_t) s->channels : n1;
    int8_t *in1[2] = { bench_alloc(n1, s->mem), bench_alloc(n1, s->mem) };
    int8_t *in2[2] = { bench_alloc(n2, s->mem), bench_alloc(n2, s->mem) };
    char shape[48];
    if (s->broadcast) {
        snprintf(shape, sizeof(shape), "%d positions x %d channels", s->positions, s->channels);
    } else {
        snprintf(shape, sizeof(shape), "size %d", (int) n1);
    }

    if (in1[0] && in1[1] && in2[0] && in2[1]) {
        math_ctx_t c = { .size = (int) n1, .positions = s->positions, .channels = s->channels };
        bench_seed_case(s->op, shape);
        for (int v = 0; v < 2; v++) {
            bench_fill(in1[v], n1);
            bench_fill(in2[v], n2);
            c.in1[v] = in1[v];
            c.in2[v] = in2[v];
        }
        const bench_case_t bc = {
            .op = s->op, .shape = shape, .mem = s->mem, .out_size = n1, .ref = s->ref,
            .variants = s->variants, .n_variants = s->n_variants, .ctx = &c,
        };
        bench_run(&bc);
    } else {
        bench_skip(s->op, shape, s->mem);
    }
    for (int v = 0; v < 2; v++) {
        bench_free(in1[v]);
        bench_free(in2[v]);
    }
}

void bench_basic_math(void)
{
    for (int i = 0; i < (int) (sizeof(s_shapes) / sizeof(s_shapes[0])); i++) {
        run_shape(&s_shapes[i]);
    }
}
