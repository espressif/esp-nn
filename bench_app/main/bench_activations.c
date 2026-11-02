/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>

#include <esp_nn.h>

#include "bench_ops.h"

typedef struct {
    int size;
    const int8_t *input[2];
} act_ctx_t;

/* ----------------------------------------------------------------- ReLU6 -- */

/* ReLU6 works in place: prep copies the input into the output buffer, and
 * only the kernel call is timed. */
static void relu6_prep(void *p, int v, int8_t *out)
{
    const act_ctx_t *c = p;
    memcpy(out, c->input[v], c->size);
}

static void relu6_ref(void *p, int v, int8_t *out)
{
    (void) v;
    esp_nn_relu6_s8_ansi(out, (uint16_t) ((const act_ctx_t *) p)->size);
}

static void relu6_opt(void *p, int v, int8_t *out)
{
    (void) v;
    esp_nn_relu6_s8(out, (uint16_t) ((const act_ctx_t *) p)->size);
}

static const bench_variant_t s_relu6_variants[] = {
    { "esp_nn", relu6_opt },
};

/* ------------------------------------------------------------ Hard swish -- */

/* MobileNetV3 parameters. */
#define HSWISH_ARGS(c, in, out) (in), (out), (c)->size, -128, 19661, 22938, 2, -1, -128

static void hswish_ref(void *p, int v, int8_t *out)
{
    const act_ctx_t *c = p;
    esp_nn_hard_swish_s8_ansi(HSWISH_ARGS(c, c->input[v], out));
}

static void hswish_opt(void *p, int v, int8_t *out)
{
    const act_ctx_t *c = p;
    esp_nn_hard_swish_s8(HSWISH_ARGS(c, c->input[v], out));
}

static const bench_variant_t s_hswish_variants[] = {
    { "esp_nn", hswish_opt },
};

/* ------------------------------------------------------------------ Cases -- */

static void run_case(const char *op, int size, bench_mem_t mem, bench_fn prep, bench_fn ref,
                     const bench_variant_t *variants, int n_variants)
{
    int8_t *in[2] = { bench_alloc(size, mem), bench_alloc(size, mem) };
    char shape[24];
    snprintf(shape, sizeof(shape), "size %d", size);

    if (in[0] && in[1]) {
        bench_seed_case(op, shape);
        bench_fill(in[0], size);
        bench_fill(in[1], size);
        act_ctx_t c = { .size = size, .input = { in[0], in[1] } };
        const bench_case_t bc = {
            .op = op, .shape = shape, .mem = mem, .out_size = size, .ref = ref, .prep = prep,
            .variants = variants, .n_variants = n_variants, .ctx = &c,
        };
        bench_run(&bc);
    } else {
        bench_skip(op, shape, mem);
    }
    bench_free(in[0]);
    bench_free(in[1]);
}

void bench_activations(void)
{
    run_case("relu6", 1615, BENCH_MEM_INT, relu6_prep, relu6_ref, s_relu6_variants,
             sizeof(s_relu6_variants) / sizeof(s_relu6_variants[0]));

    /* The optimized hard swish reads a lookup table from its scratch. */
    const int scratch_size = esp_nn_get_hard_swish_scratch_size();
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, BENCH_MEM_INT) : NULL;
    if (scratch_size > 0 && scratch == NULL) {
        bench_skip("hard_swish", "size 12544", BENCH_MEM_EXT);
        return;
    }
    if (scratch) {
        esp_nn_set_hard_swish_scratch_buf(scratch);
    }
    run_case("hard_swish", 12544, BENCH_MEM_EXT, NULL, hswish_ref, s_hswish_variants,
             sizeof(s_hswish_variants) / sizeof(s_hswish_variants[0]));
    esp_nn_set_hard_swish_scratch_buf(NULL);
    bench_free(scratch);
}
