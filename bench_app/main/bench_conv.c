/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>

#include <esp_nn.h>

#include "bench_ops.h"

typedef struct {
    int in_wd, in_ht, in_ch, out_ch;
    int f_wd, f_ht, stride, pad;
    int out_wd, out_ht;         /* 0: derived from pad and stride */
    bench_mem_t mem;
} conv_shape_t;

/* The shapes of the README tables. */
static const conv_shape_t s_shapes[] = {
    { 10, 10, 64, 64, 1, 1, 1, 0, 0, 0, BENCH_MEM_EXT },
    {  8,  8, 16, 16, 1, 1, 1, 0, 0, 0, BENCH_MEM_EXT },
    {  8,  8,  3, 64, 3, 3, 1, 0, 0, 0, BENCH_MEM_EXT },
    { 10, 10,  3, 64, 3, 3, 1, 0, 0, 0, BENCH_MEM_EXT },
    /* TFLite SAME, stride 2: the trailing padding is implicit. */
    {  8,  8, 16, 16, 3, 3, 2, 0, 4, 4, BENCH_MEM_INT },
};

typedef struct {
    data_dims_t in_dims, f_dims, out_dims;
    conv_params_t params;
    quant_data_t quant;
    const int8_t *input[2];
    const int8_t *filter;
    const int32_t *bias;
} conv_ctx_t;

static void conv_ref(void *p, int v, int8_t *out)
{
    const conv_ctx_t *c = p;
    esp_nn_conv_s8_ansi(&c->in_dims, c->input[v], &c->f_dims, c->filter, c->bias,
                        &c->out_dims, out, &c->params, &c->quant);
}

static void conv_opt(void *p, int v, int8_t *out)
{
    const conv_ctx_t *c = p;
    esp_nn_conv_s8(&c->in_dims, c->input[v], &c->f_dims, c->filter, c->bias,
                   &c->out_dims, out, &c->params, &c->quant);
}

static const bench_variant_t s_variants[] = {
    { "esp_nn", conv_opt },
};

static void run_shape(const conv_shape_t *s)
{
    const int out_wd = s->out_wd ? s->out_wd : (s->in_wd + 2 * s->pad - s->f_wd) / s->stride + 1;
    const int out_ht = s->out_ht ? s->out_ht : (s->in_ht + 2 * s->pad - s->f_ht) / s->stride + 1;
    const size_t in_size = (size_t) s->in_wd * s->in_ht * s->in_ch;
    const size_t f_size = (size_t) s->f_wd * s->f_ht * s->in_ch * s->out_ch;

    conv_ctx_t c = {
        .in_dims = { .width = s->in_wd, .height = s->in_ht, .channels = s->in_ch, 1 },
        .f_dims = { .width = s->f_wd, .height = s->f_ht, .channels = s->in_ch, 1 },
        .out_dims = { .width = out_wd, .height = out_ht, .channels = s->out_ch, 1 },
        .params = { .in_offset = 128, .out_offset = -5,
                    .stride = { s->stride, s->stride }, .padding = { s->pad, s->pad },
                    .dilation = { 0, 0 }, .activation = { -128, 127 } },
    };
    int8_t *in0 = bench_alloc(in_size, s->mem);
    int8_t *in1 = bench_alloc(in_size, s->mem);
    int8_t *filter = bench_alloc(f_size, s->mem);
    int32_t *bias = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *mult = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *shift = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    const int scratch_size = esp_nn_get_conv_scratch_size(&c.in_dims, &c.f_dims,
                                                          &c.out_dims, &c.params);
    /* Scratch sits with the tensors. With mem=ext that is PSRAM, as in a TFLite
     * Micro arena in PSRAM; the S3 preferred-scratch hooks stay unset. */
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, s->mem) : NULL;
    char shape[96];
    snprintf(shape, sizeof(shape), "input %dx%dx%d, filter %dx%dx%dx%d, pad %d, stride %d",
             s->in_wd, s->in_ht, s->in_ch, s->out_ch, s->f_wd, s->f_ht, s->in_ch,
             s->pad, s->stride);

    if (in0 && in1 && filter && bias && mult && shift && (scratch_size <= 0 || scratch)) {
        bench_seed_case("conv", shape);
        bench_fill(in0, in_size);
        bench_fill(in1, in_size);
        bench_fill(filter, f_size);
        bench_fill_bias(bias, s->out_ch);
        bench_fill_quant(mult, shift, s->out_ch);
        c.input[0] = in0;
        c.input[1] = in1;
        c.filter = filter;
        c.bias = bias;
        c.quant = (quant_data_t) { .shift = shift, .mult = mult };
        esp_nn_set_conv_scratch_buf(scratch);

        const bench_case_t bc = {
            .op = "conv", .shape = shape, .mem = s->mem,
            .out_size = (size_t) out_wd * out_ht * s->out_ch,
            .ref = conv_ref, .variants = s_variants,
            .n_variants = sizeof(s_variants) / sizeof(s_variants[0]), .ctx = &c,
        };
        bench_run(&bc);
        esp_nn_set_conv_scratch_buf(NULL);
    } else {
        bench_skip("conv", shape, s->mem);
    }
    bench_free(in0);
    bench_free(in1);
    bench_free(filter);
    bench_free(bias);
    bench_free(mult);
    bench_free(shift);
    bench_free(scratch);
}

void bench_conv(void)
{
    for (int i = 0; i < (int) (sizeof(s_shapes) / sizeof(s_shapes[0])); i++) {
        run_shape(&s_shapes[i]);
    }
}
