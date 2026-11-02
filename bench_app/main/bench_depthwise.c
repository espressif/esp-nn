/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>

#include <esp_nn.h>

#include "bench_ops.h"

typedef struct {
    int in_wd, in_ht, ch, ch_mult;
    int f_wd, f_ht;
    int stride_wd, stride_ht, pad_wd, pad_ht;
    bench_mem_t mem;
} dw_shape_t;

/* The shapes of the README tables. */
static const dw_shape_t s_shapes[] = {
    { 18, 18, 16, 1, 3, 3, 1, 1, 0, 0, BENCH_MEM_EXT },
    { 16, 16, 16, 1, 3, 3, 1, 1, 0, 0, BENCH_MEM_EXT },
    { 10, 10, 16, 1, 3, 3, 1, 1, 1, 1, BENCH_MEM_INT },
    { 12, 12,  4, 8, 5, 5, 1, 1, 1, 1, BENCH_MEM_EXT },
    /* Width-1 tensors with a (3,1) filter, as in speech models. */
    {  1, 384, 128, 1, 1, 3, 1, 1, 0, 1, BENCH_MEM_INT },
    {  1, 192, 256, 1, 1, 3, 1, 2, 0, 1, BENCH_MEM_INT },
};

typedef struct {
    data_dims_t in_dims, f_dims, out_dims;
    dw_conv_params_t params;
    quant_data_t quant;
    const int8_t *input[2];
    const int8_t *filter;
    const int32_t *bias;
} dw_ctx_t;

static void dw_ref(void *p, int v, int8_t *out)
{
    const dw_ctx_t *c = p;
    esp_nn_depthwise_conv_s8_ansi(&c->in_dims, c->input[v], &c->f_dims, c->filter, c->bias,
                                  &c->out_dims, out, &c->params, &c->quant);
}

static void dw_opt(void *p, int v, int8_t *out)
{
    const dw_ctx_t *c = p;
    esp_nn_depthwise_conv_s8(&c->in_dims, c->input[v], &c->f_dims, c->filter, c->bias,
                             &c->out_dims, out, &c->params, &c->quant);
}

static const bench_variant_t s_variants[] = {
    { "esp_nn", dw_opt },
};

static void run_shape(const dw_shape_t *s)
{
    const int out_ch = s->ch * s->ch_mult;
    const int out_wd = (s->in_wd + 2 * s->pad_wd - s->f_wd) / s->stride_wd + 1;
    const int out_ht = (s->in_ht + 2 * s->pad_ht - s->f_ht) / s->stride_ht + 1;
    const size_t in_size = (size_t) s->in_wd * s->in_ht * s->ch;
    const size_t f_size = (size_t) s->f_wd * s->f_ht * out_ch;

    dw_ctx_t c = {
        .in_dims = { .width = s->in_wd, .height = s->in_ht, .channels = s->ch, 1 },
        .f_dims = { .width = s->f_wd, .height = s->f_ht, 0, 0 },
        .out_dims = { .width = out_wd, .height = out_ht, .channels = out_ch, 1 },
        .params = { .in_offset = 128, .out_offset = -5, .ch_mult = s->ch_mult,
                    .stride = { s->stride_wd, s->stride_ht },
                    .padding = { s->pad_wd, s->pad_ht },
                    .dilation = { 0, 0 }, .activation = { -128, 127 } },
    };
    int8_t *in0 = bench_alloc(in_size, s->mem);
    int8_t *in1 = bench_alloc(in_size, s->mem);
    int8_t *filter = bench_alloc(f_size, s->mem);
    int32_t *bias = bench_alloc(out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *mult = bench_alloc(out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *shift = bench_alloc(out_ch * sizeof(int32_t), BENCH_MEM_INT);
    const int scratch_size = esp_nn_get_depthwise_conv_scratch_size(&c.in_dims, &c.f_dims,
                                                                    &c.out_dims, &c.params);
    /* Scratch sits with the tensors. With mem=ext that is PSRAM, as in a TFLite
     * Micro arena in PSRAM; the S3 preferred-scratch hooks stay unset. */
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, s->mem) : NULL;
    char shape[112];
    snprintf(shape, sizeof(shape),
             "input %dx%dx%d, filter %dx%dx%dx%d, pad (%d,%d), stride (%d,%d)",
             s->in_wd, s->in_ht, s->ch, s->ch_mult, s->f_wd, s->f_ht, s->ch,
             s->pad_wd, s->pad_ht, s->stride_wd, s->stride_ht);

    if (in0 && in1 && filter && bias && mult && shift && (scratch_size <= 0 || scratch)) {
        bench_seed_case("depthwise_conv", shape);
        bench_fill(in0, in_size);
        bench_fill(in1, in_size);
        bench_fill(filter, f_size);
        bench_fill_bias(bias, out_ch);
        bench_fill_quant(mult, shift, out_ch);
        c.input[0] = in0;
        c.input[1] = in1;
        c.filter = filter;
        c.bias = bias;
        c.quant = (quant_data_t) { .shift = shift, .mult = mult };
        esp_nn_set_depthwise_conv_scratch_buf(scratch);

        const bench_case_t bc = {
            .op = "depthwise_conv", .shape = shape, .mem = s->mem,
            .out_size = (size_t) out_wd * out_ht * out_ch,
            .ref = dw_ref, .variants = s_variants,
            .n_variants = sizeof(s_variants) / sizeof(s_variants[0]), .ctx = &c,
        };
        bench_run(&bc);
        esp_nn_set_depthwise_conv_scratch_buf(NULL);
    } else {
        bench_skip("depthwise_conv", shape, s->mem);
    }
    bench_free(in0);
    bench_free(in1);
    bench_free(filter);
    bench_free(bias);
    bench_free(mult);
    bench_free(shift);
    bench_free(scratch);
}

void bench_depthwise(void)
{
    for (int i = 0; i < (int) (sizeof(s_shapes) / sizeof(s_shapes[0])); i++) {
        run_shape(&s_shapes[i]);
    }
}
