/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "sdkconfig.h"

#include <esp_nn.h>

#include "bench_ops.h"

/* ------------------------------------------------------- Per-row FC (s8) -- */

typedef struct {
    int row_len, out_ch;
    int32_t input_offset;
    bench_mem_t mem;
    /* Warn when esp_nn takes more than this percentage of the cycles of
     * esp_nn_operands_8mod16. 0: no check. */
    int max_pct;
} fc_shape_t;

/* The aligned fast paths once ran ~4x slower than the s16 assembly while
 * staying bit-exact (esp-nn#36). The limits below catch that: 10% over the
 * assembly for dispatch and noise. With input offset 0 on ESP32-S3, the
 * dispatcher must pick the dot path and beat the assembly (measured 0.68x). */
#define FC_MAX_PCT 110
#if CONFIG_IDF_TARGET_ESP32S3
#define FC_MAX_PCT_IO0 90
#else
#define FC_MAX_PCT_IO0 FC_MAX_PCT
#endif

static const fc_shape_t s_fc_shapes[] = {
    { 271,   3,  7, BENCH_MEM_INT, 0 },  /* README table */
    /* Both sides of the S3 dot-product threshold (row_len 256). */
    { 256, 256, -3, BENCH_MEM_EXT, FC_MAX_PCT },
    { 512, 128, -3, BENCH_MEM_EXT, FC_MAX_PCT },
    { 320, 320, -3, BENCH_MEM_EXT, FC_MAX_PCT },
    { 128, 256, -3, BENCH_MEM_EXT, FC_MAX_PCT },
    {  64, 256, -3, BENCH_MEM_EXT, FC_MAX_PCT },
    { 128, 256,  0, BENCH_MEM_EXT, FC_MAX_PCT_IO0 },
};

typedef struct {
    int row_len, out_ch;
    int32_t input_offset;
    const int8_t *input[2];
    const int8_t *filter;
    /* The same input and filter, 8 bytes past a 16-byte boundary. */
    const int8_t *input_off8[2];
    const int8_t *filter_off8;
    const int32_t *bias;
} fc_ctx_t;

/* Symmetric weights, as in TFLite int8 models. */
#define FC_FILTER_OFFSET 0
#define FC_OUT_MULT      1355715584
#define FC_OUT_SHIFT     -6

static void fc_ref(void *p, int v, int8_t *out)
{
    const fc_ctx_t *c = p;
    esp_nn_fully_connected_s8_ansi(c->input[v], c->input_offset, c->row_len, c->filter,
                                   FC_FILTER_OFFSET, c->bias, out, c->out_ch, 0,
                                   FC_OUT_SHIFT, FC_OUT_MULT, -128, 127);
}

static void fc_opt(void *p, int v, int8_t *out)
{
    const fc_ctx_t *c = p;
    esp_nn_fully_connected_s8(c->input[v], c->input_offset, c->row_len, c->filter,
                              FC_FILTER_OFFSET, c->bias, out, c->out_ch, 0,
                              FC_OUT_SHIFT, FC_OUT_MULT, -128, 127);
}

/* The dispatcher takes its fast paths only for 16-byte aligned operands.
 * At 8 mod 16, ESP32-S3 runs its s16 assembly instead: the baseline the
 * fast paths have to beat. */
static void fc_opt_off8(void *p, int v, int8_t *out)
{
    const fc_ctx_t *c = p;
    esp_nn_fully_connected_s8(c->input_off8[v], c->input_offset, c->row_len, c->filter_off8,
                              FC_FILTER_OFFSET, c->bias, out, c->out_ch, 0,
                              FC_OUT_SHIFT, FC_OUT_MULT, -128, 127);
}

static const bench_variant_t s_fc_variants[] = {
    { "esp_nn", fc_opt },
    { "esp_nn_operands_8mod16", fc_opt_off8 },
};

static void run_fc(const fc_shape_t *s)
{
    const size_t f_size = (size_t) s->row_len * s->out_ch;
    int8_t *in[2] = { bench_alloc(s->row_len, s->mem), bench_alloc(s->row_len, s->mem) };
    int8_t *in8[2] = { bench_alloc(s->row_len + 8, s->mem), bench_alloc(s->row_len + 8, s->mem) };
    int8_t *filter = bench_alloc(f_size, s->mem);
    int8_t *filter8 = bench_alloc(f_size + 8, s->mem);
    int32_t *bias = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    char shape[64];
    snprintf(shape, sizeof(shape), "length %d, %d outputs, input offset %d",
             (int) s->row_len, (int) s->out_ch, (int) s->input_offset);

    if (in[0] && in[1] && in8[0] && in8[1] && filter && filter8 && bias) {
        fc_ctx_t c = { .row_len = s->row_len, .out_ch = s->out_ch,
                       .input_offset = s->input_offset, .bias = bias };
        bench_seed_case("fully_connected", shape);
        for (int v = 0; v < 2; v++) {
            bench_fill(in[v], s->row_len);
            memcpy(in8[v] + 8, in[v], s->row_len);
            c.input[v] = in[v];
            c.input_off8[v] = in8[v] + 8;
        }
        bench_fill(filter, f_size);
        memcpy(filter8 + 8, filter, f_size);
        bench_fill_bias(bias, s->out_ch);
        c.filter = filter;
        c.filter_off8 = filter8 + 8;

        uint32_t cycles[2] = { 0 };
        const bench_case_t bc = {
            .op = "fully_connected", .shape = shape, .mem = s->mem,
            .out_size = s->out_ch, .ref = fc_ref, .variants = s_fc_variants,
            .n_variants = sizeof(s_fc_variants) / sizeof(s_fc_variants[0]), .ctx = &c,
            .cycles = cycles,
        };
        bench_run(&bc);
        if (s->max_pct && (uint64_t) cycles[0] * 100 > (uint64_t) cycles[1] * s->max_pct) {
            printf("BENCH_WARN op=fully_connected esp_nn=%" PRIu32 " esp_nn_operands_8mod16=%"
                   PRIu32 " limit=%d%% shape=\"%s\"\n", cycles[0], cycles[1], s->max_pct, shape);
        }
    } else {
        bench_skip("fully_connected", shape, s->mem);
    }
    bench_free(in[0]);
    bench_free(in[1]);
    bench_free(in8[0]);
    bench_free(in8[1]);
    bench_free(filter);
    bench_free(filter8);
    bench_free(bias);
}

/* ------------------------------------- Batched per-channel FC (s8, batch) -- */

typedef struct {
    int batches, row_len, out_ch;
    bench_mem_t mem;
} fc_batch_shape_t;

/* batch x inner dimension x outputs: attention projections and feed-forward
 * layers of small transformers. */
static const fc_batch_shape_t s_batch_shapes[] = {
    { 16,  64,  64, BENCH_MEM_INT },
    { 32,  64, 128, BENCH_MEM_INT },
    {  4, 128, 192, BENCH_MEM_INT },
    {  4, 128, 200, BENCH_MEM_INT },
    { 48, 160,  32, BENCH_MEM_INT },
    { 64, 128, 256, BENCH_MEM_INT },
    {  5,  32,  32, BENCH_MEM_INT },
    { 66,  16,  16, BENCH_MEM_INT },
};

typedef struct {
    int batches, row_len, out_ch;
    const int8_t *input[2];
    const int8_t *filter;
    const int32_t *bias, *mult, *shift;
    void *scratch;
    size_t scratch_size;
} fc_batch_ctx_t;

#define FC_BATCH_INPUT_OFFSET 128

static void fc_batch_ref(void *p, int v, int8_t *out)
{
    const fc_batch_ctx_t *c = p;
    esp_nn_fully_connected_per_ch_s8_batch_ansi(c->input[v], FC_BATCH_INPUT_OFFSET, c->row_len,
                                                c->filter, 0, c->bias, out, c->out_ch, 0,
                                                c->shift, c->mult, -128, 127, c->batches);
}

static void fc_batch_opt(void *p, int v, int8_t *out)
{
    const fc_batch_ctx_t *c = p;
    esp_nn_fully_connected_per_ch_s8_batch(c->input[v], FC_BATCH_INPUT_OFFSET, c->row_len,
                                           c->filter, 0, c->bias, out, c->out_ch, 0,
                                           c->shift, c->mult, -128, 127, c->batches);
}

/* The scratch call runs the batch as a 1x1 convolution when its gate
 * accepts the shape, and otherwise the same kernel as esp_nn. */
static void fc_batch_scratch(void *p, int v, int8_t *out)
{
    const fc_batch_ctx_t *c = p;
    esp_nn_fully_connected_per_ch_s8_batch_scratch(c->input[v], FC_BATCH_INPUT_OFFSET,
                                                   c->row_len, c->filter, 0, c->bias, out,
                                                   c->out_ch, 0, c->shift, c->mult, -128, 127,
                                                   c->batches, c->scratch, c->scratch_size);
}

/* Another way to compute the same batch goes here as one more entry: it is
 * timed and checked like the others. */
static const bench_variant_t s_batch_variants[] = {
    { "esp_nn", fc_batch_opt },
    { "esp_nn_scratch", fc_batch_scratch },
};

/* Warn when the scratch call takes more than this percentage of the cycles
 * of esp_nn: its gate should pick the 1x1 path only where it wins. */
#define FC_BATCH_SCRATCH_MAX_PCT 105

static void run_fc_batch(const fc_batch_shape_t *s)
{
    const size_t in_size = (size_t) s->batches * s->row_len;
    const size_t f_size = (size_t) s->row_len * s->out_ch;
    int8_t *in[2] = { bench_alloc(in_size, s->mem), bench_alloc(in_size, s->mem) };
    int8_t *filter = bench_alloc(f_size, s->mem);
    int32_t *bias = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *mult = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    int32_t *shift = bench_alloc(s->out_ch * sizeof(int32_t), BENCH_MEM_INT);
    /* With the tensors, as in a TFLite Micro arena. */
    const int32_t scratch_size = esp_nn_get_fc_batch_scratch_size(s->batches, s->row_len,
                                                                  s->out_ch);
    void *scratch = scratch_size > 0 ? bench_alloc(scratch_size, s->mem) : NULL;
    char shape[64];
    snprintf(shape, sizeof(shape), "batch %d, length %d, %d outputs",
             s->batches, s->row_len, s->out_ch);

    if (in[0] && in[1] && filter && bias && mult && shift && (scratch_size <= 0 || scratch)) {
        bench_seed_case("fully_connected_batch", shape);
        bench_fill(in[0], in_size);
        bench_fill(in[1], in_size);
        bench_fill(filter, f_size);
        bench_fill_bias(bias, s->out_ch);
        bench_fill_quant(mult, shift, s->out_ch);
        fc_batch_ctx_t c = { .batches = s->batches, .row_len = s->row_len, .out_ch = s->out_ch,
                             .input = { in[0], in[1] }, .filter = filter,
                             .bias = bias, .mult = mult, .shift = shift,
                             .scratch = scratch,
                             .scratch_size = scratch ? (size_t) scratch_size : 0 };
        uint32_t cycles[2] = { 0 };
        const bench_case_t bc = {
            .op = "fully_connected_batch", .shape = shape, .mem = s->mem,
            .out_size = (size_t) s->batches * s->out_ch, .ref = fc_batch_ref,
            .variants = s_batch_variants,
            .n_variants = sizeof(s_batch_variants) / sizeof(s_batch_variants[0]), .ctx = &c,
            .cycles = cycles,
        };
        bench_run(&bc);
        if ((uint64_t) cycles[1] * 100 > (uint64_t) cycles[0] * FC_BATCH_SCRATCH_MAX_PCT) {
            printf("BENCH_WARN op=fully_connected_batch esp_nn_scratch=%" PRIu32 " esp_nn=%"
                   PRIu32 " limit=%d%% shape=\"%s\"\n", cycles[1], cycles[0],
                   FC_BATCH_SCRATCH_MAX_PCT, shape);
        }
    } else {
        bench_skip("fully_connected_batch", shape, s->mem);
    }
    bench_free(in[0]);
    bench_free(in[1]);
    bench_free(filter);
    bench_free(bias);
    bench_free(mult);
    bench_free(shift);
    bench_free(scratch);
}

void bench_fully_connected(void)
{
    for (int i = 0; i < (int) (sizeof(s_fc_shapes) / sizeof(s_fc_shapes[0])); i++) {
        run_fc(&s_fc_shapes[i]);
    }
    for (int i = 0; i < (int) (sizeof(s_batch_shapes) / sizeof(s_batch_shapes[0])); i++) {
        run_fc_batch(&s_batch_shapes[i]);
    }
}
