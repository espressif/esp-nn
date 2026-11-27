/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * FC multi-path dispatcher for ESP32-S3.
 * - 16-byte aligned input, filter_offset == 0, row_len >= 16: the one-pass
 *   kernel (esp_nn_fc_fused_s8_esp32s3.c), which folds the input offset into
 *   its MACs
 * - Otherwise: the s8 dot path, which adds the per-channel offset correction
 *   (input_offset * filter row sum + bias) in C, or the s16 assembly
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <common_functions.h>
#include <esp_nn_ansi_headers.h>
#include "esp_nn_fc_internal.h"
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_fc_requant_esp32s3.h"

/* Original s16 assembly (renamed) */
extern void esp_nn_fc_s16_esp32s3(const int8_t *input_data,
                                   const int32_t input_offset,
                                   const uint16_t row_len,
                                   const int8_t *filter_data,
                                   const int32_t filter_offset,
                                   const int32_t *bias,
                                   int8_t *out_data,
                                   const uint16_t out_channels,
                                   const int32_t out_offset,
                                   const int32_t out_shift,
                                   const int32_t out_mult,
                                   const int32_t activation_min,
                                   const int32_t activation_max);

extern void esp_nn_fc_per_ch_s16_esp32s3(const int8_t *input_data,
                                          const int32_t input_offset,
                                          const uint16_t row_len,
                                          const int8_t *filter_data,
                                          const int32_t filter_offset,
                                          const int32_t *bias,
                                          int8_t *out_data,
                                          const uint16_t out_channels,
                                          const int32_t out_offset,
                                          const int32_t *out_shift,
                                          const int32_t *out_mult,
                                          const int32_t activation_min,
                                          const int32_t activation_max);

/* One-pass kernel (esp_nn_fc_fused_s8_esp32s3.c): filter_offset == 0,
 * 16-byte aligned input and row_len >= 16 (fc_fused_wins). It is exact for
 * input_offset in [-256, 254]. quant_step is 0 for per-tensor and 1 for per-channel
 * mult/shift. */
extern void esp_nn_fc_fused_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t quant_step,
                                       const int32_t activation_min,
                                       const int32_t activation_max);

/* Shared s8 dot product from common. `a` must be 16-byte aligned, `b` may be
 * unaligned (handled via USAR+QUP). The product is symmetric, so whichever of
 * input/filter happens to be aligned can be passed as `a`.
 *
 * Both operands are read past their logical end, by the usual esp-nn amount:
 * the primed 2x-unrolled loop issues one 128-bit block more than it consumes.
 * Per call, with n = len_div16:
 *   a (aligned):   16 bytes over for even n, 0 for odd n
 *   b (unaligned): 17..31 bytes over for even n, 1..15 for odd n
 * Reads only. Nothing is written outside out_data.
 *
 * fc_dot_rows() passes the aligned filter row as `a` and the input as `b`, so
 * the filter is read up to 16 bytes over and the input up to 31 bytes over.
 * The batched kernel passes whichever operand is aligned as `a`. With an
 * aligned input, each filter row is then read up to 31 bytes over.
 * These are aligned loads from mapped SRAM/PSRAM and do not fault; the only
 * theoretical corner is a tensor ending within 32 bytes of the end of a mapped
 * region. */
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a,
                                                const int8_t *b,
                                                int32_t len_div16);

/* The s16 assembly loads the input with 8-byte vector loads and only derives
 * SAR_BYTE from the filter pointer, so it silently requires an 8-byte aligned
 * input. Anything less has to go to the ansi reference. */
#define FC_S16_INPUT_ALIGN  8

/* When the s8 dot path beats the s16 assembly. The dot path has about
 * twice the assembly's MAC rate, but pays a per-channel filter-sum pass when
 * input_offset != 0, which the assembly folds into its MAC. Measured on S3
 * (row_len 16 to 1024, out_ch 1 to 256, which never moves the boundary):
 * - input_offset != 0: from row_len 192.
 * - input_offset == 0: from row_len 64.
 * Its callers take the dot path only for whole 16-byte rows, so the tail
 * term is 0 there. It stays because it keeps the compiler from testing the
 * filter alignment ahead of the one-pass branch, which costs that branch
 * 2-3 cycles per call.
 * The boundary is empirical and can drift with cache geometry. Both paths
 * are bit-exact, so a mis-tuned boundary costs a few percent, never
 * correctness. The per-row kernels take the one-pass kernel first whenever
 * fc_fused_wins(), so this gate only picks between the two older paths. */
static inline bool fc_dot_path_wins(uint16_t row_len, int32_t input_offset)
{
    const int tail = row_len & 15;
    if (input_offset == 0) {
        return row_len >= 64 + tail * 8;
    }
    return row_len >= 192 + tail * 16;
}

/* When the one-pass kernel beats both older paths. It needs
 * filter_offset == 0 and a 16-byte aligned input. It takes any input_offset
 * in [-256, 254], which covers the API range. With input_offset != 0 it
 * saves the dot path's filter-sum pass. In every case it runs about twice the
 * s16 assembly's MAC rate, and its row loop needs no call per channel. Rows
 * shorter than one vector stay on the s16 assembly: the kernel's per-call
 * tail setup costs more than it saves there. */
static inline bool fc_fused_wins(const int8_t *input_data, uint16_t row_len,
                                 int32_t filter_offset)
{
    return filter_offset == 0 && ((uintptr_t)input_data & 15) == 0 && row_len >= 16;
}

/* Per-tensor entry to the one-pass kernel. noinline keeps the address-taken
 * copies of mult/shift out of the caller's frame. */
static __attribute__((noinline)) void esp_nn_fc_fused_s8_pt(const int8_t *input_data,
        const int32_t input_offset, const uint16_t row_len,
        const int8_t *filter_data, const int32_t *bias, int8_t *out_data,
        const uint16_t out_channels, const int32_t out_offset,
        int32_t out_shift, int32_t out_mult,
        const int32_t activation_min, const int32_t activation_max)
{
    esp_nn_fc_fused_s8_esp32s3(input_data, input_offset, row_len, filter_data,
                               bias, out_data, out_channels, out_offset,
                               &out_shift, &out_mult, 0,
                               activation_min, activation_max);
}

/* The s8 dot path: filter_offset == 0 and 16-byte aligned filter rows
 * (aligned base, row_len % 16 == 0). The one-pass kernel takes every aligned
 * input, so only unaligned inputs come here. Both entry points share it to
 * keep the code small. quant_step is 0 for per-tensor and 1 for per-channel
 * mult/shift. */
static __attribute__((noinline)) void fc_dot_rows(const int8_t *input_data,
        const int32_t input_offset, const uint16_t row_len,
        const int8_t *filter_data, const int32_t *bias, int8_t *out_data,
        const uint16_t out_channels, const int32_t out_offset,
        const int32_t *out_shift, const int32_t *out_mult, const int32_t quant_step,
        const int32_t activation_min, const int32_t activation_max)
{
    const int32_t row_len_div16 = row_len >> 4;

    for (int ch = 0; ch < out_channels; ch++) {
        const int8_t *f_ptr = filter_data + ch * row_len;
        /* Per-channel correction, inline (no out_channels-sized VLA) */
        int32_t corr = 0;
        if (input_offset != 0) {
            corr = esp_nn_filter_sum_s8_esp32s3(f_ptr, row_len) * input_offset;
        }
        if (bias) {
            corr += bias[ch];
        }
        /* The aligned operand goes first; the dot product is symmetric. */
        int32_t acc = esp_nn_dot_s8_unaligned_esp32s3(f_ptr, input_data, row_len_div16);

        acc += corr;

        acc = fc_requant(acc, out_mult[ch * quant_step], out_shift[ch * quant_step]);
        acc += out_offset;
        acc = max(acc, activation_min);
        acc = min(acc, activation_max);
        out_data[ch] = (int8_t)acc;
    }
}

/* Per-tensor entry to fc_dot_rows(). */
static __attribute__((noinline)) void fc_dot_rows_pt(const int8_t *input_data,
        const int32_t input_offset, const uint16_t row_len,
        const int8_t *filter_data, const int32_t *bias, int8_t *out_data,
        const uint16_t out_channels, const int32_t out_offset,
        int32_t out_shift, int32_t out_mult,
        const int32_t activation_min, const int32_t activation_max)
{
    fc_dot_rows(input_data, input_offset, row_len, filter_data, bias, out_data,
                out_channels, out_offset, &out_shift, &out_mult, 0,
                activation_min, activation_max);
}

void esp_nn_fully_connected_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t out_shift,
                                       const int32_t out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max)
{
    if (fc_fused_wins(input_data, row_len, filter_offset)) {
        esp_nn_fc_fused_s8_pt(input_data, input_offset, row_len, filter_data,
                              bias, out_data, out_channels, out_offset,
                              out_shift, out_mult, activation_min, activation_max);
        return;
    }
    /* The input is not 16-byte aligned here (or the s16 assembly takes the
     * call), so the dot path needs aligned filter rows: an aligned base and a
     * row_len that is a whole number of vectors. */
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);

    if (__builtin_expect(filter_offset != 0 || !fc_dot_path_wins(row_len, input_offset)
        || !filter_rows_aligned, 0)) {
        if ((uintptr_t)input_data & (FC_S16_INPUT_ALIGN - 1)) {
            esp_nn_fully_connected_s8_ansi(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult,
                                           activation_min, activation_max);
            return;
        }
        /* Fallback to original s16 assembly: tail call, no extra overhead */
        esp_nn_fc_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                              filter_offset, bias, out_data, out_channels,
                              out_offset, out_shift, out_mult,
                              activation_min, activation_max);
        return;
    }
    fc_dot_rows_pt(input_data, input_offset, row_len, filter_data, bias, out_data,
                   out_channels, out_offset, out_shift, out_mult,
                   activation_min, activation_max);
}

void esp_nn_fully_connected_per_ch_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max)
{
    if (fc_fused_wins(input_data, row_len, filter_offset)) {
        esp_nn_fc_fused_s8_esp32s3(input_data, input_offset, row_len, filter_data,
                                   bias, out_data, out_channels, out_offset,
                                   out_shift, out_mult, 1,
                                   activation_min, activation_max);
        return;
    }
    /* As in the per-tensor entry: the dot path needs aligned filter rows. */
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);

    if (__builtin_expect(filter_offset != 0 || !fc_dot_path_wins(row_len, input_offset)
        || !filter_rows_aligned, 0)) {
        if ((uintptr_t)input_data & (FC_S16_INPUT_ALIGN - 1)) {
            esp_nn_fully_connected_per_ch_s8_ansi(input_data, input_offset, row_len,
                                                  filter_data, filter_offset, bias,
                                                  out_data, out_channels, out_offset,
                                                  out_shift, out_mult,
                                                  activation_min, activation_max);
            return;
        }
        esp_nn_fc_per_ch_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                                     filter_offset, bias, out_data, out_channels,
                                     out_offset, out_shift, out_mult,
                                     activation_min, activation_max);
        return;
    }
    fc_dot_rows(input_data, input_offset, row_len, filter_data, bias, out_data,
                out_channels, out_offset, out_shift, out_mult, 1,
                activation_min, activation_max);
}

/* Batch 1 uses the public per-channel entry, so the header gate can split it.
 * Other batches compute `batches` input rows against the SAME filter in a
 * single call. Two amortizations over calling the per-row kernel `batches`
 * times:
 *   1. the per-channel correction (input_offset*sum(filter_row) + bias) does
 *      not depend on the batch on the fast path (filter_offset == 0), so each
 *      filter row is summed once and the result serves every input row:
 *      (batches-1) fewer filter walks;
 *   2. the loop is reordered out_channel-outer / batch-inner, so each filter row
 *      (row_len bytes) is fetched from PSRAM once per channel and reused from
 *      dcache across all batches, weight-stationary instead of weight-streaming.
 * Bit-identical to the per-row kernel; falls back to it when the s8 fast path is
 * ineligible. Layout: input [batches][row_len], out_data [batches][out_channels]. */
void esp_nn_fc_batch_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max,
                                       const int32_t batches)
{
    if (batches == 1) {
        esp_nn_fully_connected_per_ch_s8(input_data, input_offset, row_len, filter_data, filter_offset, bias,
            out_data, out_channels, out_offset, out_shift, out_mult,
            activation_min, activation_max);
        return;
    }

    const bool input_aligned = (((uintptr_t)input_data & 15) == 0)
                               && ((row_len & 15) == 0);
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);

    /* One row has nothing to share, so it takes the per-row kernel and its
     * one-pass path. For more rows, each filter-row sum is shared by all of
     * them, so its cost is amortised and the offset-free threshold applies.
     * The dot needs one operand 16-byte aligned (input rows are the common
     * case). Otherwise defer to the per-row kernel, which further dispatches
     * to s16 asm / ansi as needed. */
    if (__builtin_expect(batches == 1 || filter_offset != 0
        || !fc_dot_path_wins(row_len, 0)
        || (!input_aligned && !filter_rows_aligned), 0)) {
        for (int32_t b = 0; b < batches; b++) {
            esp_nn_fully_connected_per_ch_s8_esp32s3(
                input_data + b * row_len, input_offset, row_len, filter_data,
                filter_offset, bias, out_data + b * out_channels, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max);
        }
        return;
    }

    /* Rows here are whole 16-byte vectors (input_aligned or
     * filter_rows_aligned above), so there is no scalar tail. */
    const int32_t row_len_div16 = row_len >> 4;

    /* out_channel outer, batch inner: f_ptr stays cache-resident across batches.
     * The correction is batch-independent (filter_offset == 0 here), so it is
     * computed once per channel, right before the row is reused for every batch;
     * no per-channel array on the caller's stack. */
    for (int ch = 0; ch < out_channels; ch++) {
        const int8_t *f_ptr = filter_data + ch * row_len;
        int32_t corr = 0;
        if (input_offset != 0) {
            corr = esp_nn_filter_sum_s8_esp32s3(f_ptr, row_len) * input_offset;
        }
        if (bias) {
            corr += bias[ch];
        }
        const int32_t mult = out_mult[ch];
        const int32_t shift = out_shift[ch];
        for (int32_t b = 0; b < batches; b++) {
            const int8_t *in = input_data + b * row_len;
            /* Pass the aligned operand first; the dot is symmetric. */
            int32_t acc = input_aligned
                ? esp_nn_dot_s8_unaligned_esp32s3(in, f_ptr, row_len_div16)
                : esp_nn_dot_s8_unaligned_esp32s3(f_ptr, in, row_len_div16);
            acc += corr;
            acc = fc_requant(acc, mult, shift);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[b * out_channels + ch] = (int8_t)acc;
        }
    }
}

#if ESP_NN_DUAL_CORE_SUPPORTED
/* The public batch entry calls this for two or more rows when a dual-core
 * worker exists. Split the output count in half. With a zero filter offset
 * each core keeps the weight-stationary order: its own channels outer,
 * every batch inner, so each filter row stays on one core. A nonzero
 * filter offset calls the per-row kernel on that same channel slice, once
 * per batch. A single batch, fewer than ESP_NN_FC_BATCH_DUAL_MIN_OUT
 * outputs, a row the s8 dot path declines, neither operand 16-byte aligned,
 * or a worker that will not take the job stays on the single-core batch
 * kernel. */
typedef struct {
    const int8_t *input;
    const int8_t *filter;
    const int32_t *bias;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int8_t *out;
    int32_t input_offset;
    int32_t filter_offset;
    int32_t out_offset;
    int32_t activation_min;
    int32_t activation_max;
    int32_t batches;
    int32_t c0, c1, b0, b1;
    int32_t input_aligned;
    uint16_t row_len;
    uint16_t out_channels;
} fc_batch_n_job_t;

/* Channels [c0, c1), batches [b0, b1). Outer loop is the channel, so the
 * filter row stays live across the batches this core owns. The aligned
 * operand is passed first; the dot is symmetric. */
static void fc_batch_n_range(const fc_batch_n_job_t *j, int32_t c0, int32_t c1,
                             int32_t b0, int32_t b1)
{
    const int32_t row_len = j->row_len;
    const int32_t out_channels = j->out_channels;
    const int32_t row_len_div16 = row_len >> 4;
    const int32_t row_len_rem = row_len & 15;
    const int32_t simd_bytes = row_len_div16 << 4;
    for (int32_t out_c = c0; out_c < c1; ++out_c) {
        const int8_t *filter_row = j->filter + row_len * out_c;
        int32_t corr = 0;
        if (j->input_offset != 0) {
            corr = esp_nn_filter_sum_s8_esp32s3(filter_row, row_len) * j->input_offset;
        }
        if (j->bias) {
            corr += j->bias[out_c];
        }
        const int32_t mult = j->out_mult[out_c];
        const int32_t shift = j->out_shift[out_c];
        for (int32_t b = b0; b < b1; b++) {
            const int8_t *in = j->input + b * row_len;
            int32_t acc = j->input_aligned
                ? esp_nn_dot_s8_unaligned_esp32s3(in, filter_row, row_len_div16)
                : esp_nn_dot_s8_unaligned_esp32s3(filter_row, in, row_len_div16);
            for (int i = 0; i < row_len_rem; i++) {
                acc += (int32_t)in[simd_bytes + i] * (int32_t)filter_row[simd_bytes + i];
            }
            acc += corr;
            acc = esp_nn_multiply_by_quantized_mult(acc, mult, shift);
            acc += j->out_offset;
            acc = max(acc, j->activation_min);
            acc = min(acc, j->activation_max);
            j->out[b * out_channels + out_c] = (int8_t)acc;
        }
    }
}

/* Channels [c0, c1), batches [b0, b1). The per-row kernel owns the filter
 * offset. One call per batch writes a contiguous slice of that batch. */
static void fc_batch_n_per_row(const fc_batch_n_job_t *j, int32_t c0, int32_t c1,
                               int32_t b0, int32_t b1)
{
    const uint16_t count = (uint16_t)(c1 - c0);
    const int8_t *filter = j->filter + (size_t)c0 * j->row_len;
    const int32_t *bias = j->bias ? j->bias + c0 : NULL;
    const int32_t *shift = j->out_shift + c0;
    const int32_t *mult = j->out_mult + c0;
    for (int32_t b = b0; b < b1; b++) {
        esp_nn_fully_connected_per_ch_s8_esp32s3(
                j->input + b * j->row_len, j->input_offset, j->row_len,
                filter, j->filter_offset, bias,
                j->out + b * j->out_channels + c0, count,
                j->out_offset, shift, mult,
                j->activation_min, j->activation_max);
    }
}

static void fc_batch_n_slice(const fc_batch_n_job_t *j, int32_t c0, int32_t c1,
                             int32_t b0, int32_t b1)
{
    if (j->filter_offset != 0) {
        fc_batch_n_per_row(j, c0, c1, b0, b1);
    } else {
        fc_batch_n_range(j, c0, c1, b0, b1);
    }
}

static void fc_batch_n_worker(void *arg)
{
    const fc_batch_n_job_t *j = arg;
    fc_batch_n_slice(j, j->c0, j->c1, j->b0, j->b1);
}

__attribute__((noinline))
void esp_nn_fc_batch_n_split_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max,
                                       const int32_t batches)
{
    const bool input_aligned = (((uintptr_t)input_data & 15) == 0)
                               && ((row_len & 15) == 0);
    /* The public entry already rejected shapes that cannot split. */
    if (!esp_nn_dual_core_begin()) {
        esp_nn_fc_batch_s8_esp32s3(
                input_data, input_offset, row_len, filter_data, filter_offset,
                bias, out_data, out_channels, out_offset, out_shift, out_mult,
                activation_min, activation_max, batches);
        return;
    }
    fc_batch_n_job_t job = {
        .input = input_data, .filter = filter_data, .bias = bias,
        .out_shift = out_shift, .out_mult = out_mult, .out = out_data,
        .input_offset = input_offset, .filter_offset = filter_offset,
        .out_offset = out_offset,
        .activation_min = activation_min, .activation_max = activation_max,
        .batches = batches,
        .c0 = 0, .c1 = out_channels / 2, .b0 = 0, .b1 = batches,
        .input_aligned = input_aligned,
        .row_len = row_len, .out_channels = out_channels,
    };
    if (!esp_nn_dual_core_run(fc_batch_n_worker, &job)) {
        esp_nn_dual_core_end();
        esp_nn_fc_batch_s8_esp32s3(
                input_data, input_offset, row_len, filter_data, filter_offset,
                bias, out_data, out_channels, out_offset, out_shift, out_mult,
                activation_min, activation_max, batches);
        return;
    }
    fc_batch_n_slice(&job, job.c1, out_channels, 0, batches);
    esp_nn_dual_core_wait();
    esp_nn_dual_core_end();
}

/* Dimension and alignment checks only. Worker acquisition stays in the split. */
__attribute__((always_inline)) static inline bool fc_batch_dual_eligible(
        const int8_t *input_data, const int8_t *filter_data,
        uint16_t row_len, uint16_t out_channels, int32_t batches)
{
    const bool input_aligned = (((uintptr_t)input_data & 15) == 0)
                               && ((row_len & 15) == 0);
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);
    /* batches > 1 uses the offset-free dot threshold, matching the batch kernel. */
    return batches >= 2 && out_channels >= ESP_NN_FC_BATCH_DUAL_MIN_OUT &&
            fc_dot_path_wins(row_len, 0) &&
            (input_aligned || filter_rows_aligned);
}
#endif /* S3 dual-core batch N-split */

void esp_nn_fully_connected_per_ch_s8_batch_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max,
                                       const int32_t batches)
{
    /* One row has nothing to reuse across a batch. On a dual-core build the
     * per-channel entry is the batch-1 floor gate. */
    if (batches == 1) {
        esp_nn_fully_connected_per_ch_s8(input_data, input_offset, row_len,
                filter_data, filter_offset, bias, out_data, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max);
        return;
    }
#if ESP_NN_DUAL_CORE_SUPPORTED
    if (fc_batch_dual_eligible(input_data, filter_data, row_len, out_channels, batches)) {
        esp_nn_fc_batch_n_split_esp32s3(input_data, input_offset, row_len, filter_data, filter_offset,
                bias, out_data, out_channels, out_offset, out_shift, out_mult,
                activation_min, activation_max, batches);
        return;
    }
#endif
    esp_nn_fc_batch_s8_esp32s3(input_data, input_offset, row_len, filter_data, filter_offset,
            bias, out_data, out_channels, out_offset, out_shift, out_mult,
            activation_min, activation_max, batches);
}

#if ESP_NN_FC_BATCH_VIA_CONV
/* Weight zero-point 0 on a shape the per-chip gate accepts runs as the 1x1
 * convolution in scratch. Anything else runs the batch kernel above. */
void esp_nn_fully_connected_per_ch_s8_batch_scratch_esp32s3(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches,
                                    void *scratch,
                                    size_t scratch_size)
{
    if (esp_nn_fc_per_ch_s8_batch_via_conv(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult, activation_min,
                                           activation_max, batches,
                                           scratch, scratch_size)) {
        return;
    }
    esp_nn_fully_connected_per_ch_s8_batch(input_data, input_offset, row_len,
            filter_data, filter_offset, bias, out_data, out_channels,
            out_offset, out_shift, out_mult, activation_min, activation_max,
            batches);
}
#endif /* ESP_NN_FC_BATCH_VIA_CONV */
