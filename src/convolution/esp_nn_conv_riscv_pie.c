/*
 * SPDX-FileCopyrightText: 2024-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_riscv_pie function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_riscv_pie)
 *      - For all other cases:
 *          > Consider `filter_wd * in_ch` as a single row. These many values can
 *              be continuosly loaded from inputs as well.
 *          > multiply accumulate into a single filter output.
 *          > To speed things up further, we pre-calculate
 *              (filter * in_offset + bias term) earlier and add it at the end of filter
 *
 *      About ((filter * in_offset + bias term)) accumulate term:
 *          > The conv operation before requantization is as follows:
 *              for i in filter_size:
 *                  conv_out += (input + input_offset) * filter;
 *               conv_out += bias
 *
 *          > where input_offset is constant term hence, we can see that
 *              this term can be precalculated as:
 *                  for i in filter_size:
 *                      acc_term += input_offset * filter[i];
 *                  acc_term += bias
 *              OR
 *                   for i in filter_size:
 *                      acc_term += filter[i]; // accumulate filter values
 *                  acc_term = acc_term * input_offset + bias
 *
 *
 * In both the above versions we align the filter if needed, pad the input with
 *       -input_offset if needed and extend the channels to make those multiple
 *       of 8/16 as per function needs
 */

#include <stddef.h>
#include <stdio.h>
#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include "esp_nn_generic_opt.h"

#include <common_functions.h>
#include "../common/esp_nn_filter_sum_riscv_pie.h"
#include "esp_nn_conv_dispatch_riscv_pie.h"
#include "esp_nn_conv_requant_riscv_pie.h"
#include "esp_nn_conv_tile_requant_riscv_pie.h"
#include "esp_nn_conv_oc16_riscv_pie.h"
#include <esp_nn_multicore.h>

void esp_nn_conv_s8_riscv_pie(const data_dims_t *input_dims,
                              const int8_t *input,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const conv_params_t *conv_params,
                              const quant_data_t *quant_data);

static int16_t *scratch_buffer = NULL;

/*
 * XACC dot of `rows` filter rows against strided input rows, double-buffered
 * like pie_dot_s8 (q0/q2 + q1/q3) so vld results are consumed three
 * instructions later instead of one - the single-buffered per-row loop
 * stalled on every load-use pair. Requires row_size a multiple of 16 and
 * >= 32. Filter rows are contiguous; input rows are in_stride apart.
 * Integer accumulation, same values in a reordered sum: bit-identical.
 */
static int32_t conv_dot_rows_pie(const int8_t *in, const int8_t *filt,
                                 int32_t rows, int32_t row_size,
                                 int32_t in_stride)
{
    const int32_t c32 = (row_size >> 5) - 1;
    const int32_t rem16 = row_size & 16;
    int32_t result;
    asm volatile ("esp.zero.xacc\n\t");
    for (int32_t r = 0; r < rows; r++) {
        asm volatile (
            "mv     x30, %[inp]                     \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "beqz   %[c32], 2f                      \n\t"
            /* for (k = 0; k < c32; k++) xacc += dot(in + 32 * k, flt + 32 * k, 32); */
            ESP_NN_HWLOOP_BEGIN
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            ESP_NN_HWLOOP_LAST("esp.vld.128.ip  q3, x31, 16", 1)
            ESP_NN_HWLOOP_END
            "2:                                     \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"
            "beqz   %[rem], 3f                      \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "3:                                     \n\t"
            :
            : [inp] "r"(in + r * in_stride), [flt] "r"(filt + r * row_size),
              [c32] "r"(c32), [rem] "r"(rem16)
            : "x30", "x31"
        );
    }
    asm volatile (
        "esp.movx.r.xacc.l  x30   \n\t"
        "mv %0, x30               \n\t"
        : "=r" (result)
        :
        : "x30"
    );
    return result;
}


/* Sum of `len` int8 values via the PIE dot against an all-ones vector.
 * Integer addition is associative, so the result is bit-identical to a
 * scalar loop; used to vectorize the filter_sum * input_offset prepasses. */
/* Sum via the shared PIE dot-against-ones helper (see the header for why
 * this must not be re-derived privately). */
static int32_t conv_filter_byte_sum(const int8_t *data, int32_t len)
{
    return esp_nn_filter_sum_s8_riscv_pie(data, len);
}



/**
 * Reusable PIE-accelerated dot product (same as FC version).
 * Processes 32 elements/iter (double-pump) for len >= 32,
 * 16 elements/iter for len >= 16, scalar remainder.
 */
static inline __attribute__((always_inline))
int32_t pie_dot_s8(const int8_t *a, const int8_t *b, int32_t len)
{
    int32_t result = 0;

    if (len >= 32) {
        int32_t c32 = (len >> 5) - 1;
        int32_t rem16 = len & 16;
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "beqz   %[c32], 2f                      \n\t"
            /* for (k = 0; k < c32; k++) xacc += dot(a + 32 * k, b + 32 * k, 32); */
            ESP_NN_HWLOOP_BEGIN
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            ESP_NN_HWLOOP_LAST("esp.vld.128.ip  q3, x31, 16", 1)
            ESP_NN_HWLOOP_END
            "2:                                     \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"
            "beqz   %[rem16], 3f                    \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "3:                                     \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b), [c32] "r"(c32), [rem16] "r"(rem16)
            : "x30", "x31"
        );
    } else if (len >= 16) {
        /* exactly one full 16-element block */
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b)
            : "x30", "x31"
        );
    }

    for (int32_t idx = len & ~15; idx < len; idx++) {
        result += (int32_t)a[idx] * (int32_t)b[idx];
    }
    return result;
}

/* Large pointwise filters do not fit in the target's data cache. The usual
 * pixel-major traversal walks the complete filter once per output pixel,
 * repeatedly fetching the same weights from flash. For small spatial maps,
 * keep one output-channel filter row hot and apply it to every pixel before
 * advancing to the next row. Output writes become strided, but the avoided
 * weight traffic is much larger (for example 15x for a 3x5 output map). */
__attribute__((noinline))
static void conv_1x1_filter_major(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const conv_params_t *conv_params,
                                  const quant_data_t *quant_data)
{
    const int32_t in_channels = input_dims->channels;
    const int32_t out_channels = output_dims->channels;
    const int32_t pixels = output_dims->width * output_dims->height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t output_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    for (int32_t out_ch = 0; out_ch < out_channels; ++out_ch) {
        const int8_t *filter = filter_data + out_ch * in_channels;
        if (out_ch + 1 < out_channels) {
            /* Touch the next row, one load per 64-byte line. This layer
             * is bound by PSRAM latency. Without the touch its time moved
             * by 2% with the placement of code and data between builds. */
            const int8_t *next = filter + in_channels;
            for (int32_t i = 0; i < in_channels; i += 64) {
                asm volatile ("lb x0, 0(%0)" :: "r"(next + i));
            }
        }
        int32_t filter_sum = 0;
        if (input_offset != 0) {
            /* Vectorized like the other prepasses. It matters more here: this
             * path is gated on a small spatial map, so the per-channel sum is
             * a sizeable fraction of the work, not a negligible prepass. */
            filter_sum = conv_filter_byte_sum(filter, in_channels) * input_offset;
        }
        const int32_t base = filter_sum + (bias ? bias[out_ch] : 0);
        const int32_t multiplier = quant_data->mult[out_ch];
        const int32_t shift = quant_data->shift[out_ch];

        for (int32_t pixel = 0; pixel < pixels; ++pixel) {
            const int8_t *input = input_data + pixel * in_channels;
            int32_t result = pie_dot_s8(input, filter, in_channels) + base;
            result = esp_nn_requantize(result, multiplier, shift);
            result += output_offset;
            result = max(result, activation_min);
            result = min(result, activation_max);
            out_data[pixel * out_channels + out_ch] = (int8_t) result;
        }
    }
}

/*
 * qacc += sum over k < cnt of trans[16 k .. 16 k + 15] * filt[k] (broadcast).
 * The QACC is not zeroed here. The core issues one instruction per cycle,
 * so the code counts instructions: a fused MAC + load and a broadcast load
 * per tap, with loads ahead in register pairs (no load-use stall).
 * An odd tap runs first. The pipeline then holds two taps; the remaining
 * tap pairs run as 1, 2 and 4 pairs by their bits, then 8 pairs per
 * hardware-loop pass. The loop ends on its last load (ESP_NN_HWLOOP_LAST).
 * No load runs past the last tap.
 */
#define QACC16_MAC_PAIR \
        "esp.vmulas.s8.qacc.ld.ip q0, x30, 16, q0, q1 \n\t" \
        "esp.vldbc.8.xp  q1, x31, t3         \n\t" \
        "esp.vmulas.s8.qacc.ld.ip q2, x30, 16, q2, q3 \n\t" \
        "esp.vldbc.8.xp  q3, x31, t3         \n\t"
#define QACC16_MAC_QUAD \
        "esp.vmulas.s8.qacc.ld.ip q4, x30, 16, q0, q1 \n\t" \
        "esp.vldbc.8.xp  q5, x31, t3         \n\t" \
        "esp.vmulas.s8.qacc.ld.ip q6, x30, 16, q2, q3 \n\t" \
        "esp.vldbc.8.xp  q7, x31, t3         \n\t" \
        "esp.vmulas.s8.qacc.ld.ip q0, x30, 16, q4, q5 \n\t" \
        "esp.vldbc.8.xp  q1, x31, t3         \n\t" \
        "esp.vmulas.s8.qacc.ld.ip q2, x30, 16, q6, q7 \n\t" \
        "esp.vldbc.8.xp  q3, x31, t3         \n\t"

static inline __attribute__((always_inline))
void qacc16_mac_continue(const int8_t *trans, const int8_t *filt, int32_t cnt)
{
    /* Tap pairs after the two taps the pipeline preloads. */
    const int32_t rest = (cnt >> 1) - 1;
    asm volatile (
        "li     t3, 1                        \n\t"
        "mv     x30, %[trans]                \n\t"
        "mv     x31, %[flt]                  \n\t"
        "beqz   %[odd], 1f                   \n\t"
        "esp.vld.128.ip  q0, x30, 16         \n\t"
        "esp.vldbc.8.xp  q1, x31, t3         \n\t"
        "esp.vmulas.s8.qacc q0, q1           \n\t"
        "1:                                  \n\t"
        "bltz   %[rest], 9f                  \n\t"
        "esp.vld.128.ip  q0, x30, 16         \n\t"
        "esp.vldbc.8.xp  q1, x31, t3         \n\t"
        "esp.vld.128.ip  q2, x30, 16         \n\t"
        "esp.vldbc.8.xp  q3, x31, t3         \n\t"
        "beqz   %[p1], 2f                    \n\t"
        QACC16_MAC_PAIR
        "2:                                  \n\t"
        "beqz   %[p2], 3f                    \n\t"
        QACC16_MAC_QUAD
        "3:                                  \n\t"
        "beqz   %[p4], 4f                    \n\t"
        QACC16_MAC_QUAD
        QACC16_MAC_QUAD
        "4:                                  \n\t"
        "beqz   %[p8], 5f                    \n\t"
        /* for (j = 0; j < p8; j++) sixteen taps */
        ESP_NN_HWLOOP_BEGIN
        "esp.lp.setup 0, %[p8], 1f           \n\t"
        QACC16_MAC_QUAD
        QACC16_MAC_QUAD
        QACC16_MAC_QUAD
        "esp.vmulas.s8.qacc.ld.ip q4, x30, 16, q0, q1 \n\t"
        "esp.vldbc.8.xp  q5, x31, t3         \n\t"
        "esp.vmulas.s8.qacc.ld.ip q6, x30, 16, q2, q3 \n\t"
        "esp.vldbc.8.xp  q7, x31, t3         \n\t"
        "esp.vmulas.s8.qacc.ld.ip q0, x30, 16, q4, q5 \n\t"
        "esp.vldbc.8.xp  q1, x31, t3         \n\t"
        "esp.vmulas.s8.qacc.ld.ip q2, x30, 16, q6, q7 \n\t"
        ESP_NN_HWLOOP_LAST("esp.vldbc.8.xp  q3, x31, t3", 0)
        ESP_NN_HWLOOP_END
        "5:                                  \n\t"
        "esp.vmulas.s8.qacc q0, q1           \n\t"
        "esp.vmulas.s8.qacc q2, q3           \n\t"
        "9:                                  \n\t"
        :
        : [trans] "r"(trans), [flt] "r"(filt), [odd] "r"(cnt & 1),
          [rest] "r"(rest), [p1] "r"(rest & 1), [p2] "r"(rest & 2),
          [p4] "r"(rest & 4), [p8] "r"(rest >> 3)
        : "x30", "x31", "t3", "memory"
    );
}

/* Fills tab[words * oc ..] with the fast-requant constants of each
 * channel and returns tab when every channel in [0, n_oc) takes the fast
 * requant, else NULL. words is 3, or 4 to add e1 = 1 - shift for the oc16
 * kernel. Built once per call, so the per-channel loop below only loads. */
static inline __attribute__((always_inline))
const int32_t *qacc16_requant_table_w(const int32_t *filter_sum,
                                      const int32_t *bias,
                                      const int32_t *mult,
                                      const int32_t *shift, int32_t n_oc,
                                      int32_t count, int32_t out_offset,
                                      int32_t *tab, int32_t words)
{
    int32_t *p = tab;
    for (int32_t oc = 0; oc < n_oc; oc++, p += words) {
        const int32_t comb = filter_sum[oc] + (bias ? bias[oc] : 0);
        if (!esp_nn_conv_requant_fast_ok(comb, mult[oc], shift[oc],
                                         count * 128 * 128, out_offset)) {
            return NULL;
        }
        esp_nn_conv_requant_params(comb, mult[oc], shift[oc], out_offset, p);
        if (words == 4) {
            p[3] = 1 - shift[oc];
        }
    }
    return tab;
}

static const int32_t *qacc16_requant_table(const int32_t *filter_sum,
                                           const int32_t *bias,
                                           const int32_t *mult,
                                           const int32_t *shift,
                                           int32_t n_oc, int32_t count,
                                           int32_t out_offset, int32_t *tab)
{
    return qacc16_requant_table_w(filter_sum, bias, mult, shift, n_oc,
                                  count, out_offset, tab, 3);
}

/*
 * Runs output channels [oc_start, oc_end) of a 16-pixel channel-major
 * staging: one QACC pass per channel (dot length `count`), requantize,
 * scatter to out_base[pixel * out_stride + oc]. `tab` comes from
 * qacc16_requant_table (NULL: generic requant).
 */
static void qacc16_run_channels(const int8_t *t, const int8_t *filt_base,
                                int32_t count, int32_t oc_start,
                                int32_t oc_end, const int32_t *filter_sum,
                                const int32_t *bias, const int32_t *out_mult,
                                const int32_t *out_shift, int8_t *out_base,
                                int32_t out_stride, int32_t out_offset,
                                int32_t act_min, int32_t act_max,
                                int32_t lanes, const int32_t *tab)
{
    /* The MAC asm reads t and filt_base through pointers and has no memory
     * operands. Every store that staged t must land before it. */
    asm volatile ("" ::: "memory");
    const int8_t *filt = filt_base + oc_start * count;
    int32_t results[16] __attribute__((aligned(16)));
    int32_t oc = oc_start;
    if (tab && lanes == 16) {
        /* Full batches: eight channels per tile, requantized on PIE and
         * stored as one 8-byte row per pixel. */
        int8_t tile[8 * 16] __attribute__((aligned(16)));
        const int32_t *prm = tab + 3 * oc;
        for (; oc + 8 <= oc_end; oc += 8) {
            for (int32_t j = 0; j < 8; j++, prm += 3) {
                asm volatile ("esp.zero.qacc\n\t");
                qacc16_mac_continue(t, filt, count);
                esp_nn_conv_requant_tile_row(results, prm,
                        1 - out_shift[oc + j], act_min, act_max,
                        tile + 16 * j);
                filt += count;
            }
            esp_nn_conv_tile8x16_store(tile, out_base + oc, out_stride);
        }
    }
    /* Channel tails and partial batches: one channel at a time, with
     * strided byte stores. |in * filter| <= 128 * 128 per tap. */
    const int32_t acc_bound = count * 128 * 128;
    for (; oc < oc_end; oc++) {
        asm volatile ("esp.zero.qacc\n\t");
        qacc16_mac_continue(t, filt, count);
        ESP_NN_QACC_EXTRACT_S32(results);
        if (tab) {
            esp_nn_conv_requant_lanes_fast(results, lanes, tab + 3 * oc,
                    out_shift[oc], out_base + oc, out_stride, act_min,
                    act_max);
        } else {
            const int32_t combined = filter_sum[oc] + (bias ? bias[oc] : 0);
            esp_nn_conv_requant_lanes_pie(results, lanes, combined,
                    out_mult[oc], out_shift[oc], acc_bound, out_base + oc,
                    out_stride, out_offset, act_min, act_max);
        }
        filt += count;
    }
}

static void im2col_stage_lanes16(const int8_t *const *bases, int32_t off,
                                 int32_t n, int8_t *dst);

/* Transpose 16 contiguous pixels to channel-major: t[c * 16 + p] =
 * in[p * in_ch + c]. The PIE zip transpose of the im2col staging does
 * 8-byte multiples (about 0.3 instructions per byte), bytes do the rest. */
static void qacc16_stage_pixels(const int8_t *input, int32_t in_ch, int8_t *t)
{
    const int8_t *bases[16];
    for (int p = 0; p < 16; p++) {
        bases[p] = input + p * in_ch;
    }
    im2col_stage_lanes16(bases, 0, in_ch, t);
    for (int32_t c = in_ch & ~7; c < in_ch; c++) {
        for (int p = 0; p < 16; p++) {
            t[c * 16 + p] = input[p * in_ch + c];
        }
    }
}

/*
 * 1x1 for in_ch < 16: 16 pixels are transposed to channel-major on the
 * stack, then each output channel runs one QACC pass (lanes = pixels).
 * The 16 pixels are contiguous, so their outputs are out[p * out_ch].
 */
__attribute__((noinline))
static void conv_1x1_batch16(const int8_t *input, const int8_t *filter_data,
                             const int32_t *filter_sum, const int32_t *bias,
                             int8_t *out, int32_t in_ch, int32_t out_ch,
                             int32_t out_offset, const int32_t *out_mult,
                             const int32_t *out_shift, int32_t act_min,
                             int32_t act_max, const int32_t *tab)
{
    int8_t transposed[16 * 16] __attribute__((aligned(16)));
    qacc16_stage_pixels(input, in_ch, transposed);
    qacc16_run_channels(transposed, filter_data, in_ch, 0, out_ch,
                        filter_sum, bias, out_mult, out_shift, out, out_ch,
                        out_offset, act_min, act_max, 16, tab);
}

/*
 * Stages 16 channels [c0, c0 + 16) of 8 contiguous pixels, transposed:
 * t[c * 16 + p] for c in the block and p < 8, at dst = t + c0 * 16 + p0.
 * Eight loads one pixel apart, a 3-stage zip network (8, 16, 32 bit) and
 * sixteen 8-byte stores. The loads may be unaligned (ESP_NN_PIE_ENABLE).
 * Overwrites q0-q7. GCC has no names for the PIE q registers, so they are
 * not in the clobber list, and the caller keeps nothing live in them.
 */
static inline __attribute__((always_inline))
void conv_stage8x16_pie(const int8_t *src, int32_t stride, int8_t *dst)
{
    asm volatile (
        "mv     x30, %[s]                    \n\t"
        "mv     x29, %[st]                   \n\t"
        "mv     x31, %[d]                    \n\t"
        "esp.vld.128.xp  q0, x30, x29        \n\t"
        "esp.vld.128.xp  q1, x30, x29        \n\t"
        "esp.vld.128.xp  q2, x30, x29        \n\t"
        "esp.vld.128.xp  q3, x30, x29        \n\t"
        "esp.vld.128.xp  q4, x30, x29        \n\t"
        "esp.vld.128.xp  q5, x30, x29        \n\t"
        "esp.vld.128.xp  q6, x30, x29        \n\t"
        "esp.vld.128.xp  q7, x30, x29        \n\t"
        /* Byte pairs (pixels 2k, 2k + 1), then quads, then all 8 pixels. */
        "esp.vzip.8  q0, q1                  \n\t"
        "esp.vzip.8  q2, q3                  \n\t"
        "esp.vzip.8  q4, q5                  \n\t"
        "esp.vzip.8  q6, q7                  \n\t"
        "esp.vzip.16 q0, q2                  \n\t"
        "esp.vzip.16 q1, q3                  \n\t"
        "esp.vzip.16 q4, q6                  \n\t"
        "esp.vzip.16 q5, q7                  \n\t"
        "esp.vzip.32 q0, q4                  \n\t"
        "esp.vzip.32 q2, q6                  \n\t"
        "esp.vzip.32 q1, q5                  \n\t"
        "esp.vzip.32 q3, q7                  \n\t"
        /* Channels 0-1, 2-3, ... 14-15 are in q0 q4 q2 q6 q1 q5 q3 q7. */
        "esp.vst.l.64.ip q0, x31, 16         \n\t"
        "esp.vst.h.64.ip q0, x31, 16         \n\t"
        "esp.vst.l.64.ip q4, x31, 16         \n\t"
        "esp.vst.h.64.ip q4, x31, 16         \n\t"
        "esp.vst.l.64.ip q2, x31, 16         \n\t"
        "esp.vst.h.64.ip q2, x31, 16         \n\t"
        "esp.vst.l.64.ip q6, x31, 16         \n\t"
        "esp.vst.h.64.ip q6, x31, 16         \n\t"
        "esp.vst.l.64.ip q1, x31, 16         \n\t"
        "esp.vst.h.64.ip q1, x31, 16         \n\t"
        "esp.vst.l.64.ip q5, x31, 16         \n\t"
        "esp.vst.h.64.ip q5, x31, 16         \n\t"
        "esp.vst.l.64.ip q3, x31, 16         \n\t"
        "esp.vst.h.64.ip q3, x31, 16         \n\t"
        "esp.vst.l.64.ip q7, x31, 16         \n\t"
        "esp.vst.h.64.ip q7, x31, 16         \n\t"
        :
        : [s] "r"(src), [st] "r"(stride), [d] "r"(dst)
        : "x29", "x30", "x31", "memory"
    );
}

/*
 * Transposes 16 contiguous pixels to channel-major, for in_ch >= 16:
 * transposed[c * 16 + p] = input[p * in_ch + c]. Each 16-channel block is
 * two passes of conv_stage8x16_pie, one per 8 pixels. A partial last block
 * is one more block on the last 16 channels: it rewrites the overlap with
 * the same bytes. Out of line, so its code stays out of the caller's loops.
 */
__attribute__((noinline))
static void conv_1x1_stage16(const int8_t *input, int32_t in_ch,
                             int8_t *transposed)
{
    for (int32_t c0 = 0; c0 + 16 <= in_ch; c0 += 16) {
        int8_t *dst = transposed + c0 * 16;
        conv_stage8x16_pie(input + c0, in_ch, dst);
        conv_stage8x16_pie(input + 8 * in_ch + c0, in_ch, dst + 8);
    }
    if (in_ch & 15) {
        const int32_t c0 = in_ch - 16;
        int8_t *dst = transposed + c0 * 16;
        conv_stage8x16_pie(input + c0, in_ch, dst);
        conv_stage8x16_pie(input + 8 * in_ch + c0, in_ch, dst + 8);
    }
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_1x1(const data_dims_t *input_dims,
                               const int8_t *input_data,
                               const int8_t *filter_data,
                               const int32_t *bias,
                               const data_dims_t *output_dims,
                               int8_t *out_data,
                               const conv_params_t *conv_params,
                               const quant_data_t *quant_data,
                               void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Unaligned vld/vst in the staging and the tile stores. CFG is per core
     * and not saved per task, and the dual-core worker enters here. */
    ESP_NN_PIE_ENABLE();

    const int32_t filter_bytes = in_channels * out_channels;
    const int32_t output_pixels = out_wd * out_ht;
    if (filter_bytes >= 64 * 1024 && output_pixels <= 16) {
        conv_1x1_filter_major(input_dims, input_data, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data);
        return;
    }

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    const int8_t *filter_ptr = filter_data;
    for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
        filter_sum[out_ch_idx] = conv_filter_byte_sum(
                filter_data + out_ch_idx * in_channels, in_channels) * input_offset;
    }

    /* When in_ch < 16: use QACC batch path (16 pixels at once) or channel padding.
     * QACC batch: transpose pixels, broadcast filter, per-lane MAC.
     * Channel pad: pad in/filter to 16 ch for XACC. */
    /* When in_ch < 16: use QACC batch (16 pixels at a time with broadcast filter).
     * Falls back to channel-padding for remaining pixels. */
    if (in_channels < 16) {
        int32_t total_pixels = out_wd * out_ht;
        int32_t pix = 0;

        /* Process batches of 16 pixels using QACC per-lane */
        /* filter_sum, then the requant table: 16 of the 20 bytes per
         * channel the getter gives this path. */
        const int32_t *tab = qacc16_requant_table(filter_sum, bias,
                quant_data->mult, quant_data->shift, out_channels,
                in_channels, out_offset, filter_sum + out_channels);
        for (; pix <= total_pixels - 16; pix += 16) {
            conv_1x1_batch16(input_data + pix * in_channels, filter_data,
                             filter_sum, bias, out_data + pix * out_channels,
                             in_channels, out_channels, out_offset,
                             quant_data->mult, quant_data->shift,
                             activation_min, activation_max, tab);
        }

        /* Remaining pixels (< 16): scalar fallback */
        for (; pix < total_pixels; pix++) {
            const int8_t *inp = input_data + pix * in_channels;
            filter_ptr = filter_data;
            for (int32_t oc = 0; oc < out_channels; oc++) {
                int32_t conv_out = 0;
                for (int32_t ic = 0; ic < in_channels; ic++) {
                    conv_out += inp[ic] * filter_ptr[ic];
                }
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_multiply_by_quantized_mult(conv_out,
                    quant_data->mult[oc], quant_data->shift[oc]);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                out_data[pix * out_channels + oc] = (int8_t) conv_out;
                filter_ptr += in_channels;
            }
        }
        return;
    }

    /* OC-panel tiling (see esp_nn_conv_s8_padded): sweep a panel of output
     * channels across all pixels so the panel's filter rows stay resident in
     * the 64 KB L1 D-cache; filter data is fetched from memory once per
     * layer instead of once per pixel. Arithmetic order per output element
     * is unchanged: bit-identical results. */
    int32_t oc_tile = out_channels;
    if ((int32_t)out_channels * in_channels > 24 * 1024) {
        oc_tile = (24 * 1024) / in_channels;
        if (oc_tile < 4) {
            oc_tile = 4;
        }
    }

    int32_t loop_ht = out_ht;
    int32_t loop_wd = out_wd;
    int32_t loop_in_wd = input_wd;

    /* Position-batched fast path: 16 pixels per QACC pass (lanes = pixels,
     * broadcast filter byte per input channel). Requires contiguous pixels,
     * which holds for the stride-1 unpadded 1x1 (input_wd == out_wd). The
     * per-pixel dot setup of the XACC path below disappears; the pixel tail
     * (< 16) falls through to that path. */
    if (input_wd == out_wd) {
        const int32_t total_pixels = (int32_t)out_wd * out_ht;
        const int32_t batched = total_pixels & ~15;
        if (batched > 0) {
            /* Scratch: filter_sum, the requant table (12 bytes per
             * channel), then the staging. The getter's filter_scratch is
             * at least 16 bytes per channel here (in_ch >= 16). */
            const int32_t *tab = qacc16_requant_table(filter_sum, bias,
                    quant_data->mult, quant_data->shift, out_channels,
                    in_channels, out_offset, filter_sum + out_channels);
            int8_t *transposed = (int8_t *)(((uintptr_t)((int8_t *)scratch
                    + out_channels * 16 + 15)) & ~(uintptr_t)15);
            for (int32_t pix = 0; pix < batched; pix += 16) {
                int8_t *out_batch = out_data + pix * out_channels;
                /* Staged once per batch, then used by every panel. */
                conv_1x1_stage16(input_data + pix * in_channels, in_channels,
                                 transposed);
                for (int32_t oc_base = 0; oc_base < out_channels;
                        oc_base += oc_tile) {
                    const int32_t oc_end = min(oc_base + oc_tile, out_channels);
                    qacc16_run_channels(transposed, filter_data, in_channels,
                                        oc_base, oc_end, filter_sum, bias,
                                        quant_data->mult, quant_data->shift,
                                        out_batch, out_channels, out_offset,
                                        activation_min, activation_max, 16,
                                        tab);
                }
            }
            if (batched == total_pixels) {
                return;
            }
            /* Tail (< 16 pixels): fall through to the XACC path over a
             * single row of the remaining contiguous pixels. */
            input_data += batched * in_channels;
            out_data += batched * out_channels;
            loop_ht = 1;
            loop_wd = total_pixels - batched;
            loop_in_wd = loop_wd;
        }
    }

    for (int32_t oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
    const int32_t oc_end = min(oc_base + oc_tile, out_channels);
    for (int32_t in_row = 0; in_row < loop_ht; in_row++) {
        for (int32_t in_col = 0; in_col < loop_wd; in_col++) {
            const int32_t *out_mult = quant_data->mult + oc_base;
            const int32_t *out_shift = quant_data->shift + oc_base;
            filter_ptr = filter_data + oc_base * in_channels;
            const int8_t *input_base_ptr = input_data + (in_row * loop_in_wd + in_col) * in_channels;
            int8_t *out_ptr = out_data
                    + ((int32_t)(in_row * loop_wd + in_col)) * out_channels + oc_base;
            for (int32_t out_ch_idx = oc_base; out_ch_idx < oc_end; out_ch_idx++) {
                /* initializations */
                int32_t conv_out = 0;
                const int8_t *input_ptr = input_base_ptr;

                int32_t in_ch_idx = 0;
#if 1 // inline asm
                // for now check for the alignment as well
                if (in_channels < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                    goto skip_asm;
                }

                int32_t c16 = (in_channels >> 4) - 1;
                asm volatile (
                    "mv x30, %[inp]                 \n\t"
                    "mv x31, %[flt]                 \n\t"
                    "esp.zero.xacc                  \n\t"
                    "esp.vld.128.ip  q0, x30, 16    \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"

                    "beqz %[c16], 2f                \n\t"
                    /* for (k = 0; k < c16; k++) xacc += dot(in + 16 * k, flt + 16 * k, 16); */
#if ESP_NN_HWLOOP_PADDED
                    /* The padded hardware loop runs 6 instructions per
                     * iteration here, this software loop 4. */
                    "mv   x28, %[c16]               \n\t"
                    "1:                             \n\t"
                    "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"
                    "addi x28, x28, -1              \n\t"
                    "bnez x28, 1b                   \n\t"
#else
                    ESP_NN_HWLOOP_BEGIN
                    "esp.lp.setup 0, %[c16], 1f     \n\t"
                    "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                    ESP_NN_HWLOOP_LAST("esp.vld.128.ip  q1, x31, 16", 3)
                    ESP_NN_HWLOOP_END
#endif
                    "2:                             \n\t"

                    /* input_ptr = x30; filter_ptr = x31; conv_out = xacc; */
                    "mv %[inp], x30                 \n\t"
                    "mv %[flt], x31                 \n\t"
                    "esp.vmulas.s8.xacc  q0, q1     \n\t"
                    /* esp.movx GPR operand must be x26-x31 (required on S31) */
                    "esp.movx.r.xacc.l  x29         \n\t"
                    "mv %[out], x29                 \n\t"

                    : [inp] "+r" (input_ptr), [flt] "+r" (filter_ptr), [out] "=r" (conv_out)
                    : [c16] "r"(c16)
                    : "x28", "x29", "x30", "x31"
                );
                in_ch_idx = in_channels & ~15;
skip_asm:
#endif
                for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                }

                for (; in_ch_idx < in_channels; in_ch_idx++) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                }
                conv_out = conv_out + filter_sum[out_ch_idx];
                if (bias) {
                    conv_out += bias[out_ch_idx];
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
            }
        }
    }
    }
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_padded(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    {
        const int32_t filter_len = filter_wd * filter_ht * in_channels;
        for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
            filter_sum[out_ch_idx] = conv_filter_byte_sum(
                    filter_data + out_ch_idx * filter_len, filter_len) * input_offset;
        }
    }

    const int32_t row_size = filter_wd * in_channels;

    /* Interior extent: output columns/rows whose whole filter window lies
     * inside the input. Anything beyond falls in TFLite's implicit trailing
     * padding (clipped taps), handled by the edge paths below. */
    /* Guard the truncating division: an input smaller than the filter has
     * no fully-inside output at all, not one. */
    int32_t eff_wd = (input_wd >= filter_wd) ? (input_wd - filter_wd) / stride_wd + 1 : 0;
    int32_t eff_ht = (input_ht >= filter_ht) ? (input_ht - filter_ht) / stride_ht + 1 : 0;
    if (eff_wd > out_wd) eff_wd = out_wd;
    if (eff_ht > out_ht) eff_ht = out_ht;
    const bool right_pad = eff_wd < out_wd;
    const bool bottom_pad = eff_ht < out_ht;

    /*
     * OC-panel tiling: sweep a panel of output channels across all output
     * pixels before moving to the next panel. The panel's filters
     * (oc_tile * filter_size bytes) stay resident in the L1 D-cache
     * (P4 TRM 9.3.3.2: 64 KB dcache, 2-way, 64 B lines) for the whole pixel
     * sweep, so filter data is fetched from memory once per layer instead of
     * once per output pixel. Arithmetic order per output element is
     * unchanged: bit-identical results.
     */
    const int32_t filter_size = filter_ht * row_size;
    int32_t oc_tile = out_channels;
    if ((int32_t)out_channels * filter_size > 24 * 1024 && filter_size > 0) {
        oc_tile = (24 * 1024) / filter_size;
        if (oc_tile < 4) {
            oc_tile = 4;
        }
    }


    for (int32_t oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
        const int32_t oc_end = min(oc_base + oc_tile, out_channels);
        const int8_t *panel_filter = filter_data + oc_base * filter_size;
        for (int32_t out_y = 0; out_y < eff_ht; out_y++) {
            const int32_t base_y = stride_ht * out_y;
            for (int32_t out_x = 0; out_x < eff_wd; out_x++) {
                const int32_t base_x = stride_wd * out_x;
                const int8_t *filter_data_ptr = panel_filter;
                int8_t *out_ptr = out_data
                        + ((int32_t)out_y * out_wd + out_x) * out_channels + oc_base;
                for (int32_t out_ch_idx = oc_base; out_ch_idx < oc_end; out_ch_idx++) {
                    int32_t conv_out = 0, filter_y_idx;
                    if (row_size >= 32 && (row_size & 15) == 0) {
                        /* Double-buffered fused-rows dot (no per-row scalar
                         * tail needed when row_size is a multiple of 16). */
                        conv_out = conv_dot_rows_pie(
                                input_data + ((int32_t)base_y * input_wd
                                              + base_x) * in_channels,
                                filter_data_ptr, filter_ht, row_size,
                                (int32_t)input_wd * in_channels);
                        filter_data_ptr += (int32_t)filter_ht * row_size;
                        conv_out += filter_sum[out_ch_idx];
                        if (bias) {
                            conv_out += bias[out_ch_idx];
                        }
                        conv_out = esp_nn_requantize(conv_out,
                                                     out_mult[out_ch_idx],
                                                     out_shift[out_ch_idx]);
                        conv_out += out_offset;
                        conv_out = max(conv_out, activation_min);
                        conv_out = min(conv_out, activation_max);
                        *out_ptr++ = (int8_t) conv_out;
                        continue;
                    }
                    if (row_size >= 16) {
                        asm volatile("esp.zero.xacc                  \n\t");
                    }

                    for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x;
                        const int8_t *input_data_ptr =
                                input_data + (in_row * input_wd + in_col) * in_channels;
                        int32_t row_idx = 0;
#if 1 // inline asm
                    // for now check for the alignment as well
                    if (row_size < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                        goto skip_asm_pad0;
                    }

                    {
                    int32_t c16 = (row_size >> 4) - 1;
                    /* Two loop forms, identical arithmetic: the zero-overhead
                     * hardware loop (as in the 1x1 kernel) drops the addi/bnez
                     * pair per 16-byte chunk and pays off once the trip count
                     * amortizes its setup; the software loop stays for short
                     * rows (small in_ch), where esp.lp.setup measured slower
                     * on ESP32-S31. */
                    if (c16 >= 4) {
                        asm volatile (
                            "mv x30, %[inp]                 \n\t"
                            "mv x31, %[flt]                 \n\t"
                            "esp.vld.128.ip  q0, x30, 16    \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"

                            /* for (k = 0; k < c16; k++) xacc += dot(in + 16 * k, flt + 16 * k, 16); */
                            ESP_NN_HWLOOP_BEGIN
                            "esp.lp.setup 0, %[c16], 1f     \n\t"
                            "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                            ESP_NN_HWLOOP_LAST("esp.vld.128.ip  q1, x31, 16", 3)
                            ESP_NN_HWLOOP_END

                            /* input_data_ptr = x30; filter_data_ptr = x31; */
                            "mv %[inp], x30                 \n\t"
                            "mv %[flt], x31                 \n\t"
                            "esp.vmulas.s8.xacc  q0, q1     \n\t"

                            : [inp] "+r" (input_data_ptr), [flt] "+r" (filter_data_ptr)
                            : [c16] "r"(c16)
                            : "x30", "x31"
                        );
                    } else {
                        asm volatile (
                            "mv x30, %[inp]                 \n\t"
                            "mv x31, %[flt]                 \n\t"
                            "esp.vld.128.ip  q0, x30, 16    \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"

                            "beqz %[c16], 2f                \n\t"
                            "mv   s7, %[c16]                \n\t"
                            "1:                             \n\t"
                            "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"
                            "addi s7, s7, -1                \n\t"
                            "bnez s7, 1b                    \n\t"
                            "2:                             \n\t"

                            /* input_data_ptr = x30; filter_data_ptr = x31; */
                            "mv %[inp], x30                 \n\t"
                            "mv %[flt], x31                 \n\t"
                            "esp.vmulas.s8.xacc  q0, q1     \n\t"

                            : [inp] "+r" (input_data_ptr), [flt] "+r" (filter_data_ptr)
                            : [c16] "r"(c16)
                            : "x30", "x31", "s7"
                        );
                    }
                    row_idx = row_size & ~15;
                    }
skip_asm_pad0:
#endif
                        for (; row_idx < row_size - 3; row_idx += 4) {
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        }
                        for (; row_idx < row_size; row_idx++) {
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        }
                    }
                    if (row_size >= 16) {
                        asm volatile (
                            "esp.movx.r.xacc.l  x30   \n\t"
                            "add %0, %0, x30          \n\t"
                            : "+r" (conv_out)
                            :
                            : "x30"
                        );
                    }
                    /* add input_offset term */
                    conv_out += filter_sum[out_ch_idx];

                    if (bias) {
                        conv_out += bias[out_ch_idx];
                    }
                    conv_out = esp_nn_requantize(conv_out, out_mult[out_ch_idx],
                                                 out_shift[out_ch_idx]);
                    conv_out += out_offset;
                    conv_out = max(conv_out, activation_min);
                    conv_out = min(conv_out, activation_max);
                    *out_ptr++ = (int8_t) conv_out;
                }
            }
        }
    }

    /* Right-edge columns whose filter window runs past the input: scalar
     * path with the clipped tap count, as before (few pixels). */
    for (int32_t out_y = 0; out_y < eff_ht && right_pad; out_y++) {
        for (int32_t out_x = eff_wd; out_x < out_wd; out_x++) {
            const int32_t base_y = stride_ht * out_y;
            const int32_t base_x = stride_wd * out_x;
            const int32_t *out_mult_ptr = out_mult;
            const int32_t *out_shift_ptr = out_shift;
            const int32_t *bias_ptr = bias;
            int8_t *out_ptr = out_data
                    + ((int32_t)out_y * out_wd + out_x) * out_channels;
            for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0, filter_y_idx;
                /* Clip taps per pixel: the overhang grows toward the edge. */
                const int32_t fx_end = min(filter_wd, input_wd - base_x);
                for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                    for (int32_t filter_x_idx = 0; filter_x_idx < fx_end; filter_x_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x + filter_x_idx;

                        const int8_t *input_ptr = input_data +
                                        (in_row * input_wd + in_col) * in_channels;
                        const int8_t *filter_ptr = filter_data +
                                        out_ch_idx * in_channels * filter_ht * filter_wd +
                                        (filter_y_idx * filter_wd + filter_x_idx) * in_channels;
                        int32_t in_ch_idx = 0;
                        for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                        for (; in_ch_idx < in_channels; in_ch_idx ++) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                    }
                }

                if (bias) {
                    conv_out += *bias_ptr++;
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
            }
        }
    }

    // Calculate the last row if needed. Hand the remaining input rows to the
    // generic kernel, which clamps filter windows to the input extent (the
    // rows falling in the implicit trailing padding contribute zero).
    if (bottom_pad) {
        const int32_t in_row = eff_ht * stride_ht;
        const int32_t rows = out_ht - eff_ht;
        esp_nn_conv_s8_opt(&(data_dims_t){input_dims->width, input_dims->height - in_row,
                                          input_dims->channels, 0},
                            input_data + in_row * input_dims->width * input_dims->channels,
                            filter_dims, filter_data, bias,
                            &(data_dims_t){output_dims->width, (uint16_t)rows, output_dims->channels, 0},
                            out_data + (int32_t)eff_ht * out_wd * out_channels,
                            conv_params, quant_data);
    }
}



/*
 * Stages 16 window bytes [off, off + 16) of 8 lanes, transposed:
 * dst[i * 16 + p] = bases[p][off + i] for i < 16, p < 8. Eight unaligned
 * loads, a 3-stage zip network (8, 16, 32 bit) and sixteen 8-byte stores:
 * about 0.3 instructions per byte against about 3 for byte copies.
 * Each load stays inside its lane's window row, so inside the input.
 */
static inline __attribute__((always_inline))
void im2col_stage8x16(const int8_t *const *bases, int32_t off, int8_t *dst)
{
    asm volatile (
        "mv     x29, %[b]                    \n\t"
        "lw     x30, 0(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q0, x30, 0          \n\t"
        "lw     x30, 4(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q1, x30, 0          \n\t"
        "lw     x30, 8(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q2, x30, 0          \n\t"
        "lw     x30, 12(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q3, x30, 0          \n\t"
        "lw     x30, 16(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q4, x30, 0          \n\t"
        "lw     x30, 20(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q5, x30, 0          \n\t"
        "lw     x30, 24(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q6, x30, 0          \n\t"
        "lw     x30, 28(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.128.ip  q7, x30, 0          \n\t"
        "mv     x31, %[dst]                  \n\t"
        ESP_NN_PIE_ZIP8X16
        /* Window bytes 0-1, 2-3, ... 14-15 are in q0 q4 q2 q6 q1 q5 q3 q7. */
        "esp.vst.l.64.ip q0, x31, 16         \n\t"
        "esp.vst.h.64.ip q0, x31, 16         \n\t"
        "esp.vst.l.64.ip q4, x31, 16         \n\t"
        "esp.vst.h.64.ip q4, x31, 16         \n\t"
        "esp.vst.l.64.ip q2, x31, 16         \n\t"
        "esp.vst.h.64.ip q2, x31, 16         \n\t"
        "esp.vst.l.64.ip q6, x31, 16         \n\t"
        "esp.vst.h.64.ip q6, x31, 16         \n\t"
        "esp.vst.l.64.ip q1, x31, 16         \n\t"
        "esp.vst.h.64.ip q1, x31, 16         \n\t"
        "esp.vst.l.64.ip q5, x31, 16         \n\t"
        "esp.vst.h.64.ip q5, x31, 16         \n\t"
        "esp.vst.l.64.ip q3, x31, 16         \n\t"
        "esp.vst.h.64.ip q3, x31, 16         \n\t"
        "esp.vst.l.64.ip q7, x31, 16         \n\t"
        "esp.vst.h.64.ip q7, x31, 16         \n\t"
        :
        : [b] "r"(bases), [off] "r"(off), [dst] "r"(dst)
        : "x29", "x30", "x31", "memory"
    );
}

/* As im2col_stage8x16 for 8 window bytes: 8-byte loads, so a 9-byte
 * row (3x3, in_ch 3) is read without passing its end. */
static inline __attribute__((always_inline))
void im2col_stage8x8(const int8_t *const *bases, int32_t off, int8_t *dst)
{
    asm volatile (
        "mv     x29, %[b]                    \n\t"
        "lw     x30, 0(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q0, x30, 0          \n\t"
        "lw     x30, 4(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q1, x30, 0          \n\t"
        "lw     x30, 8(x29)                  \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q2, x30, 0          \n\t"
        "lw     x30, 12(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q3, x30, 0          \n\t"
        "lw     x30, 16(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q4, x30, 0          \n\t"
        "lw     x30, 20(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q5, x30, 0          \n\t"
        "lw     x30, 24(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q6, x30, 0          \n\t"
        "lw     x30, 28(x29)                 \n\t"
        "add    x30, x30, %[off]             \n\t"
        "esp.vld.l.64.ip q7, x30, 0          \n\t"
        "mv     x31, %[dst]                  \n\t"
        "esp.vzip.8  q0, q1                  \n\t"
        "esp.vzip.8  q2, q3                  \n\t"
        "esp.vzip.8  q4, q5                  \n\t"
        "esp.vzip.8  q6, q7                  \n\t"
        "esp.vzip.16 q0, q2                  \n\t"
        "esp.vzip.16 q4, q6                  \n\t"
        "esp.vzip.32 q0, q4                  \n\t"
        "esp.vzip.32 q2, q6                  \n\t"
        /* Window bytes 0-1, 2-3, 4-5, 6-7 are in q0 q4 q2 q6. */
        "esp.vst.l.64.ip q0, x31, 16         \n\t"
        "esp.vst.h.64.ip q0, x31, 16         \n\t"
        "esp.vst.l.64.ip q4, x31, 16         \n\t"
        "esp.vst.h.64.ip q4, x31, 16         \n\t"
        "esp.vst.l.64.ip q2, x31, 16         \n\t"
        "esp.vst.h.64.ip q2, x31, 16         \n\t"
        "esp.vst.l.64.ip q6, x31, 16         \n\t"
        "esp.vst.h.64.ip q6, x31, 16         \n\t"
        :
        : [b] "r"(bases), [off] "r"(off), [dst] "r"(dst)
        : "x29", "x30", "x31", "memory"
    );
}

/* dst[i * 16 + p] = bases[p][off + i] for i < (n & ~7), p < 16: 16-byte
 * chunks, then one 8-byte chunk. Not inlined, so the asm clobbers stay
 * out of the callers' staging loops. Callers pass a copy of their pointer
 * array: an array whose address escapes stays in memory in their
 * byte-copy loops (about 10% on the in_ch 3 first layers). */
__attribute__((noinline))
static void im2col_stage_lanes16(const int8_t *const *bases, int32_t off,
                                 int32_t n, int8_t *dst)
{
    for (; n >= 16; n -= 16, off += 16, dst += 256) {
        im2col_stage8x16(bases, off, dst);
        im2col_stage8x16(bases + 8, off, dst + 8);
    }
    if (n >= 8) {
        im2col_stage8x8(bases, off, dst);
        im2col_stage8x8(bases + 8, off, dst + 8);
    }
}

/**
 * Im2col convolution: the whole filter window is one contiguous vector,
 * window_len = filter_wd * filter_ht * in_ch (e.g. 3 * 3 * 3 = 27), so
 * small in_ch wastes no MACs on channel padding.
 *
 * - Short window rows on 16 to 48 output channels and large maps run the
 *   oc16 kernel (esp_nn_conv_oc16_ok).
 * - Windows up to ESP_NN_IM2COL_BATCH_MAX_WINDOW run in 16-pixel QACC
 *   batches.
 * - Other pixels copy their window into im2col_buf and run a PIE dot
 *   product.
 *
 * Scratch layout:
 * - filter_sum: 4 bytes per output channel.
 * - im2col_buf: window_len bytes.
 * - Batch paths only: a 16-byte aligned work area. It holds the 16-pixel
 *   staging or the oc16 work, then the requant table
 *   (esp_nn_conv_im2col_tab_offset).
 */
__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t window_len = filter_wd * filter_ht * in_ch;
    const int8_t pad_val = (int8_t)(-input_offset);

    /* Unaligned vld in the staging: CFG is per core and not saved per task. */
    ESP_NN_PIE_ENABLE();

    /* Scratch: filter_sum[out_ch] + im2col_buf[window_len] */
    int32_t *filter_sum = (int32_t *)scratch;
    int8_t *im2col_buf = (int8_t *)scratch + out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset */
    for (int32_t oc = 0; oc < out_ch; oc++) {
        filter_sum[oc] = conv_filter_byte_sum(
                filter_data + oc * window_len, window_len) * input_offset;
    }

    const int32_t total_pixels = (int32_t)out_wd * out_ht;

    /* The work area: the 16-pixel staging or the oc16 work, then the
     * requant table (see esp_nn_conv_im2col_tab_offset). */
    int8_t *work = NULL;
    const int32_t *tab = NULL;
    if (window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW) {
        work = (int8_t *)(((uintptr_t)(im2col_buf + window_len) + 15)
                          & ~(uintptr_t)15);
        const bool oc16 = esp_nn_conv_oc16_ok(filter_wd, in_ch, out_ch,
                                              total_pixels);
        int32_t *tab_buf = (int32_t *)(work + esp_nn_conv_im2col_tab_offset(
                oc16, out_ch, window_len));
        if (oc16) {
            /* Short window rows: lanes are output channels. */
            tab = qacc16_requant_table_w(filter_sum, bias, quant_data->mult,
                    quant_data->shift, out_ch, window_len, out_offset,
                    tab_buf, ESP_NN_CONV_OC16_TAB_WORDS);
            if (tab) {
                esp_nn_conv_s8_oc16_pie(input_dims, input_data, filter_dims,
                                        filter_data, output_dims, out_data,
                                        conv_params, tab, work);
                return;
            }
        } else {
            tab = qacc16_requant_table(filter_sum, bias, quant_data->mult,
                    quant_data->shift, out_ch, window_len, out_offset,
                    tab_buf);
        }
    }

    /* QACC batch path: 16 consecutive output pixels (raster order, across
     * rows) per batch. Their windows are staged transposed, t[tap * 16 + p],
     * with pad_val for taps outside the input, so padded layers keep all 16
     * lanes busy and no pixel falls to the per-pixel path. Accumulation
     * order per output element is unchanged: bit-identical. */
    int32_t raster_end = 0;     /* exclusive, in pixels */
    if (window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW) {
        const int32_t row_size = filter_wd * in_ch;
        int8_t *t = work;
        /* A tail of fewer than 8 pixels costs less per pixel below. */
        raster_end = total_pixels;
        if ((total_pixels & 15) != 0 && (total_pixels & 15) < 8) {
            raster_end = total_pixels & ~15;
        }
        /* The per-pixel window buffer is free here: it holds the pad row. */
        int8_t *pad_row = im2col_buf;
        memset(pad_row, pad_val, in_ch);
        for (int32_t pix = 0; pix < raster_end; pix += 16) {
            const int32_t lanes = min(16, raster_end - pix);
            /* Window origin of each lane; lanes past `lanes` repeat pixel
             * pix - they are multiplied but never stored. */
            const int8_t *bases[16];
            int32_t org_y[16], org_x[16];
            bool interior = true;
            for (int p = 0; p < 16; p++) {
                const int32_t px = pix + ((p < lanes) ? p : 0);
                const int32_t oy = px / out_wd;
                const int32_t ox = px % out_wd;
                org_y[p] = oy * stride_ht - pad_ht;
                org_x[p] = ox * stride_wd - pad_wd;
                interior = interior && org_y[p] >= 0 && org_x[p] >= 0 &&
                        org_y[p] + filter_ht <= input_ht &&
                        org_x[p] + filter_wd <= input_wd;
                bases[p] = input_data +
                        ((int32_t)org_y[p] * input_wd + org_x[p]) * in_ch;
            }
            if (interior) {
                /* Tap-outer so each tap's 16 stores fill one 16-byte line. */
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int32_t row_off = (int32_t)fy * input_wd * in_ch;
                    int8_t *dst = t + (int32_t)fy * row_size * 16;
                    if (row_size >= 8) {
                        const int8_t *vb[16];
                        for (int p = 0; p < 16; p++) {
                            vb[p] = bases[p];
                        }
                        im2col_stage_lanes16(vb, row_off, row_size, dst);
                    }
                    const int32_t i0 = row_size & ~7;
                    dst += i0 * 16;
                    for (int32_t i = i0; i < row_size; i++, dst += 16) {
                        const int32_t off = row_off + i;
                        dst[0]  = bases[0][off];   dst[1]  = bases[1][off];
                        dst[2]  = bases[2][off];   dst[3]  = bases[3][off];
                        dst[4]  = bases[4][off];   dst[5]  = bases[5][off];
                        dst[6]  = bases[6][off];   dst[7]  = bases[7][off];
                        dst[8]  = bases[8][off];   dst[9]  = bases[9][off];
                        dst[10] = bases[10][off];  dst[11] = bases[11][off];
                        dst[12] = bases[12][off];  dst[13] = bases[13][off];
                        dst[14] = bases[14][off];  dst[15] = bases[15][off];
                    }
                }
            } else {
                /* Border batch: per filter tap, each lane reads its input
                 * pixel or the pad_val row. Only in-bounds bytes are read. */
                int8_t *dst = t;
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    for (int32_t fx = 0; fx < filter_wd; fx++) {
                        const int8_t *src[16];
                        for (int p = 0; p < 16; p++) {
                            const int32_t iy = org_y[p] + fy;
                            const int32_t ix = org_x[p] + fx;
                            src[p] = (iy >= 0 && iy < input_ht && ix >= 0 &&
                                      ix < input_wd)
                                    ? input_data + ((int32_t)iy * input_wd
                                                    + ix) * in_ch
                                    : pad_row;
                        }
                        if (in_ch >= 8) {
                            const int8_t *vb[16];
                            for (int p = 0; p < 16; p++) {
                                vb[p] = src[p];
                            }
                            im2col_stage_lanes16(vb, 0, in_ch, dst);
                        }
                        const int32_t c0 = in_ch & ~7;
                        dst += c0 * 16;
                        for (int32_t c = c0; c < in_ch; c++, dst += 16) {
                            dst[0]  = src[0][c];   dst[1]  = src[1][c];
                            dst[2]  = src[2][c];   dst[3]  = src[3][c];
                            dst[4]  = src[4][c];   dst[5]  = src[5][c];
                            dst[6]  = src[6][c];   dst[7]  = src[7][c];
                            dst[8]  = src[8][c];   dst[9]  = src[9][c];
                            dst[10] = src[10][c];  dst[11] = src[11][c];
                            dst[12] = src[12][c];  dst[13] = src[13][c];
                            dst[14] = src[14][c];  dst[15] = src[15][c];
                        }
                    }
                }
            }
            qacc16_run_channels(t, filter_data, window_len, 0, out_ch,
                                filter_sum, bias, quant_data->mult,
                                quant_data->shift,
                                out_data + (int32_t)pix * out_ch, out_ch,
                                out_offset, activation_min, activation_max,
                                lanes, tab);
        }
    }

    /* Per-pixel path: short tails and windows too long for the batch. */
    for (int32_t pixel = raster_end; pixel < total_pixels; pixel++) {
        {
            const int32_t out_y = pixel / out_wd;
            const int32_t out_x = pixel % out_wd;
            int8_t *out_ptr = out_data + (int32_t)pixel * out_ch;
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int32_t base_x = out_x * stride_wd - pad_wd;

            /* Copy input window into contiguous im2col buffer */
            int8_t *buf = im2col_buf;
            for (int32_t fy = 0; fy < filter_ht; fy++) {
                int32_t in_y = base_y + fy;
                for (int32_t fx = 0; fx < filter_wd; fx++) {
                    int32_t in_x = base_x + fx;
                    if (in_y >= 0 && in_y < input_ht && in_x >= 0 && in_x < input_wd) {
                        const int8_t *src = input_data + (in_y * input_wd + in_x) * in_ch;
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = src[c];
                        }
                    } else {
                        /* Padding pixel */
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = pad_val;
                        }
                    }
                }
            }

            /* Dot product against each output channel's filter */
            const int32_t *out_mult = quant_data->mult;
            const int32_t *out_shift = quant_data->shift;
            const int8_t *filter_ptr = filter_data;

            for (int32_t oc = 0; oc < out_ch; oc++) {
                int32_t conv_out = pie_dot_s8(im2col_buf, filter_ptr, window_len);
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
                filter_ptr += window_len;
            }
        }
    }
}

/**
 * Tiled convolution: process T output rows at a time.
 * Converts padded conv into a series of no-pad sub-problems by
 * copying/padding input tiles into the scratch buffer.
 *
 * This keeps the working set in L1D for large input tensors.
 * Reuses the existing esp_nn_conv_s8_padded PIE inner loop per tile.
 */

__attribute__ ((noinline))
static void esp_nn_conv_s8_tiled(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;

    /* Shared plan: channel padding for PIE (row_size must be >= 16) and the
     * tile height. The scratch getter runs the identical computation. */
    conv_tile_plan_t plan;
    conv_plan_tiles(input_wd, in_ch, filter_wd, filter_ht, out_ch, out_ht,
                    pad_wd, stride_ht, &plan);
    int new_ch = plan.eff_ch;
    int need_ch_pad = (plan.filt_aligned != 0);
    int padded_input_wd = input_wd + 2 * pad_wd;

    /* Scratch layout:
     * [0] filter_sum: out_ch * 4 bytes
     * [after filter_sum] aligned_filter (if ch padding): filter_wd * filter_ht * new_ch * out_ch
     * [after filter] tile_input_buf: variable per tile
     */
    int32_t *filter_sum = (int32_t *) scratch;
    int filter_sum_size = out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset (once for entire layer) */
    {
        const int32_t flen = filter_wd * filter_ht * in_ch;
        for (int32_t oc = 0; oc < out_ch; oc++) {
            filter_sum[oc] = conv_filter_byte_sum(
                    filter_data + oc * flen, flen) * input_offset;
        }
    }

    /* Channel-pad filter if needed (pad with 0s - doesn't affect filter_sum) */
    int8_t *aligned_filter = NULL;
    int aligned_filter_size = 0;
    if (need_ch_pad) {
        aligned_filter = (int8_t *)scratch + filter_sum_size;
        aligned_filter_size = filter_wd * filter_ht * new_ch * out_ch;
        memset(aligned_filter, 0, aligned_filter_size);
        const int8_t *src_f = filter_data;
        int8_t *dst_f = aligned_filter;
        for (int oc = 0; oc < out_ch; oc++) {
            for (int fh = 0; fh < filter_ht; fh++) {
                for (int fw = 0; fw < filter_wd; fw++) {
                    memcpy(dst_f, src_f, in_ch);
                    src_f += in_ch;
                    dst_f += new_ch;  /* zero-padded channels */
                }
            }
        }
    }

    /* Tile input buffer starts after filter_sum + aligned_filter */
    int8_t *tile_buf = (int8_t *)scratch + filter_sum_size + aligned_filter_size;

    /* Use effective channel count for tile buffer sizing */
    int eff_ch = need_ch_pad ? new_ch : in_ch;
    int tile_input_row_bytes = padded_input_wd * eff_ch;

    /* Tile height T (output rows per tile) comes from the shared plan */
    const int tile_T = plan.tile_T;
    (void)tile_input_row_bytes;

    /* Process tiles */
    const int8_t *use_filter = need_ch_pad ? aligned_filter : filter_data;
    data_dims_t eff_filter_dims = {filter_wd, filter_ht, eff_ch, 0};

    for (int32_t tile_y = 0; tile_y < out_ht; tile_y += tile_T) {
        int32_t actual_T = min(tile_T, out_ht - tile_y);

        /* Input rows feeding output rows [tile_y, tile_y + actual_T):
         * exactly (actual_T - 1) * stride + filter_ht of them, which is what
         * conv_plan_tiles() sized the staging buffer for. */
        int32_t in_row_start = tile_y * stride_ht - pad_ht;
        int32_t in_row_end = (tile_y + actual_T - 1) * stride_ht - pad_ht + filter_ht - 1;
        int32_t tile_input_ht = in_row_end - in_row_start + 1;

        /* Copy/pad input rows into tile buffer, with channel padding if needed */
        int8_t pad_val = (int8_t)(-input_offset);
        int8_t *dst = tile_buf;

        for (int32_t row = in_row_start; row <= in_row_end; row++) {
            if (row < 0 || row >= input_ht) {
                memset(dst, pad_val, padded_input_wd * eff_ch);
            } else {
                /* For each pixel in padded row */
                int8_t *row_dst = dst;
                /* Left padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
                /* Valid pixels - with optional channel padding */
                const int8_t *row_src = input_data + row * input_wd * in_ch;
                if (need_ch_pad) {
                    for (int px = 0; px < input_wd; px++) {
                        memcpy(row_dst, row_src, in_ch);
                        if (eff_ch > in_ch) {
                            memset(row_dst + in_ch, pad_val, eff_ch - in_ch);
                        }
                        row_src += in_ch;
                        row_dst += eff_ch;
                    }
                } else {
                    memcpy(row_dst, row_src, input_wd * in_ch);
                    row_dst += input_wd * in_ch;
                }
                /* Right padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
            }
            dst += padded_input_wd * eff_ch;
        }

        /* Sub-problem with pad=0, effective channels */
        data_dims_t tile_input_dims = {padded_input_wd, tile_input_ht, eff_ch, 0};
        data_dims_t tile_output_dims = {out_wd, actual_T, out_ch, 0};
        conv_params_t tile_conv_params = *conv_params;
        tile_conv_params.padding.width = 0;
        tile_conv_params.padding.height = 0;

        esp_nn_conv_s8_padded(&tile_input_dims, tile_buf,
                              &eff_filter_dims, use_filter, bias,
                              &tile_output_dims,
                              out_data + tile_y * out_wd * out_ch,
                              &tile_conv_params, quant_data,
                              filter_sum);
    }
}


void esp_nn_set_conv_scratch_buf_riscv_pie(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_borrow_riscv_pie(void *buf, void **scratch)
{
    *scratch = scratch_buffer;
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_restore_riscv_pie(void *scratch)
{
    scratch_buffer = (int16_t *) scratch;
}

typedef void (*conv_s8_pie_fn_t)(const data_dims_t *, const int8_t *,
                                 const data_dims_t *, const int8_t *,
                                 const int32_t *, const data_dims_t *,
                                 int8_t *, const conv_params_t *,
                                 const quant_data_t *, void *);

typedef struct {
    conv_s8_pie_fn_t fn;
    data_dims_t input_dims;
    const int8_t *input;
    const data_dims_t *filter_dims;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    conv_params_t conv_params;
    const quant_data_t *quant_data;
    void *scratch;
} conv_pie_rows_mt_job_t;

static void conv_pie_rows_mt_worker(void *p)
{
    const conv_pie_rows_mt_job_t *j = (const conv_pie_rows_mt_job_t *)p;
    j->fn(&j->input_dims, j->input, j->filter_dims, j->filter_data, j->bias,
          &j->output_dims, j->out_data, &j->conv_params, j->quant_data,
          j->scratch);
}

/* Output-row split for the pixel-independent conv paths. The top slice
 * keeps the top padding; the bottom slice enters the input at an interior
 * row with pad_ht = 0, so both slices evaluate exactly the taps the single
 * call would, on disjoint output rows (bit-identical results). Returns
 * false when the split is not applicable or the worker is unavailable. */
static bool conv_pie_rows_split(conv_s8_pie_fn_t fn,
                                const data_dims_t *input_dims,
                                const int8_t *input,
                                const data_dims_t *filter_dims,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                const data_dims_t *output_dims,
                                int8_t *out_data,
                                const conv_params_t *conv_params,
                                const quant_data_t *quant_data,
                                void *scratch)
{
    const uint16_t out_ht = output_dims->height;
    const uint16_t h0 = out_ht / 2;
    const int32_t in_row_off =
            (int32_t)h0 * conv_params->stride.height - conv_params->padding.height;
    if (out_ht < 4 || in_row_off < 0 || !esp_nn_dual_core_begin()) {
        return false;
    }
    const int scratch_size = esp_nn_conv_scratch_size_riscv_pie_impl(
            input_dims, filter_dims, output_dims, conv_params);
    void *wscr = esp_nn_dual_core_scratch(scratch_size + 16);
    if (wscr == NULL) {
        esp_nn_dual_core_end();
        return false;
    }
    conv_pie_rows_mt_job_t job = {
        .fn = fn, .input_dims = *input_dims, .input = input,
        .filter_dims = filter_dims, .filter_data = filter_data, .bias = bias,
        .output_dims = *output_dims, .out_data = out_data,
        .conv_params = *conv_params, .quant_data = quant_data, .scratch = wscr,
    };
    job.output_dims.height = h0;
    if (!esp_nn_dual_core_run(conv_pie_rows_mt_worker, &job)) {
        esp_nn_dual_core_end();
        return false;
    }
    {
        data_dims_t in1 = *input_dims;
        data_dims_t outd1 = *output_dims;
        conv_params_t params1 = *conv_params;
        in1.height = input_dims->height - in_row_off;
        outd1.height = out_ht - h0;
        params1.padding.height = 0;
        fn(&in1,
           input + in_row_off * input_dims->width * input_dims->channels,
           filter_dims, filter_data, bias, &outd1,
           out_data + (int32_t)h0 * output_dims->width * output_dims->channels,
           &params1, quant_data, scratch);
    }
    esp_nn_dual_core_wait();
    esp_nn_dual_core_end();
    return true;
}

typedef struct {
    data_dims_t input_dims;
    const int8_t *input;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    const conv_params_t *conv_params;
    const quant_data_t *quant_data;
    void *scratch;
} conv1x1_pie_mt_job_t;

static void conv1x1_pie_mt_worker(void *p)
{
    const conv1x1_pie_mt_job_t *j = (const conv1x1_pie_mt_job_t *)p;
    esp_nn_conv_s8_1x1(&j->input_dims, j->input, j->filter_data, j->bias,
                       &j->output_dims, j->out_data, j->conv_params,
                       j->quant_data, j->scratch);
}

/*
 * Whether the dual-core row split of the 1x1 conv pays off. A map of 16
 * pixels or more runs 16-pixel batches on one core. When the worker's half
 * is under 16 pixels, the worker does not batch. P4 rev v3.2, batch-16 FC:
 * - Up to 32 output channels, one core with the batch is 1.03x to 2.7x
 *   faster than two cores without it.
 * - With in_ch not a multiple of 16, one core with the batch is faster at
 *   every output width (1.27x or more).
 * - Otherwise two cores are about as fast or faster, up to 1.8x at
 *   in_ch 256 and 256 outputs.
 * The split still wins in two cases, P4 rev v3.2 with the worker on:
 * - 16 pixels with a filter of 64 KB or more take the filter-major path,
 *   which does not batch: 1.7x to 1.9x faster split.
 * - The caller's half (the taller one for an odd height) keeps every
 *   16-pixel batch, e.g. 6x5 or 4x7: 1.3x to 1.6x faster split.
 * Out of line, so the dispatcher's code layout stays as it was.
 */
__attribute__((noinline))
static bool conv_1x1_split_wins(int32_t in_ch, const data_dims_t *output_dims)
{
    const int32_t total = (int32_t)output_dims->width * output_dims->height;
    const int32_t worker = (int32_t)(output_dims->height / 2) *
                           output_dims->width;
    if (total < 16 || worker >= 16) {
        return true;
    }
    if (output_dims->channels > 32 && (in_ch & 15) == 0) {
        return true;
    }
    if ((int32_t)in_ch * output_dims->channels >= 64 * 1024 && total <= 16) {
        return true;  /* filter-major path: no batch to keep */
    }
    /* Split when the caller's half keeps every 16-pixel batch. */
    return (total - worker) / 16 == total / 16;
}

void esp_nn_conv_s8_riscv_pie(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    ESP_NN_PIE_ENABLE();

    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }

    /* The fast paths fill pad cells with (int8_t)-input_offset.
     * For input_offset -128 that value is 128, which does not fit int8.
     * Use the reference for it. */
    if (conv_params->in_offset < -127) {
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    /* Grouped conv (filter_ch < input_ch) must be caught before any fast
     * path: they all assume full-depth filters. Run each group through the
     * optimized path via repacked slices (recursing into this dispatcher
     * with matching channel counts), or fall back to the reference with a
     * dual-core row split. Lives here, not in a sub-kernel, because every
     * sub-kernel is only reached with matching channels. */
    if (input_dims->channels != filter_dims->channels) {
        data_dims_t in_g = *input_dims;
        data_dims_t out_g = *output_dims;
        const int32_t groups = filter_dims->channels
                ? input_dims->channels / filter_dims->channels : 0;
        if (groups > 1 && output_dims->channels % groups == 0) {
            in_g.channels = filter_dims->channels;
            out_g.channels = output_dims->channels / groups;
            const int inner = esp_nn_conv_scratch_size_riscv_pie_impl(
                    &in_g, filter_dims, &out_g, conv_params);
            const int total = esp_nn_conv_scratch_size_riscv_pie_impl(
                    input_dims, filter_dims, output_dims, conv_params);
            if (esp_nn_conv_s8_grouped_repack(
                    esp_nn_conv_s8_riscv_pie, input_dims, input,
                    filter_dims, filter_data, bias, output_dims, out_data,
                    conv_params, quant_data, scratch_buffer, total, inner)) {
                return;
            }
        }
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {
        /* Rows are independent for a stride-1 unpadded 1x1: split them
         * across both cores (bit-identical, disjoint outputs). */
        if (output_dims->height >= 4 &&
                conv_1x1_split_wins(input_dims->channels, output_dims) &&
                esp_nn_dual_core_begin()) {
            const int scratch_size = esp_nn_conv_scratch_size_riscv_pie_impl(
                    input_dims, filter_dims, output_dims, conv_params);
            void *wscr = esp_nn_dual_core_scratch(scratch_size + 16);
            const uint16_t h0 = output_dims->height / 2;
            if (wscr != NULL && h0 > 0) {
                conv1x1_pie_mt_job_t job = {
                    .input_dims = *input_dims, .input = input,
                    .filter_data = filter_data, .bias = bias,
                    .output_dims = *output_dims, .out_data = out_data,
                    .conv_params = conv_params, .quant_data = quant_data,
                    .scratch = wscr,
                };
                job.input_dims.height = h0;
                job.output_dims.height = h0;
                if (esp_nn_dual_core_run(conv1x1_pie_mt_worker, &job)) {
                    data_dims_t in1 = *input_dims;
                    data_dims_t outd1 = *output_dims;
                    in1.height = input_dims->height - h0;
                    outd1.height = output_dims->height - h0;
                    const int32_t off = (int32_t)h0 * input_dims->width;
                    esp_nn_conv_s8_1x1(&in1, input + off * input_dims->channels,
                                       filter_data, bias, &outd1,
                                       out_data + off * output_dims->channels,
                                       conv_params, quant_data, scratch_buffer);
                    esp_nn_dual_core_wait();
                    esp_nn_dual_core_end();
                    return;
                }
            }
            esp_nn_dual_core_end();
        }
        esp_nn_conv_s8_1x1(input_dims, input, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data,
                           scratch_buffer);
    } else if (pad_wd == 0 && pad_ht == 0 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels >= 128 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels
                       <= ESP_NN_IM2COL_BATCH_MAX_WINDOW &&
               output_dims->channels >= 16) {
        /* Dense no-pad conv whose window fits the 16-pixel QACC batch: the
         * per-element XACC path below pays a dot setup and a requant per
         * output element, while the batch stages 16 windows once and
         * amortizes both across them. Same accumulation order per output:
         * bit-identical. */
        esp_nn_conv_s8_im2col(input_dims, input, filter_dims, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data,
                              scratch_buffer);
    } else if (pad_wd == 0 && pad_ht == 0 &&
               filter_wd * input_dims->channels >= 16) {
        /* No-pad, channels large enough for PIE: use direct padded path */
        if (!conv_pie_rows_split(esp_nn_conv_s8_padded, input_dims, input,
                                 filter_dims, filter_data, bias, output_dims,
                                 out_data, conv_params, quant_data,
                                 scratch_buffer)) {
            esp_nn_conv_s8_padded(input_dims, input, filter_dims, filter_data, bias,
                                  output_dims, out_data, conv_params, quant_data,
                                  scratch_buffer);
        }
    } else if ((pad_wd != 0 || pad_ht != 0) &&
               filter_wd * input_dims->channels >= 16 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels *
                       output_dims->channels > 96 * 1024) {
        /* Padded conv whose filter exceeds the L2 cache: the pixel-outer
         * im2col loop re-streams the whole filter per pixel, and once it
         * no longer fits L2 that traffic comes from flash (yolo11n's
         * 147 KB 3x3 pad-1 head ran 6.4 c/MAC). Stage the padding once and
         * run the dense OC-panel kernel per tile. Filters that fit L2 stay
         * on im2col, whose contiguous window dot has the higher peak
         * throughput (32 B/iteration, double-buffered). */
        esp_nn_conv_s8_tiled(input_dims, input, filter_dims, filter_data, bias,
                             output_dims, out_data, conv_params, quant_data,
                             scratch_buffer);
    } else if (filter_wd * filter_ht * input_dims->channels >= 16) {
        /* Small in_ch but window_len >= 16: use im2col for zero-waste PIE.
         * Also handles padded cases naturally. */
        if (!conv_pie_rows_split(esp_nn_conv_s8_im2col, input_dims, input,
                                 filter_dims, filter_data, bias, output_dims,
                                 out_data, conv_params, quant_data,
                                 scratch_buffer)) {
            esp_nn_conv_s8_im2col(input_dims, input, filter_dims, filter_data, bias,
                                  output_dims, out_data, conv_params, quant_data,
                                  scratch_buffer);
        }
    } else if (pad_wd != 0 || pad_ht != 0) {
        /* Padded case with very small window: use tiled path */
        esp_nn_conv_s8_tiled(input_dims, input, filter_dims, filter_data, bias,
                             output_dims, out_data, conv_params, quant_data,
                             scratch_buffer);
    } else {
        /* Tiny output: fall back to generic opt */
        esp_nn_conv_s8_opt(input_dims, input, filter_dims, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data);
    }
}
