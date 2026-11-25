/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <esp_nn_defs.h>
#include <common_functions.h>
#include <esp_nn_multicore.h>
#include <stdlib.h>
#include <string.h>

#include "esp_nn_conv_dispatch_riscv_pie.h"
#include "esp_nn_depthwise_conv_mac_riscv_pie.h"

/* External fallback */
void esp_nn_depthwise_conv_s8_opt(const data_dims_t *input_dims,
                                   const int8_t *input_data,
                                   const data_dims_t *filter_dims,
                                   const int8_t *filter_data,
                                   const int32_t *bias,
                                   const data_dims_t *output_dims,
                                   int8_t *out_data,
                                   const dw_conv_params_t *conv_params,
                                   const quant_data_t *quant_data);

static int32_t *scratch_buffer = NULL;

void esp_nn_set_depthwise_conv_scratch_buf_riscv_pie(const void *buf)
{
    scratch_buffer = (int32_t *) buf;
}

/* Requantizes np (up to 32) positions x block_ch channels of acc (16 lanes per
 * position), clamps them and stores them at out_p[p] + ch_idx, or at
 * op + p * ostride + ch_idx when out_p is NULL. vrq is the table of
 * esp_nn_dw_vrq_riscv_pie, or NULL when its gate fails. */
static inline __attribute__((always_inline)) void dw_pie_store(int32_t *acc, int np, int8_t *const *out_p,
                                int8_t *op, int ostride, int ch_idx, int block_ch,
                                const int16_t *vrq, const quant_data_t *quant_data,
                                int32_t out_offset, const int32_t *act_min,
                                const int32_t *act_max)
{
    if (vrq && (block_ch == 16 || block_ch == 8)) {
        /* op is NULL when out_p is set: offset it as an integer. */
        esp_nn_dw_vrq_riscv_pie(acc, np, vrq + 7 * ch_idx,
                                (int8_t *) ((uintptr_t) op + ch_idx), ostride, out_p,
                                ch_idx, block_ch >> 3);
    } else if (vrq) {
        /* other partial blocks go through a 16-byte buffer after the
         * accumulators */
        int8_t *tmp = (int8_t *) (acc + 32 * 16);
        for (int p = 0; p < np; p++) {
            esp_nn_dw_vrq_riscv_pie(acc + 16 * p, 1, vrq + 7 * ch_idx, tmp, 0, NULL, 0, 2);
            memcpy((out_p ? out_p[p] : op + p * ostride) + ch_idx, tmp, block_ch);
        }
    } else {
        for (int p = 0; p < np; p++) {
            int8_t *o = (out_p ? out_p[p] : op + p * ostride) + ch_idx;
            for (int k = 0; k < block_ch; k++) {
                const int32_t c = ch_idx + k;
                int32_t r = acc[16 * p + k];
                r = esp_nn_requantize(r, quant_data->mult[c], quant_data->shift[c]);
                r += out_offset;
                o[k] = (int8_t) max(*act_min, min(r, *act_max));
            }
        }
    }
}

/* Channels [ch_from, ch_to) of one output position in scalar C. The
 * window starts at input (base_y, base_x) and is clipped to the input. */
static void dw_pie_scalar_pos(const data_dims_t *input_dims, const int8_t *input_data,
                              const data_dims_t *filter_dims, const int8_t *filter_data,
                              const int32_t *bias, const dw_conv_params_t *conv_params,
                              const quant_data_t *quant_data, int base_y, int base_x,
                              int8_t *op, int ch_from, int ch_to)
{
    const int input_wd = input_dims->width, channels = input_dims->channels;
    const int filter_wd = filter_dims->width;
    const int filter_y_start = max(0, -base_y);
    const int filter_x_start = max(0, -base_x);
    const int filter_y_end = min((int) filter_dims->height, input_dims->height - base_y);
    const int filter_x_end = min(filter_wd, input_wd - base_x);
    for (int ch = ch_from; ch < ch_to; ch++) {
        int32_t result = 0;
        for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int32_t idx_y = base_y + fy;
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                const int32_t idx_x = base_x + fx;
                result += (input_data[(idx_y * input_wd + idx_x) * channels + ch] +
                           conv_params->in_offset) *
                          filter_data[(fy * filter_wd + fx) * channels + ch];
            }
        }
        if (bias) result += bias[ch];
        result = esp_nn_requantize(result, quant_data->mult[ch], quant_data->shift[ch]);
        result += conv_params->out_offset;
        result = max(result, conv_params->activation.min);
        result = min(result, conv_params->activation.max);
        op[ch] = (int8_t) result;
    }
}

/* Whether the 16 channels from ch_idx include one that needs the scalar
 * requant. slow_blocks has one bit per 16-channel block, for the first 64
 * blocks. Past those, the vector requant is off for the whole layer. */
static inline bool dw_pie_block_slow(uint64_t slow_blocks, int ch_idx)
{
    const int lo = ch_idx >> 4, hi = (ch_idx + 15) >> 4;
    return slow_blocks && hi < 64 && (((slow_blocks >> lo) | (slow_blocks >> hi)) & 1);
}

/* A last block of 8-15 channels would load 16 bytes per tap and pixel,
 * past the end of the filter and of the input. Instead the block moves
 * back to end at c_end. Its first channels repeat the block before, with
 * the same results. It moves only when it stays in [c_beg, c_end) and
 * starts on an 8-channel group of the requant table. Returns false when
 * it cannot move: the caller then runs those channels in scalar C. */
static inline bool dw_pie_tail_block(int *ch_idx, int c_beg, int c_end)
{
    const int start = c_end - 16;
    if (start < c_beg || (start & 7)) {
        return false;
    }
    *ch_idx = start;
    return true;
}

/* Constants of the 3x3 edge strips of the non-slab path. */
typedef struct {
    const int8_t *input;
    int8_t *out;
    const int8_t *filter;
    const int32_t *comb;
    const int16_t *vrq;
    int32_t *acc;
    int8_t *slab;
    const int8_t *pad;
    int in_wd, in_ht, ch, out_wd, s_wd, s_ht, p_wd, p_ht;
    int c_beg, c_end;   /* the channels this call runs */
} dw_edge_t;

/* Output rows [oy0, oy0 + rh) and columns [ox0, ox0 + rw) of a 3x3 layer
 * whose channels all run in fused 16-channel blocks. Per block, the input
 * of the rectangle is copied to a slab padded with -input_offset, then one
 * fused call runs all its positions. */
static void dw_pie_edge_rect(const dw_edge_t *e, int oy0, int rh, int ox0, int rw)
{
    const int slab_w = (rw - 1) * e->s_wd + 3;
    const int slab_rows = (rh - 1) * e->s_ht + 3;
    int8_t *fslab = e->slab + slab_rows * slab_w * 16;
    /* slab columns: nl pad, nin input from column xs, nr pad */
    const int x0 = ox0 * e->s_wd - e->p_wd;
    const int nl = min(slab_w, max(0, -x0));
    const int xs = max(0, x0);
    const int nin = max(0, min(slab_w - nl, e->in_wd - xs));
    const int nr = slab_w - nl - nin;
    const int y0 = oy0 * e->s_ht - e->p_ht;
    dw_win_t sw = {
        .row_len = rw, .step = e->s_wd * 16,
        .wrap = (e->s_ht * slab_w - rw * e->s_wd) * 16,
        .fp = fslab, .acc = e->acc, .ch = 16, .rows = 3, .cols = 3,
        .ip_row = (slab_w - 3) * 16,
        .ostep = e->ch, .owrap = (e->out_wd - rw) * e->ch,
    };
    for (int ch_idx = e->c_beg; ch_idx < e->c_end; ch_idx += 16) {
        esp_nn_dw_slab_row_riscv_pie(fslab, e->filter + ch_idx, e->ch, 0, 9, 0, e->pad);
        for (int i = 0; i < slab_rows; i++) {
            const int y = y0 + i;
            int8_t *d = e->slab + i * slab_w * 16;
            if (y < 0 || y >= e->in_ht || nin == 0) {
                esp_nn_dw_slab_row_riscv_pie(d, NULL, 0, slab_w, 0, 0, e->pad);
            } else {
                esp_nn_dw_slab_row_riscv_pie(d, e->input + (y * e->in_wd + xs) * e->ch + ch_idx,
                                             e->ch, nl, nin, nr, e->pad);
            }
        }
        sw.ip = e->slab;
        sw.oxl = rw;
        sw.np = rw * rh;
        sw.comb = e->comb + ch_idx;
        sw.op = e->out + (oy0 * e->out_wd + ox0) * e->ch + ch_idx;
        sw.vrq = e->vrq + 7 * ch_idx;
        esp_nn_dw_win_vrq_riscv_pie(&sw);
    }
}

/* PIE ch_mult == 1 path, channels >= 8, 16 channels per QACC pass.
 * - QACC starts at comb[ch] = filter_sum[ch] * input_offset + bias[ch] and
 *   accumulates in * filter over the full window. Taps outside the input
 *   read -input_offset, so they add nothing.
 * - Output positions run in batches of up to 16, or 32 on the slab path.
 *   Inner positions go first, then edge positions. Per 16-channel block,
 *   the window sums go to a scratch buffer.
 *   esp_nn_dw_vrq_riscv_pie requantizes, clamps and stores them on PIE.
 * - Fewer than 8 channels left over run in scalar C. */
__attribute__ ((noinline))
static void depthwise_conv_s8_ch1_pie_impl(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data,
                                       int32_t *layer_scratch, int slab_bytes,
                                       int edge_bytes, int c_beg, int c_end)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Per-channel constants: comb = filter_sum * input_offset + bias, and
     * the int16 table of esp_nn_dw_vrq_riscv_pie (used when vrq_ok). acc is at
     * most taps * 128 * 128 in magnitude; the PIE requant needs
     * |acc + comb| < 2^26, mult < 2^31 - 2^15 (so mult splits into int16
     * halves), shift in [-20, -1] and (1 + 2 * out_offset) << rs to split
     * into two int16 factors, and an activation range inside int8. */
    const int ch_pad = (channels + 15) & ~15;
    int32_t *comb_v = (int32_t *) (((uintptr_t) layer_scratch + 15) & ~(uintptr_t) 15);
    /* Accumulators of up to 32 positions x 16 channels, position-major,
     * then a 16-byte buffer and the table. */
    int32_t *acc = comb_v + ch_pad;
    int16_t *vrq_t = (int16_t *) (acc + 32 * 16 + 4);
    const int64_t acc_bound = (int64_t) filter_ht * filter_wd * 128 * 128;
    bool vrq_ok = activation_min >= -128 && activation_max <= 127 &&
                  activation_min <= activation_max;
    /* Bit b set: 16-channel block b has a channel outside the PIE requant's
     * range, so that block alone takes the scalar requant. With more than 64
     * blocks, such a channel turns the PIE requant off for the layer. */
    uint64_t slow_blocks = 0;
    /* filter sums to comb_v: full 16-channel blocks on PIE, the rest in C */
    const int taps = filter_ht * filter_wd;
    int ch_sum = c_beg;
    if (input_offset != 0) {
        for (; ch_sum + 16 <= c_end; ch_sum += 16) {
            esp_nn_dw_tap_sum_riscv_pie(filter_data + ch_sum, channels, taps, comb_v + ch_sum);
        }
    }
    for (int ch = ch_sum; ch < c_end; ch++) {
        int32_t s = 0;
        for (int t = 0; input_offset != 0 && t < taps; t++) {
            s += filter_data[t * channels + ch];
        }
        comb_v[ch] = s;
    }
    /* |comb| must stay below comb_lim (0: no channel fits) */
    const uint32_t comb_lim = acc_bound >= (1 << 26) ? 0 : (uint32_t) ((1 << 26) - acc_bound);
    const int32_t *mult = quant_data->mult, *shift = quant_data->shift;
    /* table entries of [c_beg, c_end), and of the padding after the last
     * channel */
    const int t_end = c_end == channels ? ch_pad : c_end;
    for (int ch = c_beg; ch < t_end; ch++) {
        int32_t comb = 0, ml = 0, mh = 0, ra = 0, rb = 0, rs = 0;
        if (ch < c_end) {
            comb = comb_v[ch] * input_offset + (bias ? bias[ch] : 0);
            const int32_t m = mult[ch];
            rs = -shift[ch];
            const uint32_t abs_comb = comb < 0 ? 0u - (uint32_t) comb : (uint32_t) comb;
            /* m >= 0 and m < 0x7fff8000 together, as unsigned */
            bool ch_ok = (uint32_t) m < 0x7fff8000u && (uint32_t) (rs - 1) < 20u &&
                         abs_comb < comb_lim;
            if (!ch_ok) {
                rs = 1;
            }
            ml = (int16_t) m;
            mh = (int32_t) (((uint32_t) m - (uint32_t) ml) >> 16);
            const int b = min(rs, 14);
            ra = (1 + 2 * out_offset) * (1 << (rs - b));
            rb = 1 << b;
            if ((uint32_t) (ra + 32768) > 65535u) {
                ch_ok = false;
            }
            if (!ch_ok) {
                /* A moved last block (dw_pie_tail_block) can span blocks
                 * 63 and 64. Block 64 has no bit, so dw_pie_block_slow
                 * does not see block 63 for it. With channels past block
                 * 63, a channel of block 63 also turns the PIE requant off
                 * for the layer. */
                if ((ch >> 4) < 63 || ((ch >> 4) == 63 && c_end <= 64 * 16)) {
                    slow_blocks |= 1ULL << (ch >> 4);
                } else {
                    vrq_ok = false;
                }
            }
        }
        comb_v[ch] = comb;
        int16_t *t = vrq_t + 7 * (ch & ~7) + (ch & 7);
        t[0] = ml; t[8] = mh; t[16] = ra; t[24] = rb; t[32] = rs + 1;
        t[40] = activation_min; t[48] = activation_max;
    }
    const int16_t *vrq = vrq_ok ? vrq_t : NULL;
    /* 3x3 full 16-channel blocks with the PIE requant run window, requant
     * and store in one loop over the positions. */
    const bool fuse = filter_wd == 3 && filter_ht == 3 && vrq;

    /* Stand-in input for taps outside the input. The dispatcher keeps
     * -input_offset within int8. */
    int8_t pad_vec[16] __attribute__((aligned(16)));
    memset(pad_vec, (int8_t) -input_offset, sizeof(pad_vec));

    if (out_ht == 0 || out_wd == 0) {
        return;
    }
    if (slab_bytes) {
        /* Per 16-channel block, copy the input into a slab padded with
         * -input_offset and the filter to 16 bytes per tap. Every window
         * then lies inside the slab, so all positions take the
         * inner-window MAC with no edge cases. */
        const int slab_w = (out_wd - 1) * stride_wd + filter_wd;
        /* A dual-core slice is smaller than the full call, so its own band
         * may be longer. slab_bytes, sized for the full call, bounds it. */
        const int slab_rows = (slab_bytes / 16 - filter_ht * filter_wd) / slab_w;
        const int band = min(dw_pie_slab_band(input_dims, filter_dims, output_dims, conv_params),
                             (slab_rows - filter_ht) / stride_ht + 1);
        int8_t *slab = (int8_t *) vrq_t + 14 * ch_pad;
        int8_t *fslab = slab + ((band - 1) * stride_ht + filter_ht) * slab_w * 16;
        /* slab columns: nl pad, nin input, nr pad */
        const int nl = min(slab_w, pad_wd);
        const int nin = max(0, min(slab_w - pad_wd, (int) input_wd));
        const int nr = slab_w - nl - nin;
        /* slab steps to the next output column and from a row end to the
         * next row */
        dw_win_t sw = {
            .row_len = out_wd, .step = stride_wd * 16,
            .wrap = (stride_ht * slab_w - out_wd * stride_wd) * 16,
            .fp = fslab, .acc = acc, .ch = 16, .rows = filter_ht,
            .cols = filter_wd, .ip_row = (slab_w - filter_wd) * 16,
        };
        /* channels [0, blk_end) run in 16-channel blocks, the last one
         * partial if 8-15 channels remain */
        const int blk_end = (c_end & 15) >= 8 ? c_end : c_end & ~15;
        for (int oy0 = 0; oy0 < out_ht; oy0 += band) {
            const int band_ht = min(band, out_ht - oy0);
            const int y0 = oy0 * stride_ht - pad_ht;
            for (int ch_idx = c_beg; ch_idx < blk_end; ch_idx += 16) {
                const int block_ch = min(16, c_end - ch_idx);
                /* A 16-byte load of the last tap of a partial block would read
                 * past the filter: copy that tap in C. */
                const int vtaps = block_ch < 16 ? taps - 1 : taps;
                esp_nn_dw_slab_row_riscv_pie(fslab, filter_data + ch_idx, channels, 0, vtaps, 0, pad_vec);
                if (vtaps < taps) {
                    memcpy(fslab + 16 * vtaps, filter_data + vtaps * channels + ch_idx, block_ch);
                }
                sw.comb = comb_v + ch_idx;
                for (int i = 0; i < (band_ht - 1) * stride_ht + filter_ht; i++) {
                    const int y = y0 + i;
                    int8_t *d = slab + i * slab_w * 16;
                    if (y < 0 || y >= input_ht || nin == 0) {
                        esp_nn_dw_slab_row_riscv_pie(d, NULL, 0, slab_w, 0, 0, pad_vec);
                        continue;
                    }
                    const int8_t *src = input_data + y * input_wd * channels + ch_idx;
                    if (block_ch < 16 && y == input_ht - 1 && nin == input_wd) {
                        /* A 16-byte load of the last input pixel would read
                         * past the input: copy that pixel in C. */
                        esp_nn_dw_slab_row_riscv_pie(d, src, channels, nl, nin - 1, 0, pad_vec);
                        memcpy(d + (nl + nin - 1) * 16, src + (nin - 1) * channels, block_ch);
                        esp_nn_dw_slab_row_riscv_pie(d + (nl + nin) * 16, NULL, 0, 0, 0, nr, pad_vec);
                    } else {
                        esp_nn_dw_slab_row_riscv_pie(d, src, channels, nl, nin, nr, pad_vec);
                    }
                }
                /* Positions of the band in batches of 32: one window call,
                 * which steps through the positions itself, then one
                 * requant pass loads each channel's constants once for
                 * the 32. */
                const int16_t *bvrq = dw_pie_block_slow(slow_blocks, ch_idx) ? NULL : vrq;
                sw.ip = slab;
                sw.oxl = out_wd;
                int8_t *op = out_data + oy0 * out_wd * channels;
                if (fuse && bvrq && block_ch == 16) {
                    /* the whole band in one call, no row gap in the output */
                    sw.np = band_ht * out_wd;
                    sw.op = op + ch_idx;
                    sw.ostep = channels;
                    sw.owrap = 0;
                    sw.vrq = bvrq + 7 * ch_idx;
                    esp_nn_dw_win_vrq_riscv_pie(&sw);
                    continue;
                }
                int left = band_ht * out_wd;
                while (left > 0) {
                    const int np = min(32, left);
                    left -= np;
                    sw.np = np;
                    esp_nn_dw_win_riscv_pie(&sw);
                    dw_pie_store(acc, np, NULL, op, channels, ch_idx, block_ch, bvrq,
                                 quant_data, out_offset, &activation_min, &activation_max);
                    op += np * channels;
                }
            }
        }
        const int ch_idx = blk_end;
        for (int oy = 0; oy < out_ht && ch_idx < c_end; oy++) {
            for (int ox = 0; ox < out_wd; ox++) {
                dw_pie_scalar_pos(input_dims, input_data, filter_dims, filter_data, bias,
                                  conv_params, quant_data, oy * stride_ht - pad_ht,
                                  ox * stride_wd - pad_wd,
                                  out_data + (oy * out_wd + ox) * channels, ch_idx, c_end);
            }
        }
        return;
    }
    /* Output rows [oy_lo, oy_hi) and columns [ox_lo, ox_hi) have the
     * full window inside the input. Pass 0 walks those inner positions,
     * pass 1 the rest, so inner positions share batches and run through
     * one esp_nn_dw_win_riscv_pie call. */
    const int oy_lo = dw_pie_inner_lo(pad_ht, stride_ht, out_ht);
    const int ox_lo = dw_pie_inner_lo(pad_wd, stride_wd, out_wd);
    const int oy_hi = dw_pie_inner_hi(input_ht, pad_ht, stride_ht, filter_ht, out_ht, oy_lo);
    const int ox_hi = dw_pie_inner_hi(input_wd, pad_wd, stride_wd, filter_wd, out_wd, ox_lo);
    int pass = 0, oy = oy_lo, ox = ox_lo;
    if (oy_lo == oy_hi || ox_lo == ox_hi) {
        pass = 1;
        oy = ox = 0;
    }
    /* Inner positions step along the inner columns [ox_lo, ox_hi). */
    dw_win_t win = {
        .row_len = ox_hi - ox_lo, .step = stride_wd * channels,
        .wrap = (stride_ht * input_wd - (ox_hi - ox_lo) * stride_wd) * channels,
        .ch = channels, .rows = filter_ht, .cols = filter_wd,
        .ip_row = (input_wd - filter_wd) * channels,
    };

    if (fuse && pass == 0) {
        /* Inner positions in chunks of up to 32 in raster order. Each chunk
         * runs every channel block, so its input stays in the cache. */
        const int row_len = ox_hi - ox_lo;
        win.ostep = channels;
        win.owrap = (out_wd - row_len) * channels;
        int left = (oy_hi - oy_lo) * row_len;
        int oy_c = oy_lo, ox_c = ox_lo;
        while (left > 0) {
            const int np = min(32, left);
            const int8_t *ip0 = input_data +
                    ((oy_c * stride_ht - pad_ht) * input_wd + ox_c * stride_wd - pad_wd) * channels;
            int8_t *op0 = out_data + (oy_c * out_wd + ox_c) * channels;
            /* positions of the chunk, for blocks that do not take the
             * fused loop */
            int8_t *c_out[32];
            int c_y[32], c_x[32];
            bool have_pos = false;
            win.np = np;
            int ch_idx = c_beg;
            while (c_end - ch_idx >= 8) {
                if (c_end - ch_idx < 16 && !dw_pie_tail_block(&ch_idx, c_beg, c_end)) {
                    break;
                }
                const int block_ch = 16;
                const bool slow = dw_pie_block_slow(slow_blocks, ch_idx);
                win.ip = ip0 + ch_idx;
                win.oxl = ox_hi - ox_c;
                win.fp = filter_data + ch_idx;
                win.comb = comb_v + ch_idx;
                win.acc = acc;
                if (block_ch == 16 && !slow) {
                    win.op = op0 + ch_idx;
                    win.vrq = vrq + 7 * ch_idx;
                    esp_nn_dw_win_vrq_riscv_pie(&win);
                } else {
                    if (!have_pos) {
                        for (int p = 0, y = oy_c, x = ox_c; p < np; p++) {
                            c_out[p] = out_data + (y * out_wd + x) * channels;
                            c_y[p] = y * stride_ht - pad_ht;
                            c_x[p] = x * stride_wd - pad_wd;
                            if (++x == ox_hi) {
                                x = ox_lo;
                                y++;
                            }
                        }
                        have_pos = true;
                    }
                    esp_nn_dw_win_riscv_pie(&win);
                    dw_pie_store(acc, np, c_out, NULL, 0, ch_idx, block_ch, slow ? NULL : vrq,
                                 quant_data, out_offset, &activation_min, &activation_max);
                }
                ch_idx += block_ch;
            }
            if (ch_idx < c_end) {
                if (!have_pos) {
                    for (int p = 0, y = oy_c, x = ox_c; p < np; p++) {
                        c_out[p] = out_data + (y * out_wd + x) * channels;
                        c_y[p] = y * stride_ht - pad_ht;
                        c_x[p] = x * stride_wd - pad_wd;
                        if (++x == ox_hi) {
                            x = ox_lo;
                            y++;
                        }
                    }
                }
                for (int p = 0; p < np; p++) {
                    dw_pie_scalar_pos(input_dims, input_data, filter_dims, filter_data, bias,
                                      conv_params, quant_data, c_y[p], c_x[p], c_out[p], ch_idx, c_end);
                }
            }
            left -= np;
            ox_c += np;
            while (ox_c >= ox_hi) {
                ox_c -= row_len;
                oy_c++;
            }
        }
        pass = 1;
        oy = ox = 0;
    }

    if (fuse && edge_bytes && !slow_blocks && (c_end & 15) == 0) {
        /* Edge positions from padded slabs of their input: the top and
         * bottom rows, then the left and right columns of the inner rows
         * in chunks that fit edge_bytes. */
        const dw_edge_t e = {
            .input = input_data, .out = out_data, .filter = filter_data,
            .comb = comb_v, .vrq = vrq, .acc = acc,
            .slab = (int8_t *) vrq_t + 14 * ch_pad, .pad = pad_vec,
            .in_wd = input_wd, .in_ht = input_ht, .ch = channels, .out_wd = out_wd,
            .s_wd = stride_wd, .s_ht = stride_ht, .p_wd = pad_wd, .p_ht = pad_ht,
            .c_beg = c_beg, .c_end = c_end,
        };
        /* top and bottom strips in chunks of columns that fit edge_bytes */
        const int strip_y[2] = {0, oy_hi}, strip_h[2] = {oy_lo, out_ht - oy_hi};
        for (int k = 0; k < 2; k++) {
            if (strip_h[k] == 0) {
                continue;
            }
            const int sl_rows = (strip_h[k] - 1) * stride_ht + 3;
            const int cols = ((edge_bytes / 16 - 9) / sl_rows - 3) / stride_wd + 1;
            for (int c = 0; c < out_wd; c += cols) {
                dw_pie_edge_rect(&e, strip_y[k], strip_h[k], c, min(cols, out_wd - c));
            }
        }
        const int side_x[2] = {0, ox_hi}, side_w[2] = {ox_lo, out_wd - ox_hi};
        for (int k = 0; k < 2; k++) {
            if (side_w[k] == 0) {
                continue;
            }
            const int sw_cols = (side_w[k] - 1) * stride_wd + 3;
            const int rows = ((edge_bytes / 16 - 9) / sw_cols - 3) / stride_ht + 1;
            for (int r = oy_lo; r < oy_hi; r += rows) {
                dw_pie_edge_rect(&e, r, min(rows, oy_hi - r), side_x[k], side_w[k]);
            }
        }
        return;
    }

    /* Output positions run in batches of 16, so the requant loads each
     * channel's constants once per 16 positions, and one window call
     * covers a batch of inner positions. */
    while (pass < 2) {
        int np = 0;
        int pos_y[16], pos_x[16];
        int8_t *out_p[16];
        /* all positions of the batch have the full window inside the input */
        bool all_inner = true;
        /* input and positions left in the row of the first position */
        const int8_t *ip0 = NULL;
        int oxl0 = 0;
        while (np < 16) {
            if (pass == 1) {
                /* skip the inner columns of inner rows, and wrap rows */
                if (ox == ox_lo && oy >= oy_lo && oy < oy_hi) {
                    ox = ox_hi;
                }
                if (ox == out_wd) {
                    ox = 0;
                    if (++oy == out_ht) {
                        pass = 2;
                        break;
                    }
                    continue;
                }
            }
            pos_y[np] = oy * stride_ht - pad_ht;
            pos_x[np] = ox * stride_wd - pad_wd;
            if (np == 0) {
                ip0 = input_data + (pos_y[0] * input_wd + pos_x[0]) * channels;
                oxl0 = ox_hi - ox;
            }
            out_p[np] = out_data + (oy * out_wd + ox) * channels;
            all_inner &= pass == 0;
            np++;
            if (++ox == ox_hi && pass == 0) {
                ox = ox_lo;
                if (++oy == oy_hi) {
                    pass = 1;
                    oy = ox = 0;
                    break;
                }
            }
        }
        if (np == 0) {
            break;
        }
        win.np = np;
        /* window clipped to the input: rows [ys, ye), columns [xs, xe) */
        int16_t ys[16], ye[16], xs[16], xe[16];
        /* edge windows: input at (ys, xs) and filter offset of row ys */
        const int8_t *e_ip[16];
        int32_t e_fr[16];
        for (int p = 0; p < np && !all_inner; p++) {
            ys[p] = max(0, -pos_y[p]);
            xs[p] = max(0, -pos_x[p]);
            ye[p] = max(ys[p], min(filter_ht, input_ht - pos_y[p]));
            xe[p] = max(xs[p], min(filter_wd, input_wd - pos_x[p]));
            e_ip[p] = input_data + ((pos_y[p] + ys[p]) * input_wd + pos_x[p] + xs[p]) * channels;
            e_fr[p] = ys[p] * filter_wd * channels;
        }

        int ch_idx = c_beg;
        /* 16-channel blocks. When 8-15 channels remain, a last block that
         * ends at c_end (dw_pie_tail_block). */
        while (c_end - ch_idx >= 8) {
            if (c_end - ch_idx < 16 && !dw_pie_tail_block(&ch_idx, c_beg, c_end)) {
                break;
            }
            const int block_ch = 16;
            if (all_inner) {
                win.ip = ip0 + ch_idx;
                win.oxl = oxl0;
                win.fp = filter_data + ch_idx;
                win.comb = comb_v + ch_idx;
                win.acc = acc;
                esp_nn_dw_win_riscv_pie(&win);
            }
            for (int p = 0; p < np && !all_inner; p++) {
                const int32_t ch = channels;
                const int8_t *fp = filter_data + ch_idx;
                const int y0 = ys[p], y1 = ye[p], x0 = xs[p], x1 = xe[p];
                const int nr = y1 - y0, nc = x1 - x0;
                /* QACC starts at comb, so the requant does not add it */
                ESP_NN_QACC_LOAD_S32(comb_v + ch_idx);
                if (nr == 0 || nc == 0) {
                    /* window fully outside the input */
                    esp_nn_dw_mac_rows_riscv_pie(pad_vec, fp, 0, ch, filter_ht, filter_wd, 0, 0);
                } else {
                    const int8_t *ip = e_ip[p] + ch_idx;
                    const int8_t *fr = fp + e_fr[p];
                    /* Taps outside the input read -input_offset from pad_vec.
                     * They add (-in_offset + in_offset) * filter = 0, as
                     * QACC starts at the full-window comb. Up to 5
                     * rectangles; empty ones are skipped, as a call costs
                     * more than a few taps. */
                    if (y0) {
                        esp_nn_dw_mac_rows_riscv_pie(pad_vec, fp, 0, ch, y0, filter_wd, 0, 0);
                    }
                    if (x0) {
                        esp_nn_dw_mac_rows_riscv_pie(pad_vec, fr, 0, ch, nr, x0, 0,
                                              (filter_wd - x0) * ch);
                    }
                    esp_nn_dw_mac_rows_riscv_pie(ip, fr + x0 * ch, ch, ch, nr, nc,
                                          (input_wd - nc) * ch,
                                          (filter_wd - nc) * ch);
                    if (x1 < filter_wd) {
                        esp_nn_dw_mac_rows_riscv_pie(pad_vec, fr + x1 * ch, 0, ch, nr,
                                              filter_wd - x1, 0, x1 * ch);
                    }
                    if (y1 < filter_ht) {
                        esp_nn_dw_mac_rows_riscv_pie(pad_vec, fp + y1 * filter_wd * ch, 0, ch,
                                              filter_ht - y1, filter_wd, 0, 0);
                    }
                }
                ESP_NN_QACC_EXTRACT_S32(acc + 16 * p);
            }

            dw_pie_store(acc, np, out_p, NULL, 0, ch_idx, block_ch,
                         dw_pie_block_slow(slow_blocks, ch_idx) ? NULL : vrq,
                         quant_data, out_offset, &activation_min, &activation_max);
            ch_idx += block_ch;
        }

        /* Remaining channels < 8: scalar */
        for (int p = 0; p < np && ch_idx < c_end; p++) {
            dw_pie_scalar_pos(input_dims, input_data, filter_dims, filter_data, bias,
                              conv_params, quant_data, pos_y[p], pos_x[p], out_p[p], ch_idx, c_end);
        }
    }
}

/* Output rows below which the dual-core split is by channels */
#ifndef DW_PIE_CH_SPLIT_ROWS
#define DW_PIE_CH_SPLIT_ROWS 8
#endif

typedef struct {
    data_dims_t input_dims;
    const int8_t *input_data;
    const data_dims_t *filter_dims;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    dw_conv_params_t conv_params;
    const quant_data_t *quant_data;
    int slab_bytes;
    int edge_bytes;
    int c_beg, c_end;
} dw_pie_mt_job_t;

static void dw_pie_mt_worker(void *p)
{
    const dw_pie_mt_job_t *j = (const dw_pie_mt_job_t *)p;
    /* CFG is per core: the kernel's unaligned vector store needs it here. */
    ESP_NN_PIE_ENABLE();
    /* Worker uses its own per-layer constants buffer: no shared-scratch race. */
    int32_t *wscr = (int32_t *)esp_nn_dual_core_scratch(
            dw_pie_ch1_scratch_size(j->input_dims.channels) + j->slab_bytes + j->edge_bytes);
    depthwise_conv_s8_ch1_pie_impl(&j->input_dims, j->input_data, j->filter_dims,
                                   j->filter_data, j->bias, &j->output_dims,
                                   j->out_data, &j->conv_params, j->quant_data,
                                   wscr, j->slab_bytes, j->edge_bytes, j->c_beg, j->c_end);
}

/* Output-row split of the ch_mult==1 PIE path: taps are clipped against
 * input_ht inside the kernel, so the bottom slice can enter the input at an
 * interior row with pad_ht = 0 and both slices see exactly the taps of the
 * single call, on disjoint output rows (bit-identical). */
static bool dw_pie_rows_split(const data_dims_t *input_dims,
                              const int8_t *input_data,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const dw_conv_params_t *conv_params,
                              const quant_data_t *quant_data, int slab_bytes,
                              int edge_bytes)
{
    const uint16_t out_ht = output_dims->height;
    const uint16_t h0 = out_ht / 2;
    const int32_t in_row_off =
            (int32_t)h0 * conv_params->stride.height - conv_params->padding.height;
    if (out_ht < 4 || in_row_off < 0 || !esp_nn_dual_core_begin()) {
        return false;
    }
    /* Both slices use the slab decision and size of the full call. A
     * slice's slab is never larger than the full one. */
    if (esp_nn_dual_core_scratch(dw_pie_ch1_scratch_size(input_dims->channels) +
                                 slab_bytes + edge_bytes) == NULL) {
        esp_nn_dual_core_end();
        return false;
    }
    dw_pie_mt_job_t job = {
        .input_dims = *input_dims, .input_data = input_data,
        .filter_dims = filter_dims, .filter_data = filter_data, .bias = bias,
        .output_dims = *output_dims, .out_data = out_data,
        .conv_params = *conv_params, .quant_data = quant_data,
        .slab_bytes = slab_bytes, .edge_bytes = edge_bytes,
        .c_beg = 0, .c_end = input_dims->channels,
    };
    job.output_dims.height = h0;
    if (!esp_nn_dual_core_run(dw_pie_mt_worker, &job)) {
        esp_nn_dual_core_end();
        return false;
    }
    {
        data_dims_t in1 = *input_dims;
        data_dims_t outd1 = *output_dims;
        dw_conv_params_t params1 = *conv_params;
        in1.height = input_dims->height - in_row_off;
        outd1.height = out_ht - h0;
        params1.padding.height = 0;
        depthwise_conv_s8_ch1_pie_impl(
                &in1,
                input_data + in_row_off * input_dims->width * input_dims->channels,
                filter_dims, filter_data, bias, &outd1,
                out_data + (int32_t)h0 * output_dims->width * output_dims->channels,
                &params1, quant_data, scratch_buffer, slab_bytes, edge_bytes,
                0, input_dims->channels);
    }
    esp_nn_dual_core_wait();
    esp_nn_dual_core_end();
    return true;
}

/* Output-channel split of the ch_mult == 1 PIE path for maps of under 8
 * rows, where a row split is uneven (7 rows: 3 and 4). With a cold cache it
 * is 6-11% faster there than the row split; at 14 rows the row split is as
 * fast or faster. Both cores run every
 * position, the worker channels [0, cs) and this core [cs, channels), cs a
 * multiple of 16. Each core builds the per-channel table of its own
 * channels only. Disjoint outputs, same arithmetic: bit-identical. */
static bool dw_pie_ch_split(const data_dims_t *input_dims,
                            const int8_t *input_data,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const dw_conv_params_t *conv_params,
                            const quant_data_t *quant_data, int slab_bytes,
                            int edge_bytes)
{
    const int channels = input_dims->channels;
    /* half the 16-channel blocks, rounded down, to the worker */
    const int cs = ((channels + 15) >> 5) << 4;
    if (output_dims->height >= DW_PIE_CH_SPLIT_ROWS || cs < 32 || cs >= channels ||
            !esp_nn_dual_core_begin()) {
        return false;
    }
    if (esp_nn_dual_core_scratch(dw_pie_ch1_scratch_size(channels) +
                                 slab_bytes + edge_bytes) == NULL) {
        esp_nn_dual_core_end();
        return false;
    }
    dw_pie_mt_job_t job = {
        .input_dims = *input_dims, .input_data = input_data,
        .filter_dims = filter_dims, .filter_data = filter_data, .bias = bias,
        .output_dims = *output_dims, .out_data = out_data,
        .conv_params = *conv_params, .quant_data = quant_data,
        .slab_bytes = slab_bytes, .edge_bytes = edge_bytes,
        .c_beg = 0, .c_end = cs,
    };
    if (!esp_nn_dual_core_run(dw_pie_mt_worker, &job)) {
        esp_nn_dual_core_end();
        return false;
    }
    depthwise_conv_s8_ch1_pie_impl(input_dims, input_data, filter_dims, filter_data, bias,
                                   output_dims, out_data, conv_params, quant_data,
                                   scratch_buffer, slab_bytes, edge_bytes, cs, channels);
    esp_nn_dual_core_wait();
    esp_nn_dual_core_end();
    return true;
}

void esp_nn_depthwise_conv_s8_riscv_pie(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data)
{
    ESP_NN_PIE_ENABLE();

    if (dw_pie_ch1_ok(input_dims, conv_params)) {
        if (scratch_buffer == NULL) {
            esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims,
                                          filter_data, bias, output_dims,
                                          out_data, conv_params, quant_data);
            return;
        }
        const int slab_bytes = dw_pie_slab_size(input_dims, filter_dims, output_dims, conv_params);
        /* as in the scratch getter */
        const int edge_bytes = slab_bytes ? 0 :
                dw_pie_edge_size(input_dims, filter_dims, output_dims, conv_params);
        if (!dw_pie_ch_split(input_dims, input_data, filter_dims, filter_data,
                             bias, output_dims, out_data, conv_params, quant_data,
                             slab_bytes, edge_bytes) &&
                !dw_pie_rows_split(input_dims, input_data, filter_dims, filter_data,
                               bias, output_dims, out_data, conv_params, quant_data,
                               slab_bytes, edge_bytes)) {
            depthwise_conv_s8_ch1_pie_impl(input_dims, input_data, filter_dims, filter_data,
                                           bias, output_dims, out_data, conv_params,
                                           quant_data, scratch_buffer, slab_bytes, edge_bytes,
                                           0, input_dims->channels);
        }
        return;
    }

    /* Fall back to generic optimized */
    esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims, filter_data,
                                  bias, output_dims, out_data, conv_params, quant_data);
}
