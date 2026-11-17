/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Shape-dispatch constants and scratch sizing shared by the ESP32-P4 and
 * ESP32-S31 conv/depthwise kernels and their scratch-size getters. Keep this
 * free of target code: the getters are built on a host too, so a code
 * generator can size scratch ahead of time.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <esp_nn_defs.h>

#include "esp_nn_conv_oc16_riscv_pie.h"

/*
 * Im2col batch paths: offset of the fast requant table in the 16-byte
 * aligned work area. The table follows the 16-pixel staging, or the oc16
 * work area when that kernel runs (oc16). The kernel and the scratch
 * getter both use this.
 */
static inline int32_t esp_nn_conv_im2col_tab_offset(bool oc16, int32_t out_ch,
                                                    int32_t window_len)
{
    return oc16 ? esp_nn_conv_oc16_work_size(window_len, out_ch) :
           16 * window_len;
}

/* Words per output channel of that table: 4 for the oc16 kernel (it
 * keeps e1 there too), else 3. */
static inline int32_t esp_nn_conv_im2col_tab_words(bool oc16)
{
    return oc16 ? ESP_NN_CONV_OC16_TAB_WORDS : 3;
}

/* L1D cache budget: use half of 64KB to leave room for filter streaming */
#define L1D_BUDGET 32768

/* Largest filter window the im2col QACC batch path stages for 16 pixels
 * (16 * 2304 = 36 KB of staging). Covers 3x3 windows up to 256 channels. */
#define ESP_NN_IM2COL_BATCH_MAX_WINDOW 2304

/* Tile plan for the padded tiny-window conv. Computed by BOTH the scratch
 * getter and the kernel from the same inputs, so the two can never disagree
 * on the path taken or on the bytes staged: a divergence here silently
 * overruns the caller's arena. */
typedef struct {
    int eff_ch;        /* channels after PIE lane padding (in_ch if none) */
    int filt_aligned;  /* bytes of the channel-padded filter copy, 0 if none */
    int row_bytes;     /* one staged (padded-width) input row */
    int tile_T;        /* output rows per tile */
    int staged_rows;   /* input rows staged for the tallest tile */
    int scratch_bytes; /* filter_sum + filt_aligned + staging */
} conv_tile_plan_t;

static inline void conv_plan_tiles(int input_wd, int in_ch,
                            int filter_wd, int filter_ht, int out_ch, int out_ht,
                            int pad_wd, int stride_ht, conv_tile_plan_t *p)
{
    p->eff_ch = in_ch;
    p->filt_aligned = 0;
    if (filter_wd * in_ch < 16) {
        /* PIE row dot needs 16 lanes: pad channels up */
        p->eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
        p->filt_aligned = filter_wd * filter_ht * p->eff_ch * out_ch;
    }
    p->row_bytes = (input_wd + 2 * pad_wd) * p->eff_ch;
    const int fixed = out_ch * 4 + p->filt_aligned; /* filter_sum + filter copy */

    /* Monolithic by default. The kernel stages every input row the output
     * needs, (out_ht - 1) * stride + filter_ht of them; for TFLite SAME shapes
     * with trailing pad > leading pad that exceeds input_ht + 2 * pad_ht, so
     * size from the rows actually staged, not from the padded input. */
    p->tile_T = out_ht;
    if (((out_ht - 1) * stride_ht + filter_ht) * p->row_bytes + fixed > L1D_BUDGET) {
        const int budget = L1D_BUDGET - fixed;
        if (filter_ht * p->row_bytes <= budget) {
            p->tile_T = (budget - filter_ht * p->row_bytes)
                        / (stride_ht * p->row_bytes) + 1;
        } else {
            /* Even one filter-height band overflows L1: take the smallest
             * tile and let it spill rather than staging the whole input. */
            p->tile_T = 1;
        }
        if (p->tile_T < 1) p->tile_T = 1;
        if (p->tile_T > out_ht) p->tile_T = out_ht;
    }
    p->staged_rows = (p->tile_T - 1) * stride_ht + filter_ht;
    p->scratch_bytes = fixed + p->staged_rows * p->row_bytes;
}


static inline int esp_nn_conv_scratch_size_riscv_pie_impl(const data_dims_t *input_dims,
                                                         const data_dims_t *filter_dims,
                                                         const data_dims_t *output_dims,
                                                         const conv_params_t *conv_params)
{
    /* Grouped conv runs each group through the standard path with repacked
     * slices: inner requirement (per-group dims) + input slice + output
     * staging. */
    if (filter_dims->channels && filter_dims->channels < input_dims->channels &&
            input_dims->channels % filter_dims->channels == 0) {
        const int32_t groups_ = input_dims->channels / filter_dims->channels;
        if (groups_ > 1 && output_dims->channels % groups_ == 0) {
            data_dims_t in_g_ = *input_dims;
            data_dims_t out_g_ = *output_dims;
            in_g_.channels = filter_dims->channels;
            out_g_.channels = output_dims->channels / groups_;
            const int inner_ = esp_nn_conv_scratch_size_riscv_pie_impl(&in_g_, filter_dims, &out_g_, conv_params);
            const int32_t in_slice_ = (int32_t)input_dims->width * input_dims->height * filter_dims->channels;
            const int32_t out_slice_ = (int32_t)output_dims->width * output_dims->height * out_g_.channels;
            return inner_ + in_slice_ + out_slice_ + 64;
        }
    }

    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 32; /* extra buffer for alignment */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        if (in_ch < 16) {
            /* Channel-padding path: filter_sum + padded_filter + padded_input */
            int filter_sum_sz = out_ch * 4;
            int padded_filter_sz = 16 * out_ch;
            int padded_input_sz = 32; /* 16 bytes + alignment */
            return filter_sum_sz + padded_filter_sz + padded_input_sz + align_buf_size;
        }
        int transpose_buf_size = 2 * (8 * new_channels);
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0;
        }
        if (in_ch % 8) {
            input_scratch = input_wd * input_ht * new_channels;
        } else {
            input_scratch = 0;
        }
        filter_scratch = new_channels * out_ch;
        return input_scratch + filter_scratch + transpose_buf_size + align_buf_size;
    } else {
        new_channels = (in_ch + 15) & ~15;
        int offset_acc_scratch = out_ch * 4;

        if (pad_wd == 0 && pad_ht == 0 && filter_wd * in_ch >= 16 &&
                !(filter_wd * filter_ht * in_ch >= 128 &&
                  filter_wd * filter_ht * in_ch <= ESP_NN_IM2COL_BATCH_MAX_WINDOW &&
                  out_ch >= 16)) {
            /* Direct no-pad path: no input scratch needed. Shapes whose
             * window fits the im2col QACC batch are routed there instead
             * (see the dispatcher) and sized by the im2col branch below. */
            input_scratch = 0;
            filter_scratch = filter_wd * filter_ht * new_channels * out_ch;
            return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
        }

        /* Im2col path: scratch = filter_sum + im2col_buf, plus the
         * 16-pixel transposed staging for the QACC batch path (bounded to
         * window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW, matching the
         * kernel's batch condition).
         * Padded convs with SIMD-wide rows and a filter beyond the L1 panel
         * budget route to the tiled path (sized by the padded-case block
         * below), matching the dispatcher. */
        if (filter_wd * filter_ht * in_ch >= 16 &&
                !((pad_wd != 0 || pad_ht != 0) && filter_wd * in_ch >= 16 &&
                  (int32_t)filter_wd * filter_ht * in_ch * out_ch > 96 * 1024)) {
            int window_len = filter_wd * filter_ht * in_ch;
            int im2col_scratch = window_len;  /* one window buffer */
            if (window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW) {
                /* 16 bytes to align the work area, then the table. */
                const bool oc16 = esp_nn_conv_oc16_ok(filter_wd, in_ch,
                        out_ch, output_dims->width * output_dims->height);
                im2col_scratch += 16 + esp_nn_conv_im2col_tab_offset(oc16,
                        out_ch, window_len) + 4 * out_ch *
                        esp_nn_conv_im2col_tab_words(oc16);
            }
            return offset_acc_scratch + im2col_scratch + align_buf_size;
        }

        if (pad_wd == 0 && pad_ht == 0) {
            /* Very small window (< 16 elements total): tiled path */
            int eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
            int filt_aligned = filter_wd * filter_ht * eff_ch * out_ch;
            int tile_input = input_wd * input_ht * eff_ch;
            return offset_acc_scratch + filt_aligned + tile_input + align_buf_size;
        }

        /* Padded tiny-window case: the tiled kernel and this getter share
         * conv_plan_tiles(), so the staging buffer is sized with the same
         * eff_ch, the same tile decision and the same staged-row count the
         * kernel will use. */
        conv_tile_plan_t plan;
        conv_plan_tiles(input_wd, in_ch, filter_wd, filter_ht, out_ch,
                        output_dims->height, pad_wd, stride_ht, &plan);
        return plan.scratch_bytes + align_buf_size;
    }
    return align_buf_size;
}

/* Position arrays of the ch_mult == 1 kernel, kept in the scratch instead
 * of the stack (dw_pie_pos_t in esp_nn_depthwise_conv_riscv_pie.c). */
#define DW_PIE_POS_BYTES 832

/* Scratch of the ch_mult == 1 kernel, in order:
 * - the position arrays (DW_PIE_POS_BYTES);
 * - the per-channel comb, 16-byte aligned and padded to 16 channels, so a
 *   vector load of a partial block stays inside it;
 * - the accumulators of 32 positions x 16 lanes;
 * - a 16-byte buffer;
 * - the int16 table of esp_nn_dw_vrq_riscv_pie, 14 bytes per padded
 *   channel. */
static inline int dw_pie_ch1_scratch_size(int channels)
{
    const int ch_pad = (channels + 15) & ~15;
    return 16 + DW_PIE_POS_BYTES + ch_pad * 4 + (32 * 16 + 4) * 4 + 14 * ch_pad;
}

static inline int dw_pie_min(int a, int b)
{
    return a < b ? a : b;
}

/* Output positions [lo, hi) along one axis whose window lies inside the
 * input, with hi >= lo. The getter and the kernel both use these. */
static inline int dw_pie_inner_lo(int pad, int stride, int out)
{
    return dw_pie_min(out, (pad + stride - 1) / stride);
}

static inline int dw_pie_inner_hi(int in, int pad, int stride, int filt,
                                  int out, int lo)
{
    const int end = in + pad < filt ? 0 :
                    dw_pie_min(out, (in + pad - filt) / stride + 1);
    return end > lo ? end : lo;
}

/* The padded slab of one 16-channel block takes up to this many bytes.
 * A layer where one band of output rows needs more takes the non-slab
 * path. */
#define DW_PIE_SLAB_MAX (16 * 1024)

/* Output rows per band of the input slab: as many as fit DW_PIE_SLAB_MAX,
 * at most out_ht, or 0 when not even one fits. The slab size comes from
 * this band. The kernel may run shorter bands (dw_pie_slab_band_l2), and
 * the size does not depend on the device. */
static inline int dw_pie_slab_band(const data_dims_t *filter_dims,
                                   const data_dims_t *output_dims,
                                   const dw_conv_params_t *conv_params)
{
    const int s_ht = conv_params->stride.height;
    const int row = ((output_dims->width - 1) * conv_params->stride.width +
                     filter_dims->width) * 16;
    const int rows = DW_PIE_SLAB_MAX / row;
    if (rows < filter_dims->height) {
        return 0;
    }
    return dw_pie_min((rows - filter_dims->height) / s_ht + 1,
                      (int) output_dims->height);
}

/* Bytes of one band of the padded input slab of a 16-channel block plus
 * the filter block repacked to 16 bytes per tap, or 0 when the slab path
 * is not taken. In the slab every window lies inside, so all positions
 * take the inner-window MAC. The slab path runs channel blocks outermost,
 * so it reads the input once per block. It is taken when:
 * - at least 1/8 of the positions have a window that crosses the input
 *   edge, as the edge windows cost far more than the copy, or
 * - there is a single channel block, and some window crosses the input
 *   edge or the block is partial. Without edges, a full block reads the
 *   input once on the non-slab path too, while a partial block (8-15
 *   channels) would run in scalar C there. */
static inline int dw_pie_slab_size(const data_dims_t *input_dims,
                                   const data_dims_t *filter_dims,
                                   const data_dims_t *output_dims,
                                   const dw_conv_params_t *conv_params)
{
    const int out_wd = output_dims->width, out_ht = output_dims->height;
    if (out_wd == 0 || out_ht == 0) {
        return 0;
    }
    /* output positions whose window is inside the input, per axis */
    const int s_wd = conv_params->stride.width, s_ht = conv_params->stride.height;
    const int p_wd = conv_params->padding.width, p_ht = conv_params->padding.height;
    const int x_lo = dw_pie_inner_lo(p_wd, s_wd, out_wd);
    const int y_lo = dw_pie_inner_lo(p_ht, s_ht, out_ht);
    const int ix = dw_pie_inner_hi(input_dims->width, p_wd, s_wd, filter_dims->width,
                                   out_wd, x_lo) - x_lo;
    const int iy = dw_pie_inner_hi(input_dims->height, p_ht, s_ht, filter_dims->height,
                                   out_ht, y_lo) - y_lo;
    if (input_dims->channels > 16 && ix > 0 && iy > 0 &&
            8 * (int64_t) ix * iy > 7 * (int64_t) out_wd * out_ht) {
        return 0;
    }
    if (ix == out_wd && iy == out_ht && input_dims->channels % 16 == 0) {
        return 0;
    }
    const int band = dw_pie_slab_band(filter_dims, output_dims, conv_params);
    if (band == 0) {
        return 0;
    }
    const int slab_w = (out_wd - 1) * s_wd + filter_dims->width;
    return (((band - 1) * s_ht + filter_dims->height) * slab_w +
            filter_dims->height * filter_dims->width) * 16;
}

/* Bytes of the padded edge slab of the 3x3 kernel without the slab path:
 * the top and bottom output rows and the left and right columns outside
 * the inner positions run from a padded copy of their input. A side strip
 * runs in chunks of at least 16 rows, and a top or bottom strip in chunks
 * of columns, so the slab stays within DW_PIE_SLAB_MAX. 0, so the edges
 * take the generic edge path, when:
 * - there is no edge, or the filter is not 3x3;
 * - the channels are not whole 16-channel blocks, as the kernel takes the
 *   edge slab only then;
 * - even one column of a strip or 16 rows of a side would not fit. */
static inline int dw_pie_edge_size(const data_dims_t *input_dims,
                                   const data_dims_t *filter_dims,
                                   const data_dims_t *output_dims,
                                   const dw_conv_params_t *conv_params)
{
    const int out_wd = output_dims->width, out_ht = output_dims->height;
    if (filter_dims->width != 3 || filter_dims->height != 3 || out_wd == 0 || out_ht == 0 ||
            (input_dims->channels & 15) != 0) {
        return 0;
    }
    const int s_wd = conv_params->stride.width, s_ht = conv_params->stride.height;
    const int p_wd = conv_params->padding.width, p_ht = conv_params->padding.height;
    const int x_lo = dw_pie_inner_lo(p_wd, s_wd, out_wd);
    const int y_lo = dw_pie_inner_lo(p_ht, s_ht, out_ht);
    const int x_hi = dw_pie_inner_hi(input_dims->width, p_wd, s_wd, 3, out_wd, x_lo);
    const int y_hi = dw_pie_inner_hi(input_dims->height, p_ht, s_ht, 3, out_ht, y_lo);
    const int tb = y_lo > out_ht - y_hi ? y_lo : out_ht - y_hi;
    const int lr = x_lo > out_wd - x_hi ? x_lo : out_wd - x_hi;
    int vecs = 0, need = 0;
    if (tb > 0) {
        vecs = ((tb - 1) * s_ht + 3) * ((out_wd - 1) * s_wd + 3);
        need = ((tb - 1) * s_ht + 3) * 3;
    }
    if (lr > 0) {
        const int side = (15 * s_ht + 3) * ((lr - 1) * s_wd + 3);
        vecs = vecs > side ? vecs : side;
        need = need > side ? need : side;
    }
    if (vecs == 0 || (need + 9) * 16 > DW_PIE_SLAB_MAX) {
        return 0;
    }
    return dw_pie_min((vecs + 9) * 16, DW_PIE_SLAB_MAX);
}

/* The PIE ch_mult == 1 kernel feeds -in_offset as an int8 lane value. */
static inline bool dw_pie_ch1_ok(const data_dims_t *input_dims,
                                 const dw_conv_params_t *conv_params)
{
    return conv_params->ch_mult == 1 && input_dims->channels >= 8 &&
           conv_params->in_offset >= -127 && conv_params->in_offset <= 128;
}

static inline int esp_nn_depthwise_conv_scratch_size_riscv_pie_impl(const data_dims_t *input_dims,
                                                                   const data_dims_t *filter_dims,
                                                                   const data_dims_t *output_dims,
                                                                   const dw_conv_params_t *conv_params)
{
    if (dw_pie_ch1_ok(input_dims, conv_params)) {
        const int slab = dw_pie_slab_size(input_dims, filter_dims, output_dims, conv_params);
        return dw_pie_ch1_scratch_size(input_dims->channels) + slab +
               (slab ? 0 : dw_pie_edge_size(input_dims, filter_dims, output_dims, conv_params));
    }
    return 0;
}
