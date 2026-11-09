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

#include <stdint.h>
#include <esp_nn_defs.h>

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
                im2col_scratch += window_len * 16 + 16;
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

static inline int esp_nn_depthwise_conv_scratch_size_riscv_pie_impl(const data_dims_t *input_dims,
                                                                   const data_dims_t *filter_dims,
                                                                   const data_dims_t *output_dims,
                                                                   const dw_conv_params_t *conv_params)
{
    (void) filter_dims;
    (void) output_dims;
    if (conv_params->ch_mult == 1 && input_dims->channels >= 8) {
        return input_dims->channels * sizeof(int32_t);
    }
    return 0;
}
