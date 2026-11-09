/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Shape-dispatch constants shared by the ESP32-S3 conv/depthwise kernels and
 * their scratch-size getters. Keep this free of target code: the getters are
 * built on a host too, so a code generator can size scratch ahead of time.
 */
#pragma once

#include <stdint.h>
#include <esp_nn_defs.h>

#include "esp_nn_conv_1x1_panel_esp32s3.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

/* int8 depthwise for (K,1) filters on width-1 tensors, the shape a 1-D
 * convolutional encoder produces. Measured on ESP32-S3 with a 16-layer speech
 * encoder: 461-489 ms against 478-507 ms for the int16 path it replaces, and
 * scratch for the affected layers falls from 99 kB to about 1 kB. */
#ifndef ESP_NN_DW_S8_KX1
#define ESP_NN_DW_S8_KX1 1
#endif
/* The (K,1) int8 kernel sums raw s8 x s8 products in the 20-bit QACC lanes:
 * 32 taps of the largest product (127 * 128 = 16256) are the most that
 * always fit in +-2^19. Taller filters take the general path. */
#define ESP_NN_DW_S8_KX1_MAX_TAPS 32

/* Output positions staged per pass. One pass streams the filter once, so the
 * chunk only has to be long enough to amortise that; measured on ESP32-S3,
 * 8/16/32/48 land within a few percent of each other on yolo11n's shapes and
 * 16 is the best compromise across them. */
#define ESP_NN_3X3_BATCH_POSITIONS          16

/* Staging is worth it whenever a whole filter pass is amortised over at least
 * one full chunk of positions. It is NOT a cache-residency test: the per-pixel
 * path reloads the filter only once its working set exceeds the data cache
 * (ESP_NN_S3_DCACHE_BYTES), but the staged path also replaces a per-pixel dot
 * with the batched assembly, which measured faster on every eligible yolo11n
 * layer - including ones whose filter fits the cache many times over. */
static inline int esp_nn_conv_s8_3x3_use_batch(int out_wd, int out_ht)
{
    return ((int32_t)out_wd * out_ht) >= ESP_NN_3X3_BATCH_POSITIONS;
}

/*
 * Check if a conv can use the optimized 3x3 path.
 * Requirements:
 * - filter_wd == 3 && filter_ht == 3
 * - in_channels >= 16 (SIMD worth it)
 * - in_channels % 16 == 0 (aligned for ee.vld.128)
 */
static inline int esp_nn_conv_s8_3x3_can_use(int filter_wd, int filter_ht,
                                              int in_channels, int out_channels)
{
    /* out_channels gate is a measured profitability bound: the per-pixel
     * im2col build amortizes across the output-channel dots, and at
     * out_channels == 1 the path measured 3.8x SLOWER than the general
     * kernel on a 3x3x32 map (S3), while out_channels == 16 wins. */
    return (filter_wd == 3 && filter_ht == 3 &&
            in_channels >= 16 && (in_channels % 16) == 0 &&
            out_channels >= 16);
}

/*
 * Scratch size for the 3x3 optimized path. Mirrors the dispatch below:
 * staged positions need staging + the panel driver's work area, the
 * per-pixel layout needs one window + an aligned filter copy + corrections.
 */
static inline int esp_nn_conv_s8_3x3_scratch_size(int in_channels, int out_channels,
                                                   int out_wd, int out_ht)
{
    int window_len_aligned = (9 * in_channels + 15) & ~15;
    if (esp_nn_conv_s8_3x3_use_batch(out_wd, out_ht)) {
        /* Staged positions plus the 1x1 panel driver's work area. The filter
         * is read in place, so there is no per-layer copy: this is smaller
         * than the per-pixel layout below for every shape that reaches it. */
        int staging = ESP_NN_3X3_BATCH_POSITIONS * window_len_aligned;
        return staging + esp_nn_conv_1x1_panel_scratch_size(window_len_aligned) + 32;
    }
    int im2col = window_len_aligned;
    int filter_copy = out_channels * window_len_aligned;  /* aligned, zero-padded rows */
    int corrections = out_channels * 4;
    return im2col + filter_copy + corrections + 32;
}

/* Scratch requirement of esp_nn_conv_s8_esp32s3(). The public getter in
 * esp_nn_scratch_sizes_esp32s3.c wraps this; the kernel calls it directly
 * so it stays inlined into the dispatch. */
static inline int esp_nn_conv_scratch_size_esp32s3_impl(const data_dims_t *input_dims,
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
            const int inner_ = esp_nn_conv_scratch_size_esp32s3_impl(&in_g_, filter_dims, &out_g_, conv_params);
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

    /* Mirrors the runtime dispatch: the 3x3 path takes precedence when it
     * qualifies, and needs im2col + an aligned zero-padded filter copy +
     * corrections. */
    if (esp_nn_conv_s8_3x3_can_use(filter_wd, filter_ht, in_ch, out_ch)) {
        return esp_nn_conv_s8_3x3_scratch_size(in_ch, out_ch,
                                                              output_dims->width,
                                                              output_dims->height);
    }

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 64; /* alignment (16) + assembly pre/post access margin (48) */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        /* Transpose buffer is used by both 1x1 kernels; the filter is not
         * copied by either, so no filter term. */
        int transpose_buf_size = 2 * (8 * new_channels);
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0;
        }
        /* Neither 1x1 kernel copies or pads the input: the SIMD path
         * transposes into the buffer above, the fallback reads in place
         * (any alignment, any channel count). No input term. */
        int existing_size = transpose_buf_size + align_buf_size;
        int batched_tail_size = 16 * in_ch + 8 * in_ch + 8 * out_ch +
                                align_buf_size;
        /* OC-panel driver (filter > 24 KB): asm work + staging areas */
        int panel_size = 24 * in_ch + 16 * 1024 + align_buf_size;
        if ((int32_t)in_ch * out_ch <= 24 * 1024) {
            panel_size = 0;
        }
        return max(max(existing_size, batched_tail_size), panel_size);
    } else {
        int32_t filter_row_size = filter_wd * in_ch;
        int32_t window_len = filter_wd * filter_ht * in_ch;

        /* Im2col path: filter_wd * in_ch < 16 but window_len >= 16 */
        if (filter_row_size < 16 && window_len >= 16) {
            int32_t window_len_aligned = (window_len + 15) & ~15;
            /* filter_sum + aligned_filter_copy + im2col_buf + alignment padding */
            int im2col_scratch = out_ch * 4 + 16 + out_ch * window_len_aligned + 16 + window_len_aligned;
            return im2col_scratch + align_buf_size;
        }

        new_channels = (in_ch + 15) & ~15;

        // Padded-input scratch: leading (top/left) padding from TFLite plus the
        // trailing (bottom/right) padding derived from the output extent.
        // Must match the padding applied in esp_nn_conv_s8_esp32s3().
        int pad_right = max(0, (output_dims->width - 1) * stride_wd + filter_wd - pad_wd - input_wd);
        int pad_bottom = max(0, (output_dims->height - 1) * stride_ht + filter_ht - pad_ht - input_ht);
        if (pad_wd == 0 && pad_ht == 0 && pad_right == 0 && pad_bottom == 0) {
            input_scratch = 0;
        } else {
            input_scratch = (input_wd + pad_wd + pad_right) * (input_ht + pad_ht + pad_bottom) * in_ch;
        }
        /* At most one of the two filter copies is ever made, so max(), not
         * the sum. */
        int32_t aligned_filter_row_size = ((filter_row_size + 15) / 16) * 16;
        int row_padded_copy = aligned_filter_row_size * filter_ht * out_ch;
        int pointer_align_copy = filter_wd * filter_ht * in_ch * out_ch;
        filter_scratch = max(row_padded_copy, pointer_align_copy);

        int offset_acc_scratch = out_ch * 4;
        return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
    }
    return align_buf_size;
}

/* Scratch requirement of esp_nn_depthwise_conv_s8_esp32s3(); see above. */
static inline int esp_nn_depthwise_conv_scratch_size_esp32s3_impl(const data_dims_t *input_dims,
                                                                  const data_dims_t *filter_dims,
                                                                  const data_dims_t *output_dims,
                                                                  const dw_conv_params_t *conv_params)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t ch_mult = conv_params->ch_mult;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    /* MUST mirror the kernel's dispatch. Now that the (K,1) path declines
     * nothing on alignment grounds, its requirement is knowable here:
     * bias_adj (channels int32) + a K-row window + an aligned filter copy.
     * For a 384x1x128 layer that is ~1.3 kB against 99 kB for the int16 path
     * it replaces. */
    if (ESP_NN_DW_S8_KX1 && (ch_mult == 1) && (input_wd == 1) &&
            (filter_wd == 1) && (filter_ht <= ESP_NN_DW_S8_KX1_MAX_TAPS) &&
            (out_wd == 1) && (stride_wd == 1) &&
            (pad_wd == 0) && (channels % 16 == 0)) {
        return channels * 4 + 2 * filter_ht * channels + 64;
    }

    int filter_size = filter_wd * filter_ht * channels * ch_mult;
    int pad_width = 0, pad_height = 0;

    if ((ch_mult == 1) && (channels % 8 == 0)) {
        if(filter_wd == 3 && filter_ht == 3) {
            if (channels % 16 == 0) {
                /* Mirrors the kernel: leading pad from the caller, trailing
                 * pad from whatever the output extent still needs. Covers
                 * TFLite's asymmetric "SAME" shapes on this path. */
                pad_width = pad_wd + max(0, (out_wd - 1) * stride_wd + filter_wd
                                            - pad_wd - input_wd);
                pad_height = pad_ht + max(0, (out_ht - 1) * stride_ht + filter_ht
                                             - pad_ht - input_ht);
                if (pad_width || pad_height) {
                    int full_input = (input_wd + pad_width) * (input_ht + pad_height) * channels;
                    if (full_input <= 40 * 1024) {
                        return filter_size + full_input + 16;
                    } else {
                        /* Strip-tiled: filter + as many padded input rows as
                         * fit the same 40 KB budget the monolithic path uses,
                         * never fewer than filter_ht. The kernel derives its
                         * strip height from this size (minus its 16-byte
                         * alignment pad, hence +32 not +16), so reserving
                         * only filter_ht rows here would silently degrade it
                         * to one output row per assembly call. */
                        int row_bytes = (input_wd + pad_width) * channels;
                        int rows = (40 * 1024 - filter_size - 32) / row_bytes;
                        if (rows < filter_ht) {
                            rows = filter_ht;
                        }
                        if (rows > input_ht + pad_height) {
                            rows = input_ht + pad_height;
                        }
                        return filter_size + rows * row_bytes + 32;
                    }
                } else {
                    return filter_size + 16;
                }
            } else if (channels >= 12) {
                /* ch % 8 == 0, not % 16, ch >= 12: pad channels to 16, s8 path + compaction */
                int new_ch = (channels + 15) & ~15;
                int new_filter_size = 9 * new_ch;
                int total_pad_wd = pad_wd * 2 + max(0, (out_wd * stride_wd + 2) - input_wd);
                int total_pad_ht = pad_ht * 2 + max(0, (out_ht * stride_ht + 2) - input_ht);
                int new_input_size = (input_wd + total_pad_wd) * (input_ht + total_pad_ht) * new_ch;
                int out_buf_size = out_wd * out_ht * new_ch;
                /* + channel-padded shift/mult/bias arrays (see the kernel) */
                int quant_pad_size = 3 * new_ch * (int)sizeof(int32_t);
                return new_filter_size + new_input_size + out_buf_size + quant_pad_size + 64;
            } else {
                /* ch=8: s16 path is more efficient (no channel padding overhead) */
                int input_s = input_wd * input_ht * channels;
                return  2 * (filter_size + input_s) + 32;
            }
        } else {
            int input_size = input_wd * input_ht * channels;
            int total_s16 = 2 * (filter_size + input_size);
            if (total_s16 <= 48 * 1024) {
                return total_s16 + 32;
            } else {
                /* Tiled: only need filter_s16 + tile buffer (filter_ht rows of input s16) */
                int tile_rows = filter_ht;
                int tile_s16 = 2 * input_wd * tile_rows * channels;
                return 2 * filter_size + tile_s16 + 32;
            }
        }
    } else if ((ch_mult == 1) && (channels > 3)) {
        // ch_mult=1, channels>3 case: pad channels to multiple of 8 for mult1
        int padded_channels = (channels + 7) & ~7;
        int padded_input_size = input_wd * input_ht * padded_channels;
        int padded_filter_size = filter_wd * filter_ht * padded_channels;

        // Calculate actual memory layout with 16-byte alignments (matching usage)
        size_t filter_bytes = padded_filter_size * sizeof(int16_t);
        size_t input_start = (filter_bytes + 15) & ~15;
        size_t input_bytes = padded_input_size * sizeof(int16_t);
        size_t out_start = (input_start + input_bytes + 15) & ~15;
        size_t out_bytes = out_wd * out_ht * padded_channels * sizeof(int8_t);
        size_t bias_start = (out_start + out_bytes + 15) & ~15;
        size_t bias_bytes = padded_channels * sizeof(int32_t);
        size_t shift_bytes = padded_channels * sizeof(int32_t);
        size_t mult_bytes = padded_channels * sizeof(int32_t);
        size_t total_size = bias_start + bias_bytes + shift_bytes + mult_bytes;

        return total_size + 16; // 16 for margin
    } else if (ch_mult % 4 == 0) {
        int input_size = input_wd * input_ht * channels;
        return  2 * (filter_size + input_size) + 32; // 32 for alignment
    }

    // Default fallback
    return 32;
}
