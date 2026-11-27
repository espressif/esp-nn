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

#include <stdbool.h>
#include <stdint.h>
#include <esp_nn_defs.h>

#include "esp_nn_conv_1x1_panel_esp32s3.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
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

/* Width or height of the zero-padded map the small-map kernel builds: one
 * 3-tap window at `stride` for each of `out` outputs. */
static inline int esp_nn_conv_s8_3x3_small_map_padded_len(int out, int stride)
{
    return (out - 1) * stride + 3;
}

static inline int32_t esp_nn_conv_s8_3x3_small_map_bytes(int in_channels, int out_wd,
                                                         int out_ht, int stride_wd,
                                                         int stride_ht)
{
    return (int32_t)esp_nn_conv_s8_3x3_small_map_padded_len(out_wd, stride_wd)
           * esp_nn_conv_s8_3x3_small_map_padded_len(out_ht, stride_ht) * in_channels;
}

/* The padded map and one filter window must fit the data cache together,
 * so both stay resident across the position loop. It only pays when the
 * whole filter does not fit the cache: a cached filter is cheap to
 * re-stream, and the staged path's batched dot is faster on short rows.
 * The kernel runs each window row in chunks of 6 blocks, so it needs
 * in_channels % 32 == 0. */
static inline int esp_nn_conv_s8_3x3_small_map_fits(int in_channels, int out_channels,
                                                     int out_wd, int out_ht,
                                                     int stride_wd, int stride_ht,
                                                     int32_t cache_bytes)
{
    const int32_t map_bytes = esp_nn_conv_s8_3x3_small_map_bytes(in_channels, out_wd, out_ht,
                                                                 stride_wd, stride_ht);
    const int32_t filter_bytes = (int32_t)9 * in_channels * out_channels;
    return in_channels % 32 == 0
           && map_bytes + 9 * in_channels <= cache_bytes
           && filter_bytes > cache_bytes;
}

static inline int esp_nn_conv_s8_3x3_small_map_can_use(int in_channels, int out_channels,
                                                        int out_wd, int out_ht,
                                                        int stride_wd, int stride_ht)
{
    return esp_nn_conv_s8_3x3_small_map_fits(in_channels, out_channels, out_wd, out_ht,
                                             stride_wd, stride_ht, ESP_NN_S3_DCACHE_BYTES);
}

/* Padded input + alignment slack. */
static inline int esp_nn_conv_s8_3x3_small_map_scratch_size(int in_channels, int out_wd,
                                                             int out_ht, int stride_wd,
                                                             int stride_ht)
{
    return esp_nn_conv_s8_3x3_small_map_bytes(in_channels, out_wd, out_ht, stride_wd, stride_ht)
           + 16;
}

/* esp_nn_conv_s8_1x1_accx_esp32s3.c: s8 MACs into ACCX for in_ch % 16 == 0.
 * Returns 1 when it ran, 0 when it declines, and -1 when it declines only
 * for a short chunk (the asm table must then be rebuilt). */
int esp_nn_conv_s8_1x1_accx_esp32s3(const int8_t *input, int spatial,
                                    uint16_t in_ch, int32_t in_offset,
                                    const int8_t *filter, const int32_t *bias,
                                    int8_t *out_data, uint16_t out_ch,
                                    int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    int32_t act_min, int32_t act_max);

void esp_nn_conv_s8_3x3_small_map_accx(const int8_t *padded,
                                       int32_t padded_row, int in_ch,
                                       int32_t step_x, int32_t step_y,
                                       int out_wd, int out_ht,
                                       const int8_t *filter,
                                       const int32_t *bias, int32_t in_offset,
                                       int8_t *out_data, int out_ch,
                                       int out_stride, int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       int32_t act_min, int32_t act_max);

/* esp_nn_conv_s8_1x1_1pos_esp32s3.c: one position as a matrix-vector
 * product. `work` is 16-byte aligned and holds in_channels bytes. */
void esp_nn_conv_s8_1x1_1pos_esp32s3(const int8_t *input,
        uint16_t in_channels, int32_t input_offset, const int8_t *filter_data,
        const int32_t *bias, int8_t *out_data, uint16_t out_channels,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, int8_t *work);

/* esp_nn_conv_s8_3x3_small_map_esp32s3.c */
void esp_nn_conv_s8_3x3_small_map(const int8_t *input, uint16_t input_wd,
        uint16_t input_ht, uint16_t in_channels, int32_t input_offset,
        uint16_t pad_wd, uint16_t pad_ht, uint16_t stride_wd,
        uint16_t stride_ht, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_wd, uint16_t out_ht,
        uint16_t out_channels, int32_t out_offset, const int32_t *out_shift,
        const int32_t *out_mult, int32_t activation_min,
        int32_t activation_max, void *scratch);

/* Output positions staged per pass. One pass streams the filter once, and
 * longer chunks spread the per-call setup of the 1x1 asm. Measured on
 * ESP32-S3, 32 beats 16 by 1-4% on yolo11n's shapes and 48 adds nothing.
 * Filters over 64 KB use ESP_NN_3X3_BATCH_POSITIONS_LARGE (below). */
#define ESP_NN_3X3_BATCH_POSITIONS          32

/* Maps with fewer positions than this take the per-pixel path. */
#define ESP_NN_3X3_BATCH_MIN_POSITIONS      16

/* A filter over 64 KB cannot sit in internal RAM next to the arena, so every
 * chunk streams it from PSRAM again. Longer chunks stream it fewer times.
 * The panel's output staging is cut to these 32 positions, so the scratch of
 * the 128-channel class (1152-byte windows) stays under 64 KB. */
#define ESP_NN_3X3_BATCH_POSITIONS_LARGE    32

static inline int esp_nn_3x3_large_filter(int in_channels, int out_channels)
{
    return (uint32_t)in_channels * out_channels > (64 * 1024) / 9;
}

/* Short windows stage twice the positions per pass, which halves the per-OC
 * setup per output. The staging stays small enough that the scratch keeps
 * under the 64 KB that would push it to PSRAM. */
#define ESP_NN_3X3_BATCH_POSITIONS_SHORT    64
#define ESP_NN_3X3_SHORT_STAGING_BYTES      (24 * 1024)

static inline int esp_nn_3x3_batch_positions(int in_channels, int out_channels)
{
    if (esp_nn_3x3_large_filter(in_channels, out_channels)) {
        return ESP_NN_3X3_BATCH_POSITIONS_LARGE;
    }
    if (9 * in_channels * ESP_NN_3X3_BATCH_POSITIONS_SHORT <=
            ESP_NN_3X3_SHORT_STAGING_BYTES) {
        return ESP_NN_3X3_BATCH_POSITIONS_SHORT;
    }
    return ESP_NN_3X3_BATCH_POSITIONS;
}

/* Panel work area for one chunk of staged windows. A large filter gets the
 * minimum esp_nn_conv_s8_mult8_1x1_oc_panel_ex() takes, so its output staging
 * holds the 32 positions of a chunk. */
static inline int esp_nn_3x3_panel_work_bytes(int window_len_aligned,
                                              int in_channels, int out_channels)
{
    if (!esp_nn_3x3_large_filter(in_channels, out_channels)) {
        return esp_nn_conv_1x1_panel_scratch_size(window_len_aligned,
                                                  out_channels);
    }
    const int tile = esp_nn_conv_1x1_panel_tile(window_len_aligned, out_channels);
    /* Plus the output staging for 32 positions of a panel, and slack. */
    return esp_nn_conv_1x1_batched_work_bytes(window_len_aligned, tile) + 32 * tile + 48;
}

/* Staging is worth it whenever a whole filter pass is amortised over at least
 * one full chunk of positions. It is NOT a cache-residency test: the per-pixel
 * path reloads the filter only once its working set exceeds the data cache
 * (ESP_NN_S3_DCACHE_BYTES), but the staged path also replaces a per-pixel dot
 * with the batched assembly, which measured faster on every eligible yolo11n
 * layer - including ones whose filter fits the cache many times over. */
static inline int esp_nn_conv_s8_3x3_use_batch(int out_wd, int out_ht)
{
    return ((int32_t)out_wd * out_ht) >= ESP_NN_3X3_BATCH_MIN_POSITIONS;
}

/*
 * Check if a conv can use the optimized 3x3 path.
 * Requirements:
 * - filter_wd == 3 && filter_ht == 3
 * - in_channels >= 8 and in_channels % 8 == 0
 * - out_channels >= 8
 */
static inline int esp_nn_conv_s8_3x3_can_use(int filter_wd, int filter_ht,
                                              int in_channels, int out_channels)
{
    /* out_channels gate is a measured profitability bound: the per-pixel
     * im2col build amortizes across the output-channel dots, and at
     * out_channels == 1 the path measured 3.8x SLOWER than the general
     * kernel on a 3x3x32 map (S3), while out_channels == 8 wins (48x48x16
     * to 8: 1.6x). in_channels % 8 is what the batched mult8 asm needs
     * (48x48x8 to 16: 2.5x). */
    return (filter_wd == 3 && filter_ht == 3 &&
            in_channels >= 8 && (in_channels % 8) == 0 &&
            out_channels >= 8);
}

/*
 * Scratch size of the 3x3 paths other than small-map. Mirrors the dispatch:
 * staged positions need staging + the panel driver's work area, the
 * per-pixel layout needs one window + an aligned filter copy + corrections.
 */
static inline int esp_nn_conv_s8_3x3_other_scratch_size(int in_channels, int out_channels,
                                                         int out_wd, int out_ht)
{
    int window_len_aligned = (9 * in_channels + 15) & ~15;
    if (esp_nn_conv_s8_3x3_use_batch(out_wd, out_ht)) {
        /* Staged positions plus the 1x1 panel driver's work area. The filter
         * is read in place when 9 * in_channels is a multiple of 16. */
        const int batch = esp_nn_3x3_batch_positions(in_channels, out_channels);
        int staging = batch * window_len_aligned;
        /* A zero-padded filter copy when windows are padded (in_ch % 16 == 8). */
        int filter_copy = (window_len_aligned != 9 * in_channels)
                ? out_channels * window_len_aligned + 16 : 0;
        return filter_copy + staging + 32 +
               esp_nn_3x3_panel_work_bytes(window_len_aligned, in_channels, out_channels);
    }
    int im2col = window_len_aligned;
    int filter_copy = out_channels * window_len_aligned;  /* aligned, zero-padded rows */
    int corrections = out_channels * 4;
    return im2col + filter_copy + corrections + 32;
}

static inline int esp_nn_conv_s8_3x3_scratch_size(int in_channels, int out_channels,
                                                   int out_wd, int out_ht,
                                                   int stride_wd, int stride_ht)
{
    const int other = esp_nn_conv_s8_3x3_other_scratch_size(in_channels, out_channels,
                                                             out_wd, out_ht);
#if defined(CONFIG_ESP32S3_DATA_CACHE_SIZE)
    if (esp_nn_conv_s8_3x3_small_map_can_use(in_channels, out_channels, out_wd, out_ht,
                                             stride_wd, stride_ht)) {
        return esp_nn_conv_s8_3x3_small_map_scratch_size(
                in_channels, out_wd, out_ht, stride_wd, stride_ht);
    }
    return other;
#else
    /* A host build does not know the target's data cache, which decides
     * whether the small-map path runs. Size for either path when any S3
     * cache size (16, 32 or 64 KB) would pick it. */
    for (int32_t cache = 16 * 1024; cache <= 64 * 1024; cache *= 2) {
        if (esp_nn_conv_s8_3x3_small_map_fits(in_channels, out_channels, out_wd, out_ht,
                                              stride_wd, stride_ht, cache)) {
            return max(other, esp_nn_conv_s8_3x3_small_map_scratch_size(
                    in_channels, out_wd, out_ht, stride_wd, stride_ht));
        }
    }
    return other;
#endif
}

/* Short-row im2col path (esp_nn_conv_s8_im2col_s3() in esp_nn_conv_esp32s3.c):
 * output positions staged per pass, and the window layout it uses. */
#define ESP_NN_IM2COL_POSITIONS     64

static inline int esp_nn_im2col_s3_slot(int row_bytes, int filter_ht)
{
    const int packed = (row_bytes * filter_ht + 15) & ~15;
    if (row_bytes >= 8) {
        return 8;
    }
    return (16 * filter_ht <= packed + 16) ? 16 : row_bytes;
}

static inline int esp_nn_im2col_s3_window_aligned(int row_bytes, int filter_ht)
{
    const int slot = esp_nn_im2col_s3_slot(row_bytes, filter_ht);
    const int tail = row_bytes - min(row_bytes, slot);
    return ((slot + tail) * filter_ht + 15) & ~15;
}

static inline int esp_nn_conv_s8_im2col_s3_scratch_size(int row_bytes,
                                                        int filter_ht,
                                                        int out_ch)
{
    const int wla = esp_nn_im2col_s3_window_aligned(row_bytes, filter_ht);
    return out_ch * wla + ESP_NN_IM2COL_POSITIONS * wla +
           esp_nn_conv_1x1_batched_work_bytes(wla, out_ch) + 64;
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
                                               output_dims->height,
                                               stride_wd, stride_ht);
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
        int batched_tail_size = esp_nn_conv_1x1_batched_work_bytes(in_ch, out_ch) +
                                align_buf_size;
        /* OC-panel driver (filter > 24 KB): asm work + staging areas */
        int panel_size = esp_nn_conv_1x1_panel_scratch_size(in_ch, out_ch);
        if ((int32_t)in_ch * out_ch <= 24 * 1024) {
            panel_size = 0;
        }
        return max(max(existing_size, batched_tail_size), panel_size);
    } else {
        int32_t filter_row_size = filter_wd * in_ch;
        int32_t window_len = filter_wd * filter_ht * in_ch;

        /* Im2col path: filter_wd * in_ch < 16 but window_len >= 16 */
        if (filter_row_size < 16 && window_len >= 16) {
            return esp_nn_conv_s8_im2col_s3_scratch_size(filter_row_size, filter_ht, out_ch);
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

/* int8 KxK depthwise for ch_mult 1 (esp_nn_depthwise_conv_s8_kxk_esp32s3.c).
 * 31 taps of the largest product (-128 * -128) are the most that always fit
 * the 20-bit QACC lanes. */
#define ESP_NN_DW_S8_KXK_MAX_TAPS  31
/* Per 16-channel group, 16 int32 each of: rounding mask, half mask, right
 * shift, mult, left shift, bias. */
#define ESP_NN_DW_S8_KXK_REQ_BLOCK (6 * 16)

static inline int esp_nn_dw_s8_kxk_eligible_esp32s3(const data_dims_t *filter_dims,
                                                    const data_dims_t *input_dims,
                                                    const dw_conv_params_t *conv_params)
{
    const int fw = filter_dims->width, fh = filter_dims->height;
    const int ch = input_dims->channels;
    /* Padded taps hold (int8_t)-in_offset, so in_offset must be in
     * [-127, 128]. The clamp bounds are activation - out_offset, so the
     * activation limits must be int8: wider ones (such as INT32_MIN for
     * "no clamp") can overflow there. */
    return conv_params->ch_mult == 1 && ch % 8 == 0 && ch >= 16 &&
           fw * fh <= ESP_NN_DW_S8_KXK_MAX_TAPS &&
           conv_params->in_offset >= -127 && conv_params->in_offset <= 128 &&
           conv_params->activation.min >= -128 && conv_params->activation.max <= 127 &&
           conv_params->dilation.width <= 1 && conv_params->dilation.height <= 1;
}

/* Scratch regions of the KxK kernel, each a multiple of 16 bytes except
 * row_bytes and ring_bytes, which the kernel only adds at the end.
 * zero_copy: no padding on any side and cs == channels. The kernel then
 * reads the input rows in place and needs no ring (ring_bytes is 0). It
 * needs a 16-byte aligned input for that. The getter cannot see the
 * pointer, so the kernel declines an unaligned input of such a shape. */
typedef struct {
    int cs, taps, row_bytes, zero_copy;
    int filter_bytes, req_bytes, flag_bytes, tmp_bytes, ring_bytes;
} esp_nn_dw_s8_kxk_layout_t;

static inline void esp_nn_dw_s8_kxk_layout_esp32s3(const data_dims_t *input_dims,
                                                   const data_dims_t *filter_dims,
                                                   const data_dims_t *output_dims,
                                                   const dw_conv_params_t *cp,
                                                   esp_nn_dw_s8_kxk_layout_t *l)
{
    const int fw = filter_dims->width;
    const int pad_right = max(0, (output_dims->width - 1) * cp->stride.width + fw
                                 - cp->padding.width - input_dims->width);
    const int pad_bottom = max(0, (output_dims->height - 1) * cp->stride.height
                                  + filter_dims->height - cp->padding.height
                                  - input_dims->height);
    const int padded_wd = cp->padding.width + input_dims->width + pad_right;
    l->cs = (input_dims->channels + 15) & ~15;
    l->zero_copy = l->cs == input_dims->channels &&
                   (cp->padding.width | cp->padding.height | pad_right | pad_bottom) == 0;
    l->taps = fw * filter_dims->height;
    l->row_bytes = padded_wd * l->cs;
    l->filter_bytes = (l->taps * l->cs + 15) & ~15;
    l->req_bytes = l->cs / 16 * ESP_NN_DW_S8_KXK_REQ_BLOCK * 4;
    /* one byte per 16-channel group: it takes the general requant */
    l->flag_bytes = (l->cs / 16 + 15) & ~15;
    l->tmp_bytes = output_dims->width * 16;
    /* the filter_ht padded input rows an output row needs */
    l->ring_bytes = l->zero_copy ? 0 : filter_dims->height * l->row_bytes;
}

static inline int esp_nn_dw_s8_kxk_scratch_size_esp32s3(const data_dims_t *input_dims,
                                                        const data_dims_t *filter_dims,
                                                        const data_dims_t *output_dims,
                                                        const dw_conv_params_t *conv_params)
{
    esp_nn_dw_s8_kxk_layout_t l;
    esp_nn_dw_s8_kxk_layout_esp32s3(input_dims, filter_dims, output_dims, conv_params, &l);
    /* each region is 16-byte aligned, the base too */
    return 16 + l.filter_bytes + l.req_bytes + l.flag_bytes + l.tmp_bytes + l.ring_bytes;
}

/* esp_nn_depthwise_conv_s8_kxk_esp32s3.c. Returns false, having done
 * nothing, for a zero_copy shape with an unaligned input. */
bool esp_nn_dw_s8_kxk_esp32s3(const data_dims_t *input_dims,
                              const int8_t *input_data,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const dw_conv_params_t *conv_params,
                              const quant_data_t *quant_data,
                              void *scratch);

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

    if (esp_nn_dw_s8_kxk_eligible_esp32s3(filter_dims, input_dims, conv_params)) {
        return esp_nn_dw_s8_kxk_scratch_size_esp32s3(input_dims, filter_dims,
                                                     output_dims, conv_params);
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
