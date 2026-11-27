/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Shape dispatch and scratch sizing of the ESP32 (MAC16) conv and depthwise
 * kernels. The kernels and the scratch-size getters both use it. Keep this
 * free of target code: the getters are built on a host too, so a code
 * generator can size scratch ahead of time.
 */
#pragma once

/* Input rows per depthwise band. The asm uses it for its tap offsets. */
#define DW_ROWS 16

#ifndef __ASSEMBLER__

#include <stdbool.h>
#include <stdint.h>
#include <esp_nn_defs.h>

/* Conv scratch budget: the s16 filter row, then a band of s16 rows plus one
 * int32 sum per pixel. A bigger band widens each filter row fewer times. */
#define CONV_SCRATCH_BYTES 16384

/* Depthwise scratch budget for the planes and the int32 sums of a chunk. */
#define DW_BAND_BYTES 16380

static inline int32_t esp32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

/* Samples per im2col row: the filter window, padded to a multiple of 4. */
static inline int32_t conv_row_len(const data_dims_t *filter_dims)
{
    return (filter_dims->width * filter_dims->height * filter_dims->channels + 3) & ~3;
}

/* Output pixels per band, at least 1. */
static inline int32_t conv_band_pixels(int32_t row_len, int32_t n_pix)
{
    int32_t band = (CONV_SCRATCH_BYTES - row_len * 2 - 32) / (row_len * 2 + 4);
    band = band < 1 ? 1 : band;
    return esp32_min(band, n_pix);
}

/* Whether every output window has input rows and input columns. Explicit
 * output dims can put outputs wholly in the padding or past the input. */
static inline bool conv_esp32_windows_in(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    return pad_wd < filter_dims->width && pad_ht < filter_dims->height &&
           (output_dims->width - 1) * conv_params->stride.width - pad_wd < input_dims->width &&
           (output_dims->height - 1) * conv_params->stride.height - pad_ht < input_dims->height;
}

/* The MAC16 conv takes every conv that is not grouped, when:
 * - it has input and output channels;
 * - one band of 1 pixel fits the scratch budget;
 * - every window has input (conv_esp32_windows_in). The gather does not
 *   handle windows without input. */
static inline bool conv_esp32_supported(const data_dims_t *input_dims,
                                        const data_dims_t *filter_dims,
                                        const data_dims_t *output_dims,
                                        const conv_params_t *conv_params)
{
    const int32_t row_len = conv_row_len(filter_dims);
    return input_dims->channels > 0 && output_dims->channels > 0 &&
           input_dims->channels == filter_dims->channels &&
           CONV_SCRATCH_BYTES - row_len * 2 - 32 >= row_len * 2 + 4 &&
           conv_esp32_windows_in(input_dims, filter_dims, output_dims, conv_params);
}

static inline int esp_nn_conv_scratch_size_esp32_impl(const data_dims_t *input_dims,
                                                      const data_dims_t *filter_dims,
                                                      const data_dims_t *output_dims,
                                                      const conv_params_t *conv_params)
{
#if defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE)
    /* The MAC16 kernels do not run on these builds (esp32_mac16_data_ok). */
    (void) input_dims; (void) filter_dims; (void) output_dims; (void) conv_params;
    return 0;
#endif
    if (!conv_esp32_supported(input_dims, filter_dims, output_dims, conv_params)) {
        return 0; /* the reference kernel needs none */
    }
    const int32_t row_len = conv_row_len(filter_dims);
    const int32_t band = conv_band_pixels(row_len,
                                          output_dims->width * output_dims->height);
    /* input band + 4, filter row + 4 (s16), band int32 sums, 4 for alignment */
    return ((band + 1) * row_len + 8) * 2 + band * 4 + 4;
}

typedef struct {
    int32_t band_rows;  /* output rows per band */
    int32_t plane_wd;   /* columns per plane */
    int32_t chunk;      /* channels per chunk */
    int32_t sums_bytes;
} dw_layout_t;

/* Band and chunk layout of the MAC16 3x3 depthwise. Returns false if the
 * shape does not fit the budget. */
static inline bool dw_layout(const data_dims_t *input_dims, const data_dims_t *output_dims,
                             const dw_conv_params_t *conv_params, dw_layout_t *l)
{
    const int32_t channels = input_dims->channels;
    const int32_t out_wd = output_dims->width;
    if (channels <= 0 || out_wd <= 0 || output_dims->height <= 0 ||
            conv_params->stride.width <= 0 || conv_params->stride.height <= 0) {
        return false;
    }
    l->band_rows = esp32_min((DW_ROWS - 3) / conv_params->stride.height + 1,
                             output_dims->height);
    const int32_t stride_wd = conv_params->stride.width;
    /* The asm loads one word of the column after the last pixel. */
    l->plane_wd = (out_wd - 1) * stride_wd + 3 + (stride_wd > 2 ? stride_wd - 2 : 0);
    l->sums_bytes = l->band_rows * out_wd * 4;
    const int32_t plane_bytes = l->plane_wd * DW_ROWS * 2;
    int32_t chunk = (DW_BAND_BYTES - l->sums_bytes) / plane_bytes;
    if (chunk < 1) {
        return false;
    }
    /* Balance the chunks. */
    const int32_t n_chunks = (channels + chunk - 1) / chunk;
    l->chunk = (channels + n_chunks - 1) / n_chunks;
    /* Whole groups of 4 channels take the fast widen (esp_nn_dw_widen_u8_esp32). */
    const int32_t chunk4 = (l->chunk + 3) & ~3;
    if (chunk4 <= chunk && chunk4 <= channels) {
        l->chunk = chunk4;
    }
    return true;
}

/* The MAC16 depthwise takes 3x3 filters whose layout fits the budget. */
static inline bool dw_esp32_layout(const data_dims_t *input_dims,
                                   const data_dims_t *filter_dims,
                                   const data_dims_t *output_dims,
                                   const dw_conv_params_t *conv_params, dw_layout_t *l)
{
    return filter_dims->width == 3 && filter_dims->height == 3 &&
           dw_layout(input_dims, output_dims, conv_params, l);
}

static inline int esp_nn_depthwise_conv_scratch_size_esp32_impl(const data_dims_t *input_dims,
                                                                const data_dims_t *filter_dims,
                                                                const data_dims_t *output_dims,
                                                                const dw_conv_params_t *conv_params)
{
#if defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE)
    /* The MAC16 kernels do not run on these builds (esp32_mac16_data_ok). */
    (void) input_dims; (void) filter_dims; (void) output_dims; (void) conv_params;
    return 0;
#endif
    dw_layout_t l;
    if (!dw_esp32_layout(input_dims, filter_dims, output_dims, conv_params, &l)) {
        return 0; /* the generic kernel needs none */
    }
    return l.chunk * l.plane_wd * DW_ROWS * 2 + l.sums_bytes + 4;
}

#endif /* __ASSEMBLER__ */
