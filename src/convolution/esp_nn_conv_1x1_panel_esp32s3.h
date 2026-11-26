/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

/* The resident filter panel is sized against the data cache the image is
 * actually built with (64 KB, 8-way, 64-byte lines by default on ESP32-S3):
 * at most half of it, so the staged windows, the transpose scratch and the
 * output staging that stream past the panel still have ways to live in. */
#if defined(CONFIG_ESP32S3_DATA_CACHE_SIZE)
#define ESP_NN_S3_DCACHE_BYTES      CONFIG_ESP32S3_DATA_CACHE_SIZE
#else
#define ESP_NN_S3_DCACHE_BYTES      (64 * 1024)
#endif

#define ESP_NN_S3_PANEL_BYTES       (ESP_NN_S3_DCACHE_BYTES / 2)

/*
 * OC-panel driver for mult8 1x1 convolutions: a panel of filter rows stays
 * L1-resident while the batched assembly sweeps the spatial positions.
 * Also used by the 3x3 path, whose staged im2col windows are exactly a 1x1
 * problem with in_channels = 9 * in_channels.
 *
 * `scratch` needs esp_nn_conv_1x1_panel_scratch_size(in_channels) bytes.
 */
void esp_nn_conv_s8_mult8_1x1_oc_panel(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch);

/* Work area of esp_nn_conv_s8_mult8_1x1_batched_tail(): the asm's
 * 16 * in_channels transpose buffer, then 8 padded tail positions of input
 * (8 * in_channels) and of output (8 * out_channels). Callers add their own
 * alignment slack. */
static inline int esp_nn_conv_1x1_batched_work_bytes(int in_channels, int out_channels)
{
    return 24 * in_channels + 8 * out_channels;
}

/* Output channels per panel for a filter behind the cache: a panel of rows
 * fits ESP_NN_S3_PANEL_BYTES, with at least 16 channels so that wide rows
 * do not re-read the staged positions for every one or two channels. */
static inline int esp_nn_conv_1x1_panel_tile(int in_channels, int out_channels)
{
    int oc_tile = (ESP_NN_S3_PANEL_BYTES / in_channels) & ~7;
    if (oc_tile < 16) {
        oc_tile = 16;
    }
    if (oc_tile > 1024) {
        oc_tile = 1024;
    }
    if (oc_tile > out_channels) {
        oc_tile = out_channels;
    }
    return oc_tile;
}

/* Like esp_nn_conv_s8_mult8_1x1_oc_panel, with a work area of `work_bytes`.
 * It needs at least 24 * in_channels + 40 * tile + 48 bytes, with tile from
 * esp_nn_conv_1x1_panel_tile(): the output staging then holds 32 positions.
 * With less, it writes no output. */
void esp_nn_conv_s8_mult8_1x1_oc_panel_ex(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch,
        int work_bytes);

/* Panel work area: transpose + tails (24 * in_channels), the staging buffer
 * the driver scatters from (16 KB), and alignment slack. */
static inline int esp_nn_conv_1x1_panel_scratch_size(int in_channels)
{
    return 24 * in_channels + 16 * 1024 + 64;
}
