/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

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

/* Panel work area: transpose + tails (24 * in_channels), the staging buffer
 * the driver scatters from (16 KB), and alignment slack. */
static inline int esp_nn_conv_1x1_panel_scratch_size(int in_channels)
{
    return 24 * in_channels + 16 * 1024 + 64;
}
