/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 3x3 convolution for small feature maps (in_channels % 32 == 0).
 *
 * When the padded input fits comfortably in the data cache, the cost is
 * the filter traffic. The staged path streams the whole filter once per
 * chunk of positions. Here the loop runs output channels outermost, so
 * each filter row is read from memory once and then reused for every
 * position while it is hot. The padded input copy stays cached.
 */

#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include "esp_nn_conv_1x1_panel_esp32s3.h"
#include "esp_nn_conv_dispatch_esp32s3.h"

void esp_nn_conv_s8_3x3_small_map(const int8_t *input,
                                  const uint16_t input_wd,
                                  const uint16_t input_ht,
                                  const uint16_t in_channels,
                                  const int32_t input_offset,
                                  const uint16_t pad_wd,
                                  const uint16_t pad_ht,
                                  const uint16_t stride_wd,
                                  const uint16_t stride_ht,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  int8_t *out_data,
                                  const uint16_t out_wd,
                                  const uint16_t out_ht,
                                  const uint16_t out_channels,
                                  const int32_t out_offset,
                                  const int32_t *out_shift,
                                  const int32_t *out_mult,
                                  const int32_t activation_min,
                                  const int32_t activation_max,
                                  void *scratch)
{
    const int wp = esp_nn_conv_s8_3x3_small_map_padded_len(out_wd, stride_wd);
    const int hp = esp_nn_conv_s8_3x3_small_map_padded_len(out_ht, stride_ht);
    const int32_t padded_row = (int32_t)wp * in_channels;
    const int8_t pad_val = (int8_t)(-input_offset);

    int8_t *padded = (int8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);

    /* Padded cells hold -input_offset, so (cell + offset) * w is zero and
     * the kernel's per-channel offset correction stays exact. */
    for (int y = 0; y < hp; y++) {
        int8_t *dst = padded + y * padded_row;
        const int in_y = y - pad_ht;
        if (in_y < 0 || in_y >= input_ht) {
            memset(dst, pad_val, padded_row);
            continue;
        }
        const int x0 = pad_wd < wp ? pad_wd : wp;
        int x1 = pad_wd + input_wd;
        if (x1 > wp) {
            x1 = wp;
        }
        memset(dst, pad_val, x0 * in_channels);
        if (x1 > x0) {
            memcpy(dst + x0 * in_channels,
                   input + (int32_t)in_y * input_wd * in_channels,
                   (x1 - x0) * in_channels);
        } else {
            x1 = x0;
        }
        memset(dst + x1 * in_channels, pad_val, (wp - x1) * in_channels);
    }

    /* Output channels outermost: each filter window is read once. */
    esp_nn_conv_s8_3x3_small_map_accx(padded, padded_row, in_channels,
                                      stride_wd * in_channels,
                                      stride_ht * padded_row, out_wd, out_ht,
                                      filter_data, bias, input_offset,
                                      out_data, out_channels, out_offset,
                                      out_shift, out_mult,
                                      activation_min, activation_max);
}
