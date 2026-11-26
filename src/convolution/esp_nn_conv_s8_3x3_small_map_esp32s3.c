/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 3x3 convolution for small feature maps (in_channels % 16 == 0).
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
#include <common_functions.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_conv_inline_esp32s3.h"
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
    const int row_len = 3 * in_channels;        /* one filter row, % 16 == 0 */
    const int window_len = 9 * in_channels;
    const int blocks = row_len >> 4;
    const int32_t padded_row = (int32_t)wp * in_channels;
    const int8_t pad_val = (int8_t)(-input_offset);

    int8_t *padded = (int8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);
    int8_t *frow_copy = padded + (int32_t)hp * padded_row;

    /* Padded cells hold -input_offset, so (cell + offset) * w is zero and
     * the per-channel offset correction below stays exact. */
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

    const int copy_filter = ((uintptr_t)filter_data & 15) != 0;
    const int32_t step_x = stride_wd * in_channels;
    const int32_t step_y = stride_ht * padded_row;

    for (int oc = 0; oc < out_channels; oc++) {
        const int8_t *frow = filter_data + (int32_t)oc * window_len;
        if (copy_filter) {
            memcpy(frow_copy, frow, window_len);
            frow = frow_copy;
        }
        int32_t corr = bias ? bias[oc] : 0;
        if (input_offset != 0) {
            corr += esp_nn_filter_sum_s8_esp32s3(frow, window_len)
                    * input_offset;
        }
        const int32_t mult = out_mult[oc];
        const int32_t shift = out_shift[oc];
        int8_t *out = out_data + oc;

        const int8_t *in_row = padded;
        for (int oy = 0; oy < out_ht; oy++) {
            const int8_t *in = in_row;
            for (int ox = 0; ox < out_wd; ox++) {
                esp_nn_accx_zero_s3();
                esp_nn_dot_accx_s3(in, frow, blocks);
                esp_nn_dot_accx_s3(in + padded_row, frow + row_len, blocks);
                esp_nn_dot_accx_s3(in + 2 * padded_row, frow + 2 * row_len,
                                   blocks);
                int32_t acc = esp_nn_accx_read_s3();
                acc += corr;
                acc = esp_nn_requant_s3(acc, mult, shift);
                acc += out_offset;
                acc = max(acc, activation_min);
                acc = min(acc, activation_max);
                *out = (int8_t)acc;
                out += out_channels;
                in += step_x;
            }
            in_row += step_y;
        }
    }
}
