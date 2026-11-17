/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <esp_nn_defs.h>

/* Shapes for esp_nn_conv_s8_oc16_pie: short window rows, whole blocks of
 * 16 output channels, fewer than 64 output channels and at least 256
 * output pixels.
 *
 * Each call first transposes the whole filter. Per output it then costs
 * about 10% more than the 16-pixel batch, and it skips the batch's input
 * staging. So it wins on large maps with few output channels.
 *
 * Measured on P4 rev v3.2, 3757 shapes: 3x3, 9x1, 1x9 and 5x5 windows on
 * 1 to 31 channels, 16 to 1024 output channels, 2 to 4096 pixels, stride
 * 1 and 2, with and without padding, tensors in PSRAM, 128 KB L2.
 * - From 1024 pixels, the kernel takes on average 0.81 of the batch time
 *   at 16 output channels, 0.92 at 32, 0.93 at 48, and is slower from 64.
 * - Below 256 pixels the transpose is not paid back.
 * - With this rule the sweep takes 0.992 of the batch-only time. Turning
 *   the kernel on whenever pixels >= out_ch gives 1.048.
 * - Small maps with a 3 to 7 pixel tail can still run faster on the
 *   kernel, e.g. 3x3 on 3 channels, 6x6 to 64 output channels (0.77).
 *   Adding them back also adds about 200 shapes that run slower.
 *
 * Why the kernel stays: in real models it takes the first 3x3 layers on
 * RGB input and 3x3 on 8 channels, 16 output channels, maps of 2304 to
 * 12544 pixels. P4 rev v3.2, IDF v6.1, arena in PSRAM, model in flash:
 * - those layers run 12% to 26% faster than the batch, at 128, 256 and
 *   512 KB L2, one core and two;
 * - YOLO11n 192x192: 0.3% to 0.4% less total time;
 * - MobileNetV3-Small 224x224: 0.4% to 0.9% less total time;
 * - identical outputs, same arena size.
 * S31, IDF master: the same YOLO11n layers 13% and 25% faster, 0.3% less
 * total time.
 *
 * out_pixels is the map of this call. Each half of the dual-core row
 * split has fewer pixels than the whole map, so it takes the kernel only
 * when the whole map does. The scratch getter sizes for the whole map. */
static inline bool esp_nn_conv_oc16_ok(int32_t filter_wd, int32_t in_ch,
                                       int32_t out_ch, int32_t out_pixels)
{
    const int32_t row_size = filter_wd * in_ch;
    return row_size >= 2 && row_size <= 31 && out_ch % 16 == 0 &&
           out_ch < 64 && out_pixels >= 256;
}

/* Work bytes: the 16 x 16 MAC sums, the requant row and tile, the
 * transposed filter, 16 bytes of load slack, then 16 window buffers for
 * border pixels. The work area is 16-byte aligned. */
#define ESP_NN_CONV_OC16_BUF_BYTES (16 * 16 * 4 + 16 * 4 + 16 * 16)
static inline int32_t esp_nn_conv_oc16_work_size(int32_t window_len,
                                                 int32_t out_ch)
{
    return ESP_NN_CONV_OC16_BUF_BYTES + out_ch * window_len + 16 +
           16 * window_len;
}

/* Words per channel in the requant table of this kernel: the 3 fast
 * requant constants, then e1 = 1 - shift. */
#define ESP_NN_CONV_OC16_TAB_WORDS 4

/* tab: per-channel requant constants, ESP_NN_CONV_OC16_TAB_WORDS words
 * per channel, 16-byte aligned. work: esp_nn_conv_oc16_work_size bytes. */
void esp_nn_conv_s8_oc16_pie(const data_dims_t *input_dims,
                             const int8_t *input_data,
                             const data_dims_t *filter_dims,
                             const int8_t *filter_data,
                             const data_dims_t *output_dims,
                             int8_t *out_data,
                             const conv_params_t *conv_params,
                             const int32_t *tab, int8_t *work);
