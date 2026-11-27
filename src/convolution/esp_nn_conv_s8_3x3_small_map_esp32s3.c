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

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <esp_nn_multicore.h>
#include "esp_nn_conv_1x1_panel_esp32s3.h"
#include "esp_nn_conv_dispatch_esp32s3.h"

#if ESP_NN_DUAL_CORE_SUPPORTED
/* The worker's output channels. The claim (esp_nn_dual_core_begin()) makes
 * this slot single-owner, and keeps the job off the caller's stack. */
typedef struct {
    const int8_t *padded;
    int32_t padded_row;
    int in_ch;
    int32_t step_x, step_y;
    int out_wd, out_ht;
    const int8_t *filter;
    const int32_t *bias;
    int32_t in_offset;
    int8_t *out_data;
    int out_ch, out_stride;
    int32_t out_offset;
    const int32_t *out_mult, *out_shift;
    int32_t act_min, act_max;
} small_map_mt_job_t;

static small_map_mt_job_t s_small_map_job;

static void small_map_mt_job(void *p)
{
    const small_map_mt_job_t *j = (const small_map_mt_job_t *)p;
    esp_nn_conv_s8_3x3_small_map_accx(j->padded, j->padded_row, j->in_ch,
                                      j->step_x, j->step_y, j->out_wd,
                                      j->out_ht, j->filter, j->bias,
                                      j->in_offset, j->out_data, j->out_ch,
                                      j->out_stride, j->out_offset,
                                      j->out_shift, j->out_mult,
                                      j->act_min, j->act_max);
}
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */

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
    const int32_t step_x = stride_wd * in_channels;
    const int32_t step_y = stride_ht * padded_row;
#if ESP_NN_DUAL_CORE_SUPPORTED
    /* Two cores split the output channels. Both read the padded map. */
    if (out_channels >= 16 && esp_nn_dual_core_begin()) {
        bool done = false;
        const int oc0 = (out_channels / 2) & ~7;
        s_small_map_job = (small_map_mt_job_t) {
            .padded = padded, .padded_row = padded_row, .in_ch = in_channels,
            .step_x = step_x, .step_y = step_y,
            .out_wd = out_wd, .out_ht = out_ht, .filter = filter_data,
            .bias = bias, .in_offset = input_offset, .out_data = out_data,
            .out_ch = oc0, .out_stride = out_channels,
            .out_offset = out_offset, .out_mult = out_mult,
            .out_shift = out_shift, .act_min = activation_min,
            .act_max = activation_max,
        };
        if (esp_nn_dual_core_run(small_map_mt_job, &s_small_map_job)) {
            esp_nn_conv_s8_3x3_small_map_accx(padded, padded_row, in_channels,
                    step_x, step_y, out_wd, out_ht,
                    filter_data + oc0 * 9 * in_channels,
                    bias ? bias + oc0 : NULL, input_offset, out_data + oc0,
                    out_channels - oc0, out_channels, out_offset,
                    out_shift + oc0, out_mult + oc0,
                    activation_min, activation_max);
            esp_nn_dual_core_wait();
            done = true;
        }
        esp_nn_dual_core_end();
        if (done) {
            return;
        }
    }
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
    esp_nn_conv_s8_3x3_small_map_accx(padded, padded_row, in_channels,
                                      step_x, step_y, out_wd, out_ht,
                                      filter_data, bias, input_offset,
                                      out_data, out_channels, out_channels,
                                      out_offset, out_shift, out_mult,
                                      activation_min, activation_max);
}
