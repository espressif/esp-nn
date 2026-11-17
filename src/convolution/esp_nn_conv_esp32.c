// Copyright 2026 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * ESP32 (MAC16) conv for any filter size, stride and padding, not grouped.
 * - The receptive field of a band of output pixels is gathered into s16
 *   scratch rows (im2col), with the input offset added and zeros for padding.
 * - Each filter row is widened once per band and run against every row of it.
 */

#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include <esp_nn_esp32.h>
#include <common_functions.h>

#include "esp_nn_conv_dispatch_esp32.h"
#include "esp_nn_mac16_esp32.h"

static int16_t *scratch_buf_esp32;

void esp_nn_set_conv_scratch_buf_esp32(const void *buf)
{
    scratch_buf_esp32 = (int16_t *) (((uintptr_t) buf + 3) & ~3);
}

/* Scratch layout (esp_nn_conv_scratch_size_esp32_impl): the s16 input band,
 * the s16 filter row, then the int32 sums of the band. */
static __attribute__((noinline))
void conv_s8_mac16(const data_dims_t *input_dims,
                   const int8_t *input_data,
                   const data_dims_t *filter_dims,
                   const int8_t *filter_data,
                   const int32_t *bias,
                   const data_dims_t *output_dims,
                   int8_t *out_data,
                   const conv_params_t *conv_params,
                   const quant_data_t *quant_data)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t in_ch = input_dims->channels;
    const int32_t filter_wd = filter_dims->width;
    const int32_t filter_ht = filter_dims->height;
    const int32_t filter_len = filter_wd * filter_ht * in_ch;
    const int32_t row_len = conv_row_len(filter_dims);
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ch = output_dims->channels;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;
    const int32_t n_pix = out_wd * output_dims->height;
    const int32_t band = conv_band_pixels(row_len, n_pix);
    const int32_t contiguous = filter_len == row_len && filter_wd == 1 &&
                               filter_ht == 1 && stride_wd == 1 &&
                               pad_wd == 0 && pad_ht == 0;

    /* For the fused requant range. With input_offset >= -127,
     * |(in + offset) * filter| <= 255 * 128 < 2^15. The bias limits below
     * use the same 255. */
    const int fused_ok = act_min == -128 && act_max == 127 && filter_len <= (1 << 13) &&
                         input_offset >= -127 &&
                         out_offset >= -128 && out_offset <= 127;
    /*
     * The x256 filter keeps 256 * |sum + bias| < 2^31: |bias| < bias256_max.
     * Longer rows use the 40-bit ACC (esp_nn_conv_dot_rq256w_mac16_esp32) with
     * the range of the rq dot: shift in [-20, -1] and |bias| <= 2^28. That dot
     * clamps to any activation range.
     */
    const int32_t bias256_max = (1 << 23) - filter_len * 255 * 128;
    const int wide = bias256_max <= 0;
    const int32_t shift_min = wide ? -20 : -15;
    const int32_t bias_lim = wide ? (1 << 28) + 1 : bias256_max;

    /*
     * in_ch a multiple of 4 and every channel in the x256 range: the filter is
     * widened with masks (esp_nn_widen_lanes_esp32), so its samples come in
     * the order [0, 2, 1, 3]. The input band uses the same order and starts
     * one sample in, see esp_nn_conv_rq256s_oc_esp32.
     */
    int lanes = (in_ch & 3) == 0 && ((uintptr_t) filter_data & 3) == 0 &&
                filter_len <= (1 << 13) && out_offset >= -128 && out_offset <= 127 &&
                (fused_ok || wide);
    for (int32_t oc = 0; lanes && oc < out_ch; oc++) {
        const int32_t b = bias ? bias[oc] : 0;
        const int32_t shift = quant_data->shift[oc];
        lanes = shift < 0 && shift >= shift_min && b > -bias_lim && b < bias_lim;
    }
    void (*widen)(int16_t *, const int8_t *, int32_t, int32_t) =
        !lanes ? esp_nn_widen_s8_s16_esp32 :
        input_offset == 128 && ((uintptr_t) input_data & 3) == 0 ? esp_nn_widen_u8_esp32 :
        esp_nn_widen_perm_esp32;

    int16_t *in_s16 = scratch_buf_esp32;
    /*
     * filter row: one zero sample in front, see the asm. The lanes path
     * starts the row on that sample and stores 4 samples after it instead.
     */
    int16_t *filt_s16 = in_s16 + band * row_len + 4;
    int32_t *sums = (int32_t *) (filt_s16 + row_len + 4);

    /* The pad samples stay zero in the filter row, so the input pad is don't-care. */
    filt_s16[0] = 0;
    filt_s16++;
    for (int32_t i = filter_len; i < row_len; i++) {
        filt_s16[i] = 0;
    }

    int32_t out_x = 0, out_y = 0;
    for (int32_t p0 = 0; p0 < n_pix; p0 += band) {
        const int32_t n = min(band, n_pix - p0);

        /* Gather the receptive field of each pixel. Padding contributes zero. */
        int16_t *dst = in_s16 + lanes;
        for (int32_t p = 0; p < n;) {
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int32_t base_x = out_x * stride_wd - pad_wd;
            if (contiguous) {
                /* 1x1 stride 1: a run of pixels in one row is one block. */
                const int32_t run = min(n - p, out_wd - out_x);
                widen(dst, input_data +
                        (base_y * input_wd + base_x) * in_ch, run * in_ch, input_offset);
                dst += run * row_len;
                p += run;
                out_x += run;
            } else {
                const int32_t fx_start = max(0, -base_x);
                const int32_t fx_end = min(filter_wd, input_wd - base_x);
                const int32_t lead = fx_start * in_ch;
                const int32_t len = (fx_end - fx_start) * in_ch;
                const int32_t trail = filter_wd * in_ch - lead - len;
                int16_t *row = dst;
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int32_t iy = base_y + fy;
                    if (iy < 0 || iy >= input_ht) {
                        memset(row, 0, filter_wd * in_ch * 2);
                    } else {
                        if (lead) {
                            memset(row, 0, lead * 2);
                        }
                        widen(row + lead, input_data +
                                (iy * input_wd + base_x + fx_start) * in_ch, len, input_offset);
                        if (trail) {
                            memset(row + lead + len, 0, trail * 2);
                        }
                    }
                    row += filter_wd * in_ch;
                }
                dst += row_len;
                p++;
                out_x++;
            }
            if (out_x == out_wd) {
                out_x = 0;
                out_y++;
            }
        }

        const int8_t *filt = filter_data;
        int8_t *out_base = out_data + p0 * out_ch;
        if (lanes && !wide) {
            /* The asm loops over the output channels. */
            static const int32_t zero_bias = 0;
            const int32_t q[7] = {
                (int32_t) filter_data, (int32_t) (bias ? bias : &zero_bias), bias ? 4 : 0,
                (int32_t) quant_data->mult, (int32_t) quant_data->shift, out_ch, out_offset,
            };
            esp_nn_conv_rq256s_oc_esp32(filt_s16 - 1, in_s16, out_base, n, row_len >> 2, q);
            continue;
        }
        if (lanes) {
            /* Rows past 256 samples (wide). */
            for (int32_t oc = 0; oc < out_ch; oc++) {
                const int32_t b = bias ? bias[oc] : 0;
                const int32_t shift = quant_data->shift[oc];
                int32_t p[8];
                p[0] = (int32_t) ((uint32_t) b << 8);
                p[2] = 1 - shift;
                p[3] = 1 + (1 << -shift) + out_offset * (1 << (1 - shift));
                p[5] = b >> 24;
                p[1] = quant_data->mult[oc];
                p[4] = out_ch;
                p[6] = act_min;
                p[7] = act_max;
                uint32_t *fw = (uint32_t *) (filt_s16 - 1);
                esp_nn_widen_lanes_esp32((int16_t *) fw, filt, filter_len >> 2);
                /* The first 4 samples again after the row, see the asm. */
                fw[row_len >> 1] = fw[0];
                fw[(row_len >> 1) + 1] = fw[1];
                filt += filter_len;
                esp_nn_conv_dot_rq256w_mac16_esp32(filt_s16 - 1, in_s16, out_base + oc, n,
                                                   row_len >> 2, p);
            }
            continue;
        }
        for (int32_t oc = 0; oc < out_ch; oc++) {
            const int32_t bias_oc = bias ? bias[oc] : 0;
            const int32_t shift = quant_data->shift[oc];
            esp_nn_widen_s8_s16_esp32(filt_s16, filt, filter_len, 0);
            filt += filter_len;
            /*
             * The common case: requant fused into the dot. |sum| < 2^28 and
             * |bias| <= 2^28 keep the sum in its exact range, see the asm.
             */
            if (fused_ok && shift < 0 && shift >= -20 &&
                    bias_oc >= -(1 << 28) && bias_oc <= (1 << 28)) {
                const int32_t p[5] = {
                    bias_oc, quant_data->mult[oc], 1 - shift,
                    1 + (1 << -shift) + out_offset * (1 << (1 - shift)), out_ch,
                };
                esp_nn_conv_dot_rq_mac16_esp32(filt_s16 - 1, in_s16, out_base + oc, n,
                                               row_len >> 2, p);
                continue;
            }

            esp_nn_conv_dot_mac16_esp32(filt_s16 - 1, in_s16, sums, n, row_len >> 2,
                                        bias_oc);

            const int32_t right_shift = shift > 0 ? 0 : -shift;
            const int32_t params[6] = {
                quant_data->mult[oc], shift > 0 ? shift : 0, right_shift, out_offset,
                act_min, act_max,
            };
            esp_nn_conv_requant_store_esp32(sums, n, out_base + oc, out_ch, params);
        }
    }
}

void esp_nn_conv_s8_esp32(const data_dims_t *input_dims,
                          const int8_t *input_data,
                          const data_dims_t *filter_dims,
                          const int8_t *filter_data,
                          const int32_t *bias,
                          const data_dims_t *output_dims,
                          int8_t *out_data,
                          const conv_params_t *conv_params,
                          const quant_data_t *quant_data)
{
    if (conv_esp32_supported(input_dims, filter_dims, output_dims, conv_params) &&
            scratch_buf_esp32 &&
            esp32_mac16_data_ok(input_data, filter_data, out_data, scratch_buf_esp32) &&
            esp32_mac16_quant_ok(bias, quant_data->mult, quant_data->shift)) {
        conv_s8_mac16(input_dims, input_data, filter_dims, filter_data, bias,
                      output_dims, out_data, conv_params, quant_data);
        return;
    }
    /* The generic 1x1 kernel also reads input for every output, so windows
     * without input run the reference. */
    if (!conv_esp32_windows_in(input_dims, filter_dims, output_dims, conv_params)) {
        esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data, bias,
                            output_dims, out_data, conv_params, quant_data);
        return;
    }
    /* The other shapes conv_esp32_supported refuses, calls without a
     * scratch buffer and the cases of esp32_mac16_data_ok and
     * esp32_mac16_quant_ok run the generic
     * kernel. */
    esp_nn_conv_s8_opt(input_dims, input_data, filter_dims, filter_data, bias,
                       output_dims, out_data, conv_params, quant_data);
}
