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
 * ESP32 (MAC16) 3x3 depthwise conv, any stride, padding and ch_mult.
 * - A band of DW_ROWS input rows and a chunk of channels is widened to s16,
 *   one column-major plane per channel, with the input offset added.
 *   Padding is zero in the planes.
 * - The asm runs the 9 taps of one output channel over the band.
 * - The fused dot requantizes in the asm. Otherwise the requant reuses the
 *   conv asm (esp_nn_conv_requant_store_esp32).
 */

#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include <esp_nn_esp32.h>
#include <common_functions.h>

#include "esp_nn_conv_dispatch_esp32.h"
#include "esp_nn_mac16_esp32.h"

_Static_assert(DW_ROWS == 16, "esp_nn_dw_zero_esp32 zeroes 32-byte columns");

/* Bias limit for the x256 fused dot: |sum + bias| < 2^23. */
#define DW_BIAS256_MAX ((1 << 23) - 9 * 255 * 128)

/* Two taps times 256 as an s16 pair, lo in the low half. */
static inline uint32_t dw_pair256(int8_t lo, int8_t hi)
{
    return ((uint32_t) (uint8_t) lo << 8) | ((uint32_t) (uint8_t) hi << 24);
}

static int16_t *dw_scratch_esp32;

void esp_nn_set_depthwise_conv_scratch_buf_esp32(const void *buf)
{
    dw_scratch_esp32 = (int16_t *) (((uintptr_t) buf + 3) & ~3);
}

/* Scratch layout (esp_nn_depthwise_conv_scratch_size_esp32_impl): the s16
 * planes of a chunk of channels, then the int32 sums of a band. */
static __attribute__((noinline))
void dw_s8_mac16(const data_dims_t *input_dims,
                 const int8_t *input_data,
                 const data_dims_t *filter_dims,
                 const int8_t *filter_data,
                 const int32_t *bias,
                 const data_dims_t *output_dims,
                 int8_t *out_data,
                 const dw_conv_params_t *conv_params,
                 const quant_data_t *quant_data,
                 const dw_layout_t *layout)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t channels = input_dims->channels;
    const int32_t ch_mult = conv_params->ch_mult;
    const int32_t out_ch = channels * ch_mult;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ht = output_dims->height;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;
    /* The x256 fused dot needs |in + in_offset| <= 255, see below. */
    const bool fuse_ok = act_min == -128 && act_max == 127 && conv_params->in_offset >= -127 &&
                         out_offset >= -128 && out_offset <= 127;

    const dw_layout_t l = *layout;
    const int32_t plane_len = l.plane_wd * DW_ROWS;
    int16_t *planes = dw_scratch_esp32;
    int32_t *sums = (int32_t *) (planes + l.chunk * plane_len);

    /* Input columns that land in the planes. */
    const int32_t n_x = max(0, min(l.plane_wd - pad_wd, input_wd));
    const int32_t widen_p[3] = { channels, plane_len * 2, conv_params->in_offset };
    const int widen_u8 = conv_params->in_offset == 128 && (channels & 3) == 0 &&
                         ((uintptr_t) input_data & 3) == 0;

    /*
     * Stride 1 with out_offset -128: the fused dot runs row pairs and shares
     * columns between outputs. Its top rows are always even.
     */
    const int32_t s1_groups = stride_wd == 1 && stride_ht == 1 && out_offset == -128 ?
                              out_wd / 3 : 0;
    int32_t fp[21];
    fp[6] = stride_wd * DW_ROWS * 2;
    fp[7] = stride_ht * 2;
    fp[11] = out_ch;
    fp[18] = s1_groups;
    fp[19] = out_wd - 3 * (out_wd / 3);
    fp[20] = (out_wd - 1) * out_ch;

    for (int32_t oy0 = 0; oy0 < out_ht; oy0 += l.band_rows) {
        const int32_t n_rows = min(l.band_rows, out_ht - oy0);
        const int32_t rows_used = (n_rows - 1) * stride_ht + 3;
        const int32_t iy0 = oy0 * stride_ht - pad_ht;
        /*
         * The widen never writes the pad columns, so they stay zero after the
         * first band. Only bands with pad rows need the memset again.
         */
        const int zero = iy0 < 0 || iy0 + rows_used > input_ht ||
                         (oy0 == 0 && (pad_wd > 0 || n_x < l.plane_wd));

        for (int32_t c0 = 0; c0 < channels; c0 += l.chunk) {
            const int32_t nc = min(l.chunk, channels - c0);
            if (zero) {
                /* The rows past rows_used are never read. */
                esp_nn_dw_zero_esp32(planes, nc * l.plane_wd, rows_used <= DW_ROWS / 2);
            }
            /* Offset 128 and whole words: one load per 4 channels. */
            void (*widen)(int16_t *, const int8_t *, int32_t, int32_t, const int32_t *) =
                widen_u8 && (c0 & 3) == 0 && (nc & 3) == 0 ?
                esp_nn_dw_widen_u8_esp32 : esp_nn_dw_widen_t_esp32;
            for (int32_t r = 0; r < rows_used; r++) {
                const int32_t iy = iy0 + r;
                if (iy >= 0 && iy < input_ht) {
                    widen(planes + pad_wd * DW_ROWS + r,
                          input_data + iy * input_wd * channels + c0, n_x, nc, widen_p);
                }
            }

            int8_t *out_band = out_data + oy0 * out_wd * out_ch;
            const int16_t *plane = planes;
            for (int32_t oc = c0 * ch_mult, m = 0; oc < (c0 + nc) * ch_mult; oc++, m++) {
                if (m == ch_mult) {
                    m = 0;
                    plane += plane_len;
                }
                const int8_t *f = filter_data + oc;
                const int32_t bias_oc = bias ? bias[oc] : 0;
                const int32_t shift = quant_data->shift[oc];
                int8_t *out = out_band + oc;
                /*
                 * The common case: requant fused into the dot, with taps and
                 * bias times 256. With in_offset >= -127, |in + offset| <= 255,
                 * so |sum| <= 9 * 255 * 128 < 2^19, and the bias limit keeps
                 * |sum + bias| < 2^23, see the asm.
                 */
                if (fuse_ok && shift < 0 && shift >= -15 &&
                        bias_oc > -DW_BIAS256_MAX && bias_oc < DW_BIAS256_MAX) {
                    /* Tap pairs times 256, see the asm for the orders. */
                    const uint32_t e0 = dw_pair256(f[0], f[3 * out_ch]);
                    const uint32_t e1 = dw_pair256(f[out_ch], f[4 * out_ch]);
                    const uint32_t e2 = dw_pair256(f[2 * out_ch], f[5 * out_ch]);
                    const uint32_t e3 = dw_pair256(f[6 * out_ch], f[7 * out_ch]);
                    const int32_t e4 = f[8 * out_ch] * 256;
                    fp[0] = e0;
                    fp[1] = e1;
                    fp[2] = e2;
                    fp[3] = e3;
                    fp[4] = e4;
                    if ((stride_ht & 1) && s1_groups == 0) {
                        /* Odd top rows only come with an odd stride. */
                        fp[13] = (e0 >> 16) | (e3 << 16);
                        fp[14] = (e1 >> 16) | (e3 & 0xffff0000u);
                        fp[15] = (e2 >> 16) | ((uint32_t) e4 << 16);
                        fp[16] = (e0 & 0xffff) | (e1 << 16);
                        fp[17] = (int16_t) e2;
                    }
                    fp[5] = bias_oc * 256;
                    fp[8] = quant_data->mult[oc];
                    fp[9] = 7 - shift;
                    fp[10] = 64 + (1 << (6 - shift)) + out_offset * (1 << (7 - shift));
                    /* out_offset -128 needs no sign term, see the asm. */
                    fp[12] = out_offset == -128 ? fp[10] : fp[10] - 128;
                    esp_nn_dw3x3_dot_rq_esp32(plane, out, out_wd, n_rows, fp);
                    continue;
                }
                for (int32_t t = 0; t < 4; t++) {
                    fp[t] = (uint16_t) f[2 * t * out_ch] |
                            ((uint32_t)(uint16_t) f[(2 * t + 1) * out_ch] << 16);
                }
                fp[4] = f[8 * out_ch];
                fp[5] = bias_oc;
                esp_nn_dw3x3_dot_esp32(plane, sums, out_wd, n_rows, fp);

                const int32_t right_shift = shift > 0 ? 0 : -shift;
                const int32_t params[6] = {
                    quant_data->mult[oc], shift > 0 ? shift : 0, right_shift, out_offset,
                    act_min, act_max,
                };
                esp_nn_conv_requant_store_esp32(sums, n_rows * out_wd, out, out_ch, params);
            }
        }
    }
}

void esp_nn_depthwise_conv_s8_esp32(const data_dims_t *input_dims,
                                    const int8_t *input_data,
                                    const data_dims_t *filter_dims,
                                    const int8_t *filter_data,
                                    const int32_t *bias,
                                    const data_dims_t *output_dims,
                                    int8_t *out_data,
                                    const dw_conv_params_t *conv_params,
                                    const quant_data_t *quant_data)
{
    dw_layout_t l;
    if (dw_esp32_layout(input_dims, filter_dims, output_dims, conv_params, &l) &&
            dw_scratch_esp32 &&
            esp32_mac16_data_ok(input_data, filter_data, out_data, dw_scratch_esp32)) {
        dw_s8_mac16(input_dims, input_data, filter_dims, filter_data, bias,
                    output_dims, out_data, conv_params, quant_data, &l);
        return;
    }
    /* Other filters, shapes that do not fit, calls without a scratch buffer
     * and the cases of esp32_mac16_data_ok run the generic kernel. */
    esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims, filter_data,
                                 bias, output_dims, out_data, conv_params, quant_data);
}
