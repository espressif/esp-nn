/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <esp_nn_defs.h>
#include <common_functions.h>

/**
 * Depthwise convolution for s8 using ESP32-P4 PIE SIMD.
 *
 * For ch_mult=1 and channels >= 16: uses QACC per-lane accumulation.
 *   esp.vmulas.s8.qacc accumulates input[i]*filter[i] per-lane across
 *   the spatial filter window. After the window, extract and requantize.
 *
 * Falls back to generic opt for ch_mult > 1 or channels < 16.
 */

/* External fallback */
void esp_nn_depthwise_conv_s8_opt(const data_dims_t *input_dims,
                                   const int8_t *input_data,
                                   const data_dims_t *filter_dims,
                                   const int8_t *filter_data,
                                   const int32_t *bias,
                                   const data_dims_t *output_dims,
                                   int8_t *out_data,
                                   const dw_conv_params_t *conv_params,
                                   const quant_data_t *quant_data);

int esp_nn_get_depthwise_conv_scratch_size_esp32p4(const data_dims_t *input_dims,
                                                    const data_dims_t *filter_dims,
                                                    const data_dims_t *output_dims,
                                                    const dw_conv_params_t *conv_params)
{
    return 0;
}

void esp_nn_set_depthwise_conv_scratch_buf_esp32p4(const void *buf)
{
    (void) buf;
}

/* PIE-optimized ch_mult=1, channels>=16 path */
__attribute__ ((noinline))
static void depthwise_conv_s8_ch1_pie(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Enable PIE */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    const int32_t ch_16 = channels >> 4;

    int out_idx = 0;
    for (int out_y = 0; out_y < out_ht; out_y++) {
        const int16_t base_y = (out_y * stride_ht) - pad_ht;
        for (int out_x = 0; out_x < out_wd; out_x++) {
            const int16_t base_x = (out_x * stride_wd) - pad_wd;

            const int32_t *out_shift = quant_data->shift;
            const int32_t *out_mult = quant_data->mult;

            int filter_y_start = max(0, -base_y);
            int filter_x_start = max(0, -base_x);
            int filter_y_end = min(filter_ht, input_ht - base_y);
            int filter_x_end = min(filter_wd, input_wd - base_x);

            /* Process 16 channels at a time using QACC per-lane accumulation */
            int ch_idx = 0;
            for (int ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_idx += 16) {

                /* Clear QACC per-lane accumulators */
                asm volatile ("esp.zero.qacc \n\t");

                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                    const int32_t idx_y = base_y + fy;
                    for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                        const int32_t idx_x = base_x + fx;
                        const int8_t *ip = input_data + (idx_y * input_wd + idx_x) * channels + ch_idx;
                        const int8_t *fp = filter_data + (fy * filter_wd + fx) * channels + ch_idx;

                        /* Per-lane MAC: qacc[i] += input[i] * filter[i] */
                        asm volatile (
                            "mv              x30, %0      \n\t"
                            "mv              x31, %1      \n\t"
                            "esp.vld.128.ip  q0, x30, 0   \n\t"
                            "esp.vld.128.ip  q1, x31, 0   \n\t"
                            "esp.vmulas.s8.qacc q0, q1    \n\t"
                            :: "r"(ip), "r"(fp)
                            : "x30", "x31"
                        );
                    }
                }

                /* Extract 16 per-lane int32 results from QACC */
                int32_t result[16] __attribute__((aligned(16)));
                asm volatile (
                    "mv                      x30, %0     \n\t"
                    "esp.st.qacc.l.l.128.ip  x30, 16     \n\t"
                    "esp.st.qacc.l.h.128.ip  x30, 16     \n\t"
                    "esp.st.qacc.h.l.128.ip  x30, 16     \n\t"
                    "esp.st.qacc.h.h.128.ip  x30, 0      \n\t"
                    :: "r"(result)
                    : "x30", "memory"
                );

                /* Requantize per channel.
                 * If input_offset != 0: need to add offset * filter_sum.
                 * This is pre-computed per output position (same filter window
                 * for all channels in this block). */
                if (input_offset != 0) {
                    /* Pre-compute filter sums for this block's 16 channels */
                    int32_t fsum[16] = {0};
                    for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                        for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                            const int8_t *fp = filter_data + (fy * filter_wd + fx) * channels + ch_idx;
                            for (int k = 0; k < 16; k++) fsum[k] += fp[k];
                        }
                    }
                    for (int k = 0; k < 16; k++) result[k] += fsum[k] * input_offset;
                }

                for (int k = 0; k < 16; k++) {
                    int32_t r = result[k];
                    if (bias) r += bias[ch_idx + k];
                    r = esp_nn_multiply_by_quantized_mult_fast(r, out_mult[ch_idx + k], out_shift[ch_idx + k]);
                    r += out_offset;
                    r = max(r, activation_min);
                    r = min(r, activation_max);
                    out_data[out_idx++] = (int8_t) r;
                }
            }

            /* Remaining channels (< 16) - scalar */
            for (; ch_idx < channels; ch_idx++) {
                int32_t result = 0;
                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                    const int32_t idx_y = base_y + fy;
                    for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                        const int32_t idx_x = base_x + fx;
                        int32_t input_index = (idx_y * input_wd + idx_x) * channels + ch_idx;
                        int32_t filter_index = (fy * filter_wd + fx) * channels + ch_idx;
                        result += (input_data[input_index] + input_offset) * filter_data[filter_index];
                    }
                }
                if (bias) result += bias[ch_idx];
                result = esp_nn_requantize(result, out_mult[ch_idx], out_shift[ch_idx]);
                result += out_offset;
                result = max(result, activation_min);
                result = min(result, activation_max);
                out_data[out_idx++] = (int8_t) result;
            }
        }
    }
}

void esp_nn_depthwise_conv_s8_esp32p4(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data)
{
    const uint16_t ch_mult = conv_params->ch_mult;
    const uint16_t channels = input_dims->channels;

    /* QACC per-lane path ready but input_offset handling overhead makes it
     * slower than generic opt for typical cases. Enable when input_offset == 0
     * (common in post-training quantized models) or when we pre-compute
     * filter sums once per layer instead of per output position. */
    if (ch_mult == 1 && channels >= 16 && conv_params->in_offset == 0) {
        depthwise_conv_s8_ch1_pie(input_dims, input_data, filter_dims, filter_data,
                                   bias, output_dims, out_data, conv_params, quant_data);
        return;
    }
    esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims, filter_data,
                                  bias, output_dims, out_data, conv_params, quant_data);
}
