/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <esp_nn_defs.h>
#include <common_functions.h>
#include <stdlib.h>

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

/* PIE-optimized ch_mult=1, channels>=16 path using QACC per-lane MAC.
 * Pre-computes filter_sum[ch] = sum of filter[ch] across all filter positions.
 * For non-edge output positions: result[ch] = QACC_MAC + filter_sum[ch] * input_offset
 * For edge positions: falls back to scalar with input_offset applied directly. */
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

    /* Set up activation min/max vectors for PIE clamp */
    {
        int8_t act_min_val = (int8_t) activation_min;
        int8_t act_max_val = (int8_t) activation_max;
        asm volatile (
            "mv     x30, %0             \n\t"
            "esp.vldbc.8.ip q4, x30, 0  \n\t"
            "mv     x30, %1             \n\t"
            "esp.vldbc.8.ip q5, x30, 0  \n\t"
            :: "r"(&act_min_val), "r"(&act_max_val)
            : "x30"
        );
    }

    /* Pre-compute full filter sums per channel on stack.
     * filter_sum[ch] = sum(filter[fy][fx][ch]) * input_offset for all fy,fx.
     * This is constant for the entire layer - computed once. */
    int32_t filter_sum_buf[256]; /* support up to 256 channels on stack */
    int32_t *filter_sum = NULL;
    if (input_offset != 0 && channels <= 256) {
        filter_sum = filter_sum_buf;
        for (int ch = 0; ch < channels; ch++) {
            int32_t s = 0;
            for (int fy = 0; fy < filter_ht; fy++) {
                for (int fx = 0; fx < filter_wd; fx++) {
                    s += filter_data[(fy * filter_wd + fx) * channels + ch];
                }
            }
            filter_sum[ch] = s * input_offset;
        }
    }

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

            /* Check if this is a non-edge position (full filter window) */
            int is_full_window = (filter_y_start == 0 && filter_x_start == 0 &&
                                  filter_y_end == filter_ht && filter_x_end == filter_wd);

            /* Process 16 channels at a time using QACC */
            int ch_idx = 0;
            for (int ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_idx += 16) {
                asm volatile ("esp.zero.qacc \n\t");

                /* Accumulate across filter window using QACC per-lane MAC.
                 * Minimize overhead: pre-compute row pointers, use stride for fx. */
                const int32_t ch_stride = channels;
                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                    const int32_t idx_y = base_y + fy;
                    const int8_t *ip_row = input_data + (idx_y * input_wd + base_x + filter_x_start) * channels + ch_idx;
                    const int8_t *fp_row = filter_data + (fy * filter_wd + filter_x_start) * channels + ch_idx;
                    int fx_count = filter_x_end - filter_x_start;

                    /* Use register-based pointer advance for inner fx loop */
                    asm volatile (
                        "mv     x30, %[ip]               \n\t"
                        "mv     x31, %[fp]               \n\t"
                        "mv     s7,  %[cnt]              \n\t"
                        "1:                              \n\t"
                        "esp.vld.128.ip  q0, x30, 0      \n\t"
                        "esp.vld.128.ip  q1, x31, 0      \n\t"
                        "esp.vmulas.s8.qacc q0, q1       \n\t"
                        "add    x30, x30, %[stride]      \n\t"
                        "add    x31, x31, %[stride]      \n\t"
                        "addi   s7, s7, -1               \n\t"
                        "bnez   s7, 1b                   \n\t"
                        :
                        : [ip] "r"(ip_row), [fp] "r"(fp_row),
                          [cnt] "r"(fx_count), [stride] "r"(ch_stride)
                        : "x30", "x31", "s7"
                    );
                }

                /* Extract 16 per-lane results */
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

                /* Add pre-computed offset term + bias + requantize */
                if (input_offset != 0 && filter_sum) {
                    if (is_full_window) {
                        /* Fast: use pre-computed full filter sums */
                        for (int k = 0; k < 16; k++) {
                            result[k] += filter_sum[ch_idx + k];
                        }
                    } else {
                        /* Edge: compute partial filter sum for clipped window */
                        for (int k = 0; k < 16; k++) {
                            int32_t fsum = 0;
                            for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                                for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                                    fsum += filter_data[(fy * filter_wd + fx) * channels + ch_idx + k];
                                }
                            }
                            result[k] += fsum * input_offset;
                        }
                    }
                }

                /* Per-channel requantize using inline fast path */
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

            /* Remaining channels < 16 */
            for (; ch_idx < channels; ch_idx++) {
                int32_t result = 0;
                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                    const int32_t idx_y = base_y + fy;
                    for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                        const int32_t idx_x = base_x + fx;
                        result += (input_data[(idx_y * input_wd + idx_x) * channels + ch_idx] + input_offset)
                                  * filter_data[(fy * filter_wd + fx) * channels + ch_idx];
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

    if (ch_mult == 1 && channels >= 16) {
        depthwise_conv_s8_ch1_pie(input_dims, input_data, filter_dims, filter_data,
                                   bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    /* Fall back to generic optimized */
    esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims, filter_data,
                                  bias, output_dims, out_data, conv_params, quant_data);
}
