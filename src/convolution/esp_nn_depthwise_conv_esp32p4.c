/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <esp_nn_defs.h>
#include <common_functions.h>
#include <stdlib.h>

/* Note: esp_nn_requant_2x_esp32p4.S exists but inline ESP_NN_REQUANT_2X macro
 * from common_functions.h is used instead (avoids function call overhead). */

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

    /* Pre-compute combined offset: filter_sum * input_offset + bias per channel.
     * This fuses two additions per channel into one pre-computed value.
     * Constant for the entire layer - computed once. */
    int32_t combined_offset_buf[256]; /* support up to 256 channels on stack */
    int32_t *combined_offset = NULL;
    if (channels <= 256) {
        combined_offset = combined_offset_buf;
        for (int ch = 0; ch < channels; ch++) {
            int32_t s = 0;
            if (input_offset != 0) {
                for (int fy = 0; fy < filter_ht; fy++) {
                    for (int fx = 0; fx < filter_wd; fx++) {
                        s += filter_data[(fy * filter_wd + fx) * channels + ch];
                    }
                }
                s *= input_offset;
            }
            combined_offset[ch] = s + (bias ? bias[ch] : 0);
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

            /* Process 16 channels at a time using QACC.
             * Inline helper macro for QACC MAC across filter window. */
            #define QACC_MAC_WINDOW(ch_off) do { \
                asm volatile ("esp.zero.qacc \n\t"); \
                for (int _fy = filter_y_start; _fy < filter_y_end; _fy++) { \
                    const int32_t _iy = base_y + _fy; \
                    const int8_t *_ip = input_data + (_iy * input_wd + base_x + filter_x_start) * channels + (ch_off); \
                    const int8_t *_fp = filter_data + (_fy * filter_wd + filter_x_start) * channels + (ch_off); \
                    int _fc = filter_x_end - filter_x_start; \
                    asm volatile ( \
                        "mv     x30, %[ip]               \n\t" \
                        "mv     x31, %[fp]               \n\t" \
                        "mv     s7,  %[cnt]              \n\t" \
                        "1:                              \n\t" \
                        "esp.vld.128.ip  q0, x30, 0      \n\t" \
                        "esp.vld.128.ip  q1, x31, 0      \n\t" \
                        "esp.vmulas.s8.qacc q0, q1       \n\t" \
                        "add    x30, x30, %[stride]      \n\t" \
                        "add    x31, x31, %[stride]      \n\t" \
                        "addi   s7, s7, -1               \n\t" \
                        "bnez   s7, 1b                   \n\t" \
                        : \
                        : [ip] "r"(_ip), [fp] "r"(_fp), \
                          [cnt] "r"(_fc), [stride] "r"((int32_t)channels) \
                        : "x30", "x31", "s7" \
                    ); \
                } \
            } while(0)

            #define QACC_EXTRACT(dst) do { \
                asm volatile ( \
                    "mv                      x30, %0     \n\t" \
                    "esp.st.qacc.l.l.128.ip  x30, 16     \n\t" \
                    "esp.st.qacc.l.h.128.ip  x30, 16     \n\t" \
                    "esp.st.qacc.h.l.128.ip  x30, 16     \n\t" \
                    "esp.st.qacc.h.h.128.ip  x30, 0      \n\t" \
                    :: "r"(dst) \
                    : "x30", "memory" \
                ); \
            } while(0)

            int ch_idx = 0;
            for (int ch_blk = 0; ch_blk < ch_16; ch_blk++, ch_idx += 16) {
                QACC_MAC_WINDOW(ch_idx);

                /* Extract 16 per-lane results */
                int32_t result[16] __attribute__((aligned(16)));
                QACC_EXTRACT(result);

                /* Add fused offset (filter_sum * input_offset + bias) + requantize */
                if (combined_offset) {
                    if (is_full_window) {
                        /* Fast: use pre-computed combined offset (fused filter_sum + bias) */
                        for (int k = 0; k < 16; k++) {
                            result[k] += combined_offset[ch_idx + k];
                        }
                    } else {
                        /* Edge: compute partial filter sum + add bias */
                        for (int k = 0; k < 16; k++) {
                            int32_t fsum = 0;
                            if (input_offset != 0) {
                                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                                    for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                                        fsum += filter_data[(fy * filter_wd + fx) * channels + ch_idx + k];
                                    }
                                }
                                fsum *= input_offset;
                            }
                            result[k] += fsum + (bias ? bias[ch_idx + k] : 0);
                        }
                    }
                }

                /* Per-channel requantize: 2-wide interleaved inline */
                {
                    const int32_t *mp = out_mult + ch_idx;
                    const int32_t *sp = out_shift + ch_idx;

                    for (int k = 0; k < 16; k += 2) {
                        int32_t r0 = result[k]; int32_t r1 = result[k+1];

                        int32_t m0 = mp[k], s0 = sp[k];
                        int32_t m1 = mp[k+1], s1 = sp[k+1];

                        /* 2-wide interleaved requant via inline asm macro.
                         * Macro handles left_shift internally - do NOT pre-shift. */
                        int32_t h0, h1;
                        ESP_NN_REQUANT_2X(r0, r1, m0, m1, s0, s1, h0, h1);

                        h0 += out_offset; h1 += out_offset;
                        out_data[out_idx++] = (int8_t)max(activation_min, min(h0, activation_max));
                        out_data[out_idx++] = (int8_t)max(activation_min, min(h1, activation_max));
                    }
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
