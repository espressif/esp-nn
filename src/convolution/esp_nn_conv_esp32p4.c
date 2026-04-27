/*
 * SPDX-FileCopyrightText: 2024-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_esp32p4 function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_esp32p4)
 *      - For all other cases:
 *          > Consider `filter_wd * in_ch` as a single row. These many values can
 *              be continuosly loaded from inputs as well.
 *          > multiply accumulate into a single filter output.
 *          > To speed things up further, we pre-calculate
 *              (filter * in_offset + bias term) earlier and add it at the end of filter
 *
 *      About ((filter * in_offset + bias term)) accumulate term:
 *          > The conv operation before requantization is as follows:
 *              for i in filter_size:
 *                  conv_out += (input + input_offset) * filter;
 *               conv_out += bias
 *
 *          > where input_offset is constant term hence, we can see that
 *              this term can be precalculated as:
 *                  for i in filter_size:
 *                      acc_term += input_offset * filter[i];
 *                  acc_term += bias
 *              OR
 *                   for i in filter_size:
 *                      acc_term += filter[i]; // accumulate filter values
 *                  acc_term = acc_term * input_offset + bias
 *
 *
 * In both the above versions we align the filter if needed, pad the input with
 *       -input_offset if needed and extend the channels to make those multiple
 *       of 8/16 as per function needs
 */

#include <stdio.h>
#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include "esp_nn_generic_opt.h"

#include <common_functions.h>

static int16_t *scratch_buffer = NULL;

/**
 * Reusable PIE-accelerated dot product (same as FC version).
 * Processes 32 elements/iter (double-pump) for len >= 32,
 * 16 elements/iter for len >= 16, scalar remainder.
 */
static inline __attribute__((always_inline))
int32_t pie_dot_s8(const int8_t *a, const int8_t *b, int32_t len)
{
    int32_t result = 0;
    int32_t idx = 0;

    if (len >= 32) {
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "li     %[idx], 32                      \n\t"
            "addi   s7, %[len], -31                 \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "j      2f                              \n\t"
            "1:                                     \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "addi   %[idx], %[idx], 32              \n\t"
            "2:                                     \n\t"
            "blt    %[idx], s7, 1b                  \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"
            "addi   s7, %[len], -15                 \n\t"
            "bge    %[idx], s7, 3f                  \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "addi   %[idx], %[idx], 16              \n\t"
            "3:                                     \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [idx] "+r"(idx), [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b), [len] "r"(len)
            : "x30", "x31", "s7"
        );
    } else if (len >= 16) {
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "li     %[idx], 16                      \n\t"
            "addi   s7, %[len], -15                 \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "j      5f                              \n\t"
            "4:                                     \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "addi   %[idx], %[idx], 16              \n\t"
            "5:                                     \n\t"
            "blt    %[idx], s7, 4b                  \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [idx] "+r"(idx), [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b), [len] "r"(len)
            : "x30", "x31", "s7"
        );
    }

    for (; idx < len; idx++) {
        result += (int32_t)a[idx] * (int32_t)b[idx];
    }
    return result;
}

/**
 * Batched 1x1 conv using QACC per-lane: processes 16 pixels simultaneously.
 * Transposes input so each QACC lane = one pixel, then broadcasts filter
 * coefficients for per-lane accumulation. Critical for small in_ch where
 * XACC can't be used (in_ch < 16).
 *
 * For in_ch=8: 4.5x faster than scalar per-pixel approach.
 */
__attribute__((noinline))
static void conv_1x1_batch16(const int8_t *pixel_ptrs[16],
                      const int8_t *filter_data,
                      const int32_t *filter_sum,
                      const int32_t *bias,
                      int8_t *out_ptrs[16],
                      int32_t in_ch, int32_t out_ch,
                      int32_t out_offset,
                      const int32_t *out_mult, const int32_t *out_shift,
                      int32_t act_min, int32_t act_max)
{
    /* Ensure PIE is enabled (might be lost across noinline function call) */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    /* Transpose: arrange 16 pixels' data as ch0[p0..p15], ch1[p0..p15], ... */
    int8_t transposed[16 * 16] __attribute__((aligned(16)));  /* in_ch <= 16 for this path */
    for (int c = 0; c < in_ch; c++) {
        for (int p = 0; p < 16; p++) {
            transposed[c * 16 + p] = pixel_ptrs[p][c];
        }
    }

    /* For each output channel: QACC per-lane MAC with broadcast filter.
     * Use single asm block for zero + accumulate loop to prevent
     * q register clobber between separate asm blocks. */
    const int8_t *filt = filter_data;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        /* Single asm: zero QACC, then loop over in_ch channels:
         * broadcast filter[ch], load 16 transposed pixels, MAC per-lane */
        asm volatile (
            "esp.zero.qacc                       \n\t"
            "mv     x30, %[trans]                \n\t"  /* transposed base */
            "mv     x31, %[flt]                  \n\t"  /* filter base */
            "mv     s7,  %[cnt]                  \n\t"  /* in_ch count */
            "1:                                  \n\t"
            "esp.vld.128.ip  q0, x30, 16         \n\t"  /* load 16 pixel values, advance by 16 */
            "esp.vldbc.8.ip  q1, x31, 1          \n\t"  /* broadcast filter[ch], advance by 1 */
            "esp.vmulas.s8.qacc q0, q1           \n\t"
            "addi   s7, s7, -1                   \n\t"
            "bnez   s7, 1b                       \n\t"
            :
            : [trans] "r"(transposed), [flt] "r"(filt), [cnt] "r"(in_ch)
            : "x30", "x31", "s7"
        );

        /* Extract 16 results */
        int32_t results[16] __attribute__((aligned(16)));
        ESP_NN_QACC_EXTRACT_S32(results);

        /* Add filter_sum + bias, requant, clamp, store for each pixel */
        int32_t fs = filter_sum[oc];
        int32_t b = bias ? bias[oc] : 0;
        int32_t combined = fs + b;
        int32_t m = out_mult[oc];
        int32_t s = out_shift[oc];

        for (int p = 0; p < 16; p++) {
            int32_t r = results[p] + combined;
            r = esp_nn_multiply_by_quantized_mult(r, m, s);
            r += out_offset;
            r = max(r, act_min);
            r = min(r, act_max);
            out_ptrs[p][oc] = (int8_t) r;
        }

        filt += in_ch;
    }
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_1x1(const data_dims_t *input_dims,
                               const int8_t *input_data,
                               const int8_t *filter_data,
                               const int32_t *bias,
                               const data_dims_t *output_dims,
                               int8_t *out_data,
                               const conv_params_t *conv_params,
                               const quant_data_t *quant_data,
                               void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    const int8_t *filter_ptr = filter_data;
    for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
        int32_t sum = 0;
        int32_t in_ch_idx = 0;
        for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
            sum += *filter_ptr++;
            sum += *filter_ptr++;
            sum += *filter_ptr++;
            sum += *filter_ptr++;
        }
        for (; in_ch_idx < in_channels; in_ch_idx ++) {
            sum += *filter_ptr++;
        }
        filter_sum[out_ch_idx] = sum * input_offset;
    }

    /* When in_ch < 16: use QACC batch path (16 pixels at once) or channel padding.
     * QACC batch: transpose pixels, broadcast filter, per-lane MAC.
     * Channel pad: pad in/filter to 16 ch for XACC. */
    /* When in_ch < 16: use QACC batch (16 pixels at a time with broadcast filter).
     * Falls back to channel-padding for remaining pixels. */
    if (in_channels < 16) {
        /* Enable PIE for QACC */
        asm volatile (
            "csrsi  0x7f2, 0b01        \n\t"
            "li     x29, 0b10          \n\t"
            "esp.movx.w.cfg x29        \n\t"
            ::: "x29"
        );

        int32_t total_pixels = out_wd * out_ht;
        int32_t pix = 0;

        /* Process batches of 16 pixels using QACC per-lane */
        for (; pix <= total_pixels - 16; pix += 16) {
            const int8_t *pp[16];
            int8_t *op[16];
            for (int p = 0; p < 16; p++) {
                pp[p] = input_data + (pix + p) * in_channels;
                op[p] = out_data + (pix + p) * out_channels;
            }
            conv_1x1_batch16(pp, filter_data, filter_sum, bias, op,
                             in_channels, out_channels, out_offset,
                             quant_data->mult, quant_data->shift,
                             activation_min, activation_max);
        }

        /* Remaining pixels (< 16): scalar fallback */
        for (; pix < total_pixels; pix++) {
            const int8_t *inp = input_data + pix * in_channels;
            filter_ptr = filter_data;
            for (int32_t oc = 0; oc < out_channels; oc++) {
                int32_t conv_out = 0;
                for (int32_t ic = 0; ic < in_channels; ic++) {
                    conv_out += inp[ic] * filter_ptr[ic];
                }
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_multiply_by_quantized_mult(conv_out,
                    quant_data->mult[oc], quant_data->shift[oc]);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                out_data[pix * out_channels + oc] = (int8_t) conv_out;
                filter_ptr += in_channels;
            }
        }
        return;
    }

    for (int32_t in_row = 0; in_row < out_ht; in_row++) {
        for (int32_t in_col = 0; in_col < out_wd; in_col++) {
            const int32_t *out_mult = quant_data->mult;
            const int32_t *out_shift = quant_data->shift;
            filter_ptr = filter_data;
            const int8_t *input_base_ptr = input_data + (in_row * input_wd + in_col) * in_channels;
            for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                /* initializations */
                int32_t conv_out = 0;
                const int8_t *input_ptr = input_base_ptr;

                int32_t in_ch_idx = 0;
#if 1 // inline asm
                // for now check for the alignment as well
                if (in_channels < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                    goto skip_asm;
                }

                asm volatile (
                    "li %0, 16                      \n\t"
                    "addi s7, %4, -15               \n\t"
                    "mv x30, %1                     \n\t"
                    "mv x31, %2                     \n\t"
                    "esp.zero.xacc                  \n\t"
                    "esp.vld.128.ip  q0, x30, 16    \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"

                    "j .loop16_end  \n\t"

                    ".loop16_start:      \n\t"
                    "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                    "esp.vld.128.ip  q1, x31, 16                     \n\t"
                    "addi %0, %0, 16                \n\t"   // in_ch_idx += 16

                    ".loop16_end:    \n\t"
                    "blt %0, s7, .loop16_start \n\t"  // if in_ch_idx < `in_channels - 15` abort

                    // move input_ptr, filter_ptr and conv_out
                    "mv %1, x30                     \n\t"
                    "mv %2, x31                     \n\t"
                    "esp.vmulas.s8.xacc  q0, q1     \n\t"
                    "esp.movx.r.xacc.l  %3          \n\t"

                    : "+r" (in_ch_idx), "+r" (input_ptr), "+r" (filter_ptr), "=r" (conv_out)
                    :  "r"(in_channels)
                    : "x30", "x31", "s7"
                );
skip_asm:
#endif
                for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                }

                for (; in_ch_idx < in_channels; in_ch_idx++) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                }
                conv_out = conv_out + filter_sum[out_ch_idx];
                if (bias) {
                    conv_out += bias[out_ch_idx];
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t) conv_out;
            }
        }
    }
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_padded(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Grouped conv (filter_ch < input_ch): fall back to ansi which handles it */
    if (in_channels != filter_dims->channels) {
        esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data,
                            bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    const int8_t *filter_ptr = filter_data;
    for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
        int32_t sum = 0;
        int32_t filter_len = filter_wd * filter_ht * in_channels;
        int32_t filter_idx = 0;
        for (; filter_idx < filter_len - 3; filter_idx += 4) {
            sum += *filter_ptr++;
            sum += *filter_ptr++;
            sum += *filter_ptr++;
            sum += *filter_ptr++;
        }
        for (; filter_idx < filter_len; filter_idx++) {
            sum += *filter_ptr++;
        }
        filter_sum[out_ch_idx] = sum * input_offset;
    }

    const int32_t row_size = filter_wd * in_channels;

    bool right_pad = max(0, ((out_wd - 1) * stride_wd + filter_wd - input_wd));
    bool bottom_pad = max(0, ((out_ht - 1) * stride_ht + filter_ht - input_ht));

    for (int32_t out_y = 0; out_y < out_ht - bottom_pad; out_y++) {
        for (int32_t out_x = 0; out_x < out_wd - right_pad; out_x++) {
            const int32_t base_y = stride_ht * out_y;
            const int32_t base_x = stride_wd * out_x;
            const int32_t *out_mult_ptr = out_mult;
            const int32_t *out_shift_ptr = out_shift;
            const int32_t *bias_ptr = bias;
            const int8_t *filter_data_ptr = filter_data;
            for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0, filter_y_idx;
                if (row_size >= 16) {
                    asm volatile("esp.zero.xacc                  \n\t");
                }

                for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                    const int32_t in_row = base_y + filter_y_idx;
                    const int32_t in_col = base_x;
                    const int8_t *input_data_ptr =
                            input_data + (in_row * input_wd + in_col) * in_channels;
                    int32_t row_idx = 0;
#if 1 // inline asm
                // for now check for the alignment as well
                if (row_size < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                    goto skip_asm_pad0;
                }

                asm volatile (
                    "li %0, 16                      \n\t"
                    "addi s7, %3, -15               \n\t"
                    "mv x30, %1                     \n\t"
                    "mv x31, %2                     \n\t"
                    "esp.vld.128.ip  q0, x30, 16    \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"

                    "j .loop16_pad0_end  \n\t"

                    ".loop16_pad0_start:      \n\t"
                    "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                    "esp.vld.128.ip  q1, x31, 16                     \n\t"
                    "addi %0, %0, 16                \n\t"   // in_ch_idx += 16

                    ".loop16_pad0_end:    \n\t"
                    "blt %0, s7, .loop16_pad0_start \n\t"  // if in_ch_idx < `in_channels - 15` abort

                    // move input_ptr, filter_ptr and conv_out
                    "mv %1, x30                     \n\t"
                    "mv %2, x31                     \n\t"
                    "esp.vmulas.s8.xacc  q0, q1     \n\t"

                    : "+r" (row_idx), "+r" (input_data_ptr), "+r" (filter_data_ptr)
                    :  "r"(row_size)
                    : "x30", "x31", "s7"
                );
skip_asm_pad0:
#endif
                    for (; row_idx < row_size - 3; row_idx += 4) {
                        conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        conv_out += *input_data_ptr++ * *filter_data_ptr++;
                    }
                    for (; row_idx < row_size; row_idx++) {
                        conv_out += *input_data_ptr++ * *filter_data_ptr++;
                    }
                }
                if (row_size >= 16) {
                    asm volatile (
                        "esp.movx.r.xacc.l  x30   \n\t"
                        "add %0, %0, x30          \n\t"
                        : "+r" (conv_out)
                        :
                        : "x30"
                    );
                }
                /* add input_offset term */
                conv_out += filter_sum[out_ch_idx];

                if (bias) {
                    conv_out += *bias_ptr++;
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t) conv_out;
            }
        }

        for (int32_t out_x = out_wd - right_pad; out_x < out_wd; out_x++) {
            const int32_t base_y = stride_ht * out_y;
            const int32_t base_x = stride_wd * out_x;
            const int32_t *out_mult_ptr = out_mult;
            const int32_t *out_shift_ptr = out_shift;
            const int32_t *bias_ptr = bias;
            for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0, filter_y_idx;
                for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                    for (int32_t filter_x_idx = 0; filter_x_idx < filter_wd - right_pad; filter_x_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x + filter_x_idx;

                        const int8_t *input_ptr = input_data +
                                        (in_row * input_wd + in_col) * in_channels;
                        const int8_t *filter_ptr = filter_data +
                                        out_ch_idx * in_channels * filter_ht * filter_wd +
                                        (filter_y_idx * filter_wd + filter_x_idx) * in_channels;
                        int32_t in_ch_idx = 0;
                        for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                        for (; in_ch_idx < in_channels; in_ch_idx ++) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                    }
                }

                if (bias) {
                    conv_out += *bias_ptr++;
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t) conv_out;
            }
        }
    }

    // Calculate the last row if needed
    if (bottom_pad) {
        int in_row = input_dims->height - filter_dims->height + 1;
        esp_nn_conv_s8_opt(&(data_dims_t){input_dims->width, 2, input_dims->channels, 0},
                            input_data + in_row * input_dims->width * input_dims->channels,
                            filter_dims, filter_data, bias,
                            &(data_dims_t){output_dims->width, 1, output_dims->channels, 0},
                            out_data, conv_params, quant_data);
    }
}

/* L1D cache budget: use half of 64KB to leave room for filter streaming */
#define L1D_BUDGET 32768

/**
 * Im2col convolution for small in_ch where filter_wd * in_ch < 16.
 *
 * Instead of padding channels (81% wasted MACs for in_ch=3),
 * concatenates the entire filter window into one contiguous vector:
 *   window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *
 * For each output pixel: copy the input window into a contiguous scratch
 * buffer, then use PIE dot product on the full window. No wasted MACs.
 *
 * Scratch layout: [filter_sum | im2col_buf]
 *   im2col_buf = filter_wd * filter_ht * in_ch bytes
 */
__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t window_len = filter_wd * filter_ht * in_ch;
    const int8_t pad_val = (int8_t)(-input_offset);

    /* Scratch: filter_sum[out_ch] + im2col_buf[window_len] */
    int32_t *filter_sum = (int32_t *)scratch;
    int8_t *im2col_buf = (int8_t *)scratch + out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset */
    const int8_t *fptr = filter_data;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        int32_t sum = 0;
        for (int32_t fi = 0; fi < window_len; fi++) {
            sum += *fptr++;
        }
        filter_sum[oc] = sum * input_offset;
    }

    /* Process each output pixel */
    int8_t *out_ptr = out_data;
    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int32_t base_x = out_x * stride_wd - pad_wd;

            /* Copy input window into contiguous im2col buffer */
            int8_t *buf = im2col_buf;
            for (int32_t fy = 0; fy < filter_ht; fy++) {
                int32_t in_y = base_y + fy;
                for (int32_t fx = 0; fx < filter_wd; fx++) {
                    int32_t in_x = base_x + fx;
                    if (in_y >= 0 && in_y < input_ht && in_x >= 0 && in_x < input_wd) {
                        const int8_t *src = input_data + (in_y * input_wd + in_x) * in_ch;
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = src[c];
                        }
                    } else {
                        /* Padding pixel */
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = pad_val;
                        }
                    }
                }
            }

            /* Dot product against each output channel's filter */
            const int32_t *out_mult = quant_data->mult;
            const int32_t *out_shift = quant_data->shift;
            const int8_t *filter_ptr = filter_data;

            for (int32_t oc = 0; oc < out_ch; oc++) {
                int32_t conv_out = pie_dot_s8(im2col_buf, filter_ptr, window_len);
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
                filter_ptr += window_len;
            }
        }
    }
}

/**
 * Tiled convolution: process T output rows at a time.
 * Converts padded conv into a series of no-pad sub-problems by
 * copying/padding input tiles into the scratch buffer.
 *
 * This keeps the working set in L1D for large input tensors.
 * Reuses the existing esp_nn_conv_s8_padded PIE inner loop per tile.
 */
__attribute__ ((noinline))
static void esp_nn_conv_s8_tiled(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;

    /* Check if we need channel padding for PIE (row_size must be >= 16) */
    int new_ch = in_ch;
    int need_ch_pad = 0;
    if (filter_wd * in_ch < 16) {
        new_ch = (16 + filter_wd - 1) / filter_wd;  /* minimum channels for PIE */
        new_ch = (new_ch + 15) & ~15;                /* align to 16 */
        need_ch_pad = 1;
    }
    int padded_input_wd = input_wd + 2 * pad_wd;

    /* Scratch layout:
     * [0] filter_sum: out_ch * 4 bytes
     * [after filter_sum] aligned_filter (if ch padding): filter_wd * filter_ht * new_ch * out_ch
     * [after filter] tile_input_buf: variable per tile
     */
    int32_t *filter_sum = (int32_t *) scratch;
    int filter_sum_size = out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset (once for entire layer) */
    const int8_t *fptr = filter_data;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        int32_t sum = 0;
        int32_t flen = filter_wd * filter_ht * in_ch;
        for (int32_t fi = 0; fi < flen; fi++) {
            sum += *fptr++;
        }
        filter_sum[oc] = sum * input_offset;
    }

    /* Channel-pad filter if needed (pad with 0s - doesn't affect filter_sum) */
    int8_t *aligned_filter = NULL;
    int aligned_filter_size = 0;
    if (need_ch_pad) {
        aligned_filter = (int8_t *)scratch + filter_sum_size;
        aligned_filter_size = filter_wd * filter_ht * new_ch * out_ch;
        memset(aligned_filter, 0, aligned_filter_size);
        const int8_t *src_f = filter_data;
        int8_t *dst_f = aligned_filter;
        for (int oc = 0; oc < out_ch; oc++) {
            for (int fh = 0; fh < filter_ht; fh++) {
                for (int fw = 0; fw < filter_wd; fw++) {
                    memcpy(dst_f, src_f, in_ch);
                    src_f += in_ch;
                    dst_f += new_ch;  /* zero-padded channels */
                }
            }
        }
    }

    /* Tile input buffer starts after filter_sum + aligned_filter */
    int8_t *tile_buf = (int8_t *)scratch + filter_sum_size + aligned_filter_size;

    /* Use effective channel count for tile buffer sizing */
    int eff_ch = need_ch_pad ? new_ch : in_ch;
    int tile_input_row_bytes = padded_input_wd * eff_ch;

    /* Compute tile height T (output rows per tile) */
    int tile_T = out_ht;
    int total_input_bytes = padded_input_wd * (input_ht + 2 * pad_ht) * eff_ch;
    int used_scratch = filter_sum_size + aligned_filter_size;
    if (total_input_bytes + used_scratch > L1D_BUDGET) {
        int budget_for_input = L1D_BUDGET - used_scratch;
        int min_input_rows = filter_ht;
        if (min_input_rows * tile_input_row_bytes <= budget_for_input) {
            tile_T = (budget_for_input - filter_ht * tile_input_row_bytes)
                     / (stride_ht * tile_input_row_bytes) + 1;
            if (tile_T < 1) tile_T = 1;
            if (tile_T > out_ht) tile_T = out_ht;
        }
    }

    /* Process tiles */
    const int8_t *use_filter = need_ch_pad ? aligned_filter : filter_data;
    data_dims_t eff_filter_dims = {filter_wd, filter_ht, eff_ch, 0};

    for (int32_t tile_y = 0; tile_y < out_ht; tile_y += tile_T) {
        int32_t actual_T = min(tile_T, out_ht - tile_y);

        int32_t in_row_start = tile_y * stride_ht - pad_ht;
        int32_t in_row_end = (tile_y + actual_T - 1) * stride_ht + filter_ht - 1;
        int32_t tile_input_ht = in_row_end - in_row_start + 1;

        /* Copy/pad input rows into tile buffer, with channel padding if needed */
        int8_t pad_val = (int8_t)(-input_offset);
        int8_t *dst = tile_buf;

        for (int32_t row = in_row_start; row <= in_row_end; row++) {
            if (row < 0 || row >= input_ht) {
                memset(dst, pad_val, padded_input_wd * eff_ch);
            } else {
                /* For each pixel in padded row */
                int8_t *row_dst = dst;
                /* Left padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
                /* Valid pixels - with optional channel padding */
                const int8_t *row_src = input_data + row * input_wd * in_ch;
                if (need_ch_pad) {
                    for (int px = 0; px < input_wd; px++) {
                        memcpy(row_dst, row_src, in_ch);
                        if (eff_ch > in_ch) {
                            memset(row_dst + in_ch, pad_val, eff_ch - in_ch);
                        }
                        row_src += in_ch;
                        row_dst += eff_ch;
                    }
                } else {
                    memcpy(row_dst, row_src, input_wd * in_ch);
                    row_dst += input_wd * in_ch;
                }
                /* Right padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
            }
            dst += padded_input_wd * eff_ch;
        }

        /* Sub-problem with pad=0, effective channels */
        data_dims_t tile_input_dims = {padded_input_wd, tile_input_ht, eff_ch, 0};
        data_dims_t tile_output_dims = {out_wd, actual_T, out_ch, 0};
        conv_params_t tile_conv_params = *conv_params;
        tile_conv_params.padding.width = 0;
        tile_conv_params.padding.height = 0;

        esp_nn_conv_s8_padded(&tile_input_dims, tile_buf,
                              &eff_filter_dims, use_filter, bias,
                              &tile_output_dims,
                              out_data + tile_y * out_wd * out_ch,
                              &tile_conv_params, quant_data,
                              filter_sum);
    }
}

int esp_nn_get_conv_scratch_size_esp32p4(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 32; /* extra buffer for alignment */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        if (in_ch < 16) {
            /* Channel-padding path: filter_sum + padded_filter + padded_input */
            int filter_sum_sz = out_ch * 4;
            int padded_filter_sz = 16 * out_ch;
            int padded_input_sz = 32; /* 16 bytes + alignment */
            return filter_sum_sz + padded_filter_sz + padded_input_sz + align_buf_size;
        }
        int transpose_buf_size = 2 * (8 * new_channels);
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0;
        }
        if (in_ch % 8) {
            input_scratch = input_wd * input_ht * new_channels;
        } else {
            input_scratch = 0;
        }
        filter_scratch = new_channels * out_ch;
        return input_scratch + filter_scratch + transpose_buf_size + align_buf_size;
    } else {
        new_channels = (in_ch + 15) & ~15;
        int offset_acc_scratch = out_ch * 4;

        if (pad_wd == 0 && pad_ht == 0 && filter_wd * in_ch >= 16) {
            /* Direct no-pad path: no input scratch needed */
            input_scratch = 0;
            filter_scratch = filter_wd * filter_ht * new_channels * out_ch;
            return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
        }

        /* Im2col path: scratch = filter_sum + im2col_buf */
        if (filter_wd * filter_ht * in_ch >= 16) {
            int window_len = filter_wd * filter_ht * in_ch;
            int im2col_scratch = window_len;  /* one window buffer */
            return offset_acc_scratch + im2col_scratch + align_buf_size;
        }

        if (pad_wd == 0 && pad_ht == 0) {
            /* Very small window (< 16 elements total): tiled path */
            int eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
            int filt_aligned = filter_wd * filter_ht * eff_ch * out_ch;
            int tile_input = input_wd * input_ht * eff_ch;
            return offset_acc_scratch + filt_aligned + tile_input + align_buf_size;
        }

        /* Padded case: check if tiling is beneficial */
        int padded_input_wd = input_wd + 2 * pad_wd;
        int full_input_size = padded_input_wd * (input_ht + 2 * pad_ht) * in_ch;

        if (full_input_size + offset_acc_scratch > L1D_BUDGET) {
            /* Tiled path: compute tile input size */
            int eff_ch = in_ch;
            int filt_aligned = 0;
            if (filter_wd * in_ch < 16) {
                eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
                filt_aligned = filter_wd * filter_ht * eff_ch * out_ch;
            }
            int tile_row_bytes = padded_input_wd * eff_ch;
            int budget_for_input = L1D_BUDGET - offset_acc_scratch - filt_aligned;
            int tile_T = 1;
            if (budget_for_input > 0 && filter_ht * tile_row_bytes <= budget_for_input) {
                tile_T = (budget_for_input - filter_ht * tile_row_bytes)
                         / (stride_ht * tile_row_bytes) + 1;
                if (tile_T > (int)(output_dims->height)) tile_T = output_dims->height;
            }
            int tile_input_rows = (tile_T - 1) * stride_ht + filter_ht + 2 * pad_ht;
            input_scratch = tile_input_rows * tile_row_bytes;
            filter_scratch = filt_aligned;
        } else {
            /* Monolithic padded path */
            input_scratch = full_input_size;
            filter_scratch = filter_wd * filter_ht * new_channels * out_ch;
        }
        return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
    }
    return align_buf_size;
}

void esp_nn_set_conv_scratch_buf_esp32p4(void *buf)
{
    // We are going to use the vector extensions
    asm volatile (
        "csrsi 0x7f2, 0b01      \n\t" // enable `esp` vector extension
        "li x29, 0b10           \n\t"
        "esp.movx.w.cfg x29     \n\t"
        :
        :
        : "x29"
    );

    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_s8_esp32p4(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }

    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {
        esp_nn_conv_s8_1x1(input_dims, input, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data,
                           scratch_buffer);
    } else if (pad_wd == 0 && pad_ht == 0 &&
               filter_wd * input_dims->channels >= 16) {
        /* No-pad, channels large enough for PIE: use direct padded path */
        esp_nn_conv_s8_padded(input_dims, input, filter_dims, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data,
                              scratch_buffer);
    } else if (filter_wd * filter_ht * input_dims->channels >= 16) {
        /* Small in_ch but window_len >= 16: use im2col for zero-waste PIE.
         * Also handles padded cases naturally. */
        esp_nn_conv_s8_im2col(input_dims, input, filter_dims, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data,
                              scratch_buffer);
    } else if (pad_wd != 0 || pad_ht != 0) {
        /* Padded case with very small window: use tiled path */
        esp_nn_conv_s8_tiled(input_dims, input, filter_dims, filter_data, bias,
                             output_dims, out_data, conv_params, quant_data,
                             scratch_buffer);
    } else {
        /* Tiny output: fall back to generic opt */
        esp_nn_conv_s8_opt(input_dims, input, filter_dims, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data);
    }
}
