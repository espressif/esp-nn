/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
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
#include "esp_nn_generic_opt.h"

#include <common_functions.h>

static int16_t *scratch_buffer = NULL;

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
                conv_out = esp_nn_multiply_by_quantized_mult_fast(conv_out, *out_mult++, *out_shift++);
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
                conv_out = esp_nn_multiply_by_quantized_mult_fast(conv_out, *out_mult_ptr++, *out_shift_ptr++);
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
                conv_out = esp_nn_multiply_by_quantized_mult_fast(conv_out, *out_mult_ptr++, *out_shift_ptr++);
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
        int transpose_buf_size = 2 * (8 * new_channels); /* to store intermediate data */
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0; // not using this for leftover
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
        if (pad_wd == 0 && pad_ht == 0) {
            input_scratch = 0;
        } else {
            input_scratch = (input_wd + 2 * pad_wd) * (input_ht + 2 * pad_ht) * in_ch;
        }
        filter_scratch = filter_wd * filter_ht * new_channels * out_ch;
        int offset_acc_scratch = out_ch * 4;
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
    } else if (pad_wd == 0 && pad_ht == 0) {
        esp_nn_conv_s8_padded(input_dims, input, filter_dims, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data,
                              scratch_buffer);
    } else {
        esp_nn_conv_s8_opt(input_dims, input, filter_dims, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data);
    }
}
