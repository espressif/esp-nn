/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <esp_nn_defs.h>

#include <common_functions.h>

static int16_t *scratch_buffer = NULL;

extern void esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3(const int16_t *input_data,
                                                        const uint16_t input_wd,
                                                        const uint16_t input_ht,
                                                        const uint16_t channels,
                                                        const uint16_t pad_wd,
                                                        const uint16_t pad_ht,
                                                        const uint16_t stride_wd,
                                                        const uint16_t stride_ht,
                                                        const uint16_t ch_mult,
                                                        const int16_t *filter_data,
                                                        const int32_t *bias,
                                                        int8_t *out_data,
                                                        const uint16_t out_wd,
                                                        const uint16_t out_ht,
                                                        const int32_t out_offset,
                                                        const int32_t *out_shift,
                                                        const int32_t *out_mult,
                                                        const int32_t activation_min,
                                                        const int32_t activation_max);

extern void esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(const int8_t *input_data,
                                                              const uint16_t input_wd,
                                                              const uint16_t input_ht,
                                                              const uint16_t channels,
                                                              const int32_t input_offset,
                                                              const uint16_t stride_wd,
                                                              const uint16_t stride_ht,
                                                              const int8_t *filter_data,
                                                              const int32_t *bias,
                                                              int8_t *out_data,
                                                              const uint16_t out_wd,
                                                              const uint16_t out_ht,
                                                              const int32_t out_offset,
                                                              const int32_t *out_shift,
                                                              const int32_t *out_mult,
                                                              const int32_t activation_min,
                                                              const int32_t activation_max);

extern void esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3(const int16_t *input_data,
                                                               const uint16_t input_wd,
                                                               const uint16_t input_ht,
                                                               const uint16_t channels,
                                                               const uint16_t stride_wd,
                                                               const uint16_t stride_ht,
                                                               const int16_t *filter_data,
                                                               const int32_t *bias,
                                                               int8_t *out_data,
                                                               const uint16_t out_wd,
                                                               const uint16_t out_ht,
                                                               const int32_t out_offset,
                                                               const int32_t *out_shift,
                                                               const int32_t *out_mult,
                                                               const int32_t activation_min,
                                                               const int32_t activation_max);

extern void esp_nn_depthwise_conv_s16_mult8_esp32s3(const int16_t *input_data,
                                                    const uint16_t input_wd,
                                                    const uint16_t input_ht,
                                                    const uint16_t channels,
                                                    const uint16_t pad_wd,
                                                    const uint16_t pad_ht,
                                                    const uint16_t stride_wd,
                                                    const uint16_t stride_ht,
                                                    const uint16_t ch_mult,
                                                    const int16_t *filter_data,
                                                    const uint16_t filter_wd,
                                                    const uint16_t filter_ht,
                                                    const int32_t *bias,
                                                    int8_t *out_data,
                                                    const uint16_t out_wd,
                                                    const uint16_t out_ht,
                                                    const int32_t out_offset,
                                                    const int32_t *out_shift,
                                                    const int32_t *out_mult,
                                                    const int32_t activation_min,
                                                    const int32_t activation_max);

extern void esp_nn_depthwise_conv_s16_mult4_esp32s3(const int16_t *input_data,
                                                    const uint16_t input_wd,
                                                    const uint16_t input_ht,
                                                    const uint16_t channels,
                                                    const uint16_t pad_wd,
                                                    const uint16_t pad_ht,
                                                    const uint16_t stride_wd,
                                                    const uint16_t stride_ht,
                                                    const uint16_t ch_mult,
                                                    const int16_t *filter_data,
                                                    const uint16_t filter_wd,
                                                    const uint16_t filter_ht,
                                                    const int32_t *bias,
                                                    int8_t *out_data,
                                                    const uint16_t out_wd,
                                                    const uint16_t out_ht,
                                                    const int32_t out_offset,
                                                    const int32_t *out_shift,
                                                    const int32_t *out_mult,
                                                    const int32_t activation_min,
                                                    const int32_t activation_max);

extern void esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3(const int16_t *input_data,
                                                        const uint16_t input_wd,
                                                        const uint16_t input_ht,
                                                        const uint16_t channels,
                                                        const uint16_t pad_wd,
                                                        const uint16_t pad_ht,
                                                        const uint16_t stride_wd,
                                                        const uint16_t stride_ht,
                                                        const int16_t *filter_data,
                                                        const int32_t *bias,
                                                        int8_t *out_data,
                                                        const uint16_t out_wd,
                                                        const uint16_t out_ht,
                                                        const int32_t out_offset,
                                                        const int32_t *out_shift,
                                                        const int32_t *out_mult,
                                                        const int32_t activation_min,
                                                        const int32_t activation_max);

extern void esp_nn_depthwise_conv_s16_mult1_esp32s3(const int16_t *input_data,
                                                    const uint16_t input_wd,
                                                    const uint16_t input_ht,
                                                    const uint16_t channels,
                                                    const uint16_t pad_wd,
                                                    const uint16_t pad_ht,
                                                    const uint16_t stride_wd,
                                                    const uint16_t stride_ht,
                                                    const int16_t *filter_data,
                                                    const uint16_t filter_wd,
                                                    const uint16_t filter_ht,
                                                    const int32_t *bias,
                                                    int8_t *out_data,
                                                    const uint16_t out_wd,
                                                    const uint16_t out_ht,
                                                    const int32_t out_offset,
                                                    const int32_t *out_shift,
                                                    const int32_t *out_mult,
                                                    const int32_t activation_min,
                                                    const int32_t activation_max);

extern void esp_nn_s8_to_s16_esp32s3(const int8_t *src, int16_t *dst, const int size);

extern void esp_nn_aligned_s8_to_s16_with_offset_esp32s3(const int8_t *src, int16_t *dst,
                                                         const int size, const int32_t offset);

static void esp_nn_depthwise_conv_s8_unrolled(const int8_t *input_data,
                                              const uint16_t input_wd,
                                              const uint16_t input_ht,
                                              const uint16_t channels,
                                              const int32_t input_offset,
                                              const uint16_t pad_wd,
                                              const uint16_t pad_ht,
                                              const uint16_t stride_wd,
                                              const uint16_t stride_ht,
                                              const uint16_t ch_mult,
                                              const int8_t *filter_data,
                                              const uint16_t filter_wd,
                                              const uint16_t filter_ht,
                                              const int32_t *bias,
                                              int8_t *out_data,
                                              const uint16_t out_wd,
                                              const uint16_t out_ht,
                                              const int32_t out_offset,
                                              const int32_t *out_shift,
                                              const int32_t *out_mult,
                                              const int32_t activation_min,
                                              const int32_t activation_max)
{
    int out_idx = 0;
    for (int out_y = 0; out_y < out_ht; out_y++) { //height loop
        const int16_t base_y = (out_y * stride_ht) - pad_ht;
        for (int out_x = 0; out_x < out_wd; out_x++) { //width_loop
            const int16_t base_x = (out_x * stride_wd) - pad_wd;
            for (int ch_idx = 0; ch_idx < channels; ch_idx++) {//channel_loop
                int ch_mult_idx = 0;
                for (; ch_mult_idx < ch_mult - 3; ch_mult_idx += 4) {
                    int32_t result0 = 0, result1 = 0, result2 = 0, result3 = 0;
                    const int out_ch_idx = ch_mult_idx + ch_idx * ch_mult;

                    /* Select filter so as the point doesn't lie outside block */
                    int filter_y_start = max(0, -base_y);
                    int filter_x_start = max(0, -base_x);
                    int filter_y_end = min(filter_ht, input_ht - base_y);
                    int filter_x_end = min(filter_wd, input_wd - base_x);

                    for (int filter_y_idx = filter_y_start; filter_y_idx < filter_y_end; filter_y_idx++) {
                        const int32_t idx_y = base_y + filter_y_idx;
                        for (int filter_x_idx = filter_x_start; filter_x_idx < filter_x_end; filter_x_idx++) {
                            const int32_t idx_x = base_x + filter_x_idx;
                            int32_t input_index = (idx_y * input_wd + idx_x) * channels + ch_idx;
                            int32_t filter_index = (filter_y_idx * filter_wd + filter_x_idx) * (channels * ch_mult) + out_ch_idx;
                            int32_t input_val = input_data[input_index] + input_offset;
                            int32_t filter_val0 = filter_data[filter_index + 0];
                            int32_t filter_val1 = filter_data[filter_index + 1];
                            int32_t filter_val2 = filter_data[filter_index + 2];
                            int32_t filter_val3 = filter_data[filter_index + 3];
                            result0 += input_val * filter_val0;
                            result1 += input_val * filter_val1;
                            result2 += input_val * filter_val2;
                            result3 += input_val * filter_val3;
                        }
                    }
                    if (bias) {
                        result0 += bias[out_ch_idx + 0];
                        result1 += bias[out_ch_idx + 1];
                        result2 += bias[out_ch_idx + 2];
                        result3 += bias[out_ch_idx + 3];
                    }
                    result0 = esp_nn_multiply_by_quantized_mult(result0,
                                out_mult[out_ch_idx + 0], out_shift[out_ch_idx + 0]);
                    result1 = esp_nn_multiply_by_quantized_mult(result1,
                                out_mult[out_ch_idx + 1], out_shift[out_ch_idx + 1]);
                    result2 = esp_nn_multiply_by_quantized_mult(result2,
                                out_mult[out_ch_idx + 2], out_shift[out_ch_idx + 2]);
                    result3 = esp_nn_multiply_by_quantized_mult(result3,
                                out_mult[out_ch_idx + 3], out_shift[out_ch_idx + 3]);

                    result0 += out_offset;
                    result1 += out_offset;
                    result2 += out_offset;
                    result3 += out_offset;

                    result0 = max(result0, activation_min);
                    result1 = max(result1, activation_min);
                    result2 = max(result2, activation_min);
                    result3 = max(result3, activation_min);

                    result0 = min(result0, activation_max);
                    result1 = min(result1, activation_max);
                    result2 = min(result2, activation_max);
                    result3 = min(result3, activation_max);

                    out_data[out_idx++] = result0;
                    out_data[out_idx++] = result1;
                    out_data[out_idx++] = result2;
                    out_data[out_idx++] = result3;
                }

                /* left-over */
                for (; ch_mult_idx < ch_mult; ch_mult_idx++) {
                    int32_t result = 0;
                    const int out_ch_idx = ch_mult_idx + ch_idx * ch_mult;

                    /* Select filter so as the point doesn't lie outside block */
                    int filter_y_start = max(0, -base_y);
                    int filter_x_start = max(0, -base_x);
                    int filter_y_end = min(filter_ht, input_ht - base_y);
                    int filter_x_end = min(filter_wd, input_wd - base_x);

                    for (int filter_y_idx = filter_y_start; filter_y_idx < filter_y_end; filter_y_idx++) {
                        const int32_t idx_y = base_y + filter_y_idx;
                        for (int filter_x_idx = filter_x_start; filter_x_idx < filter_x_end; filter_x_idx++) {
                            const int32_t idx_x = base_x + filter_x_idx;
                            int32_t input_index = (idx_y * input_wd + idx_x) * channels + ch_idx;
                            int32_t filter_index = (filter_y_idx * filter_wd + filter_x_idx) * (channels * ch_mult) + out_ch_idx;
                            int32_t input_val = input_data[input_index] + input_offset;
                            int32_t filter_val = filter_data[filter_index];
                            result += input_val * filter_val;
                        }
                    }
                    if (bias) {
                        result += bias[out_ch_idx];
                    }
                    result = esp_nn_multiply_by_quantized_mult(result, out_mult[out_ch_idx], out_shift[out_ch_idx]);
                    result += out_offset;
                    result = max(result, activation_min);
                    result = min(result, activation_max);

                    out_data[out_idx++] = result;
                }
            }
        }
    }
}

void esp_nn_depthwise_conv_s8_ch_mult1(const int8_t *input_data,
                                       const uint16_t input_wd,
                                       const uint16_t input_ht,
                                       const uint16_t channels,
                                       const int32_t input_offset,
                                       const uint16_t pad_wd,
                                       const uint16_t pad_ht,
                                       const uint16_t stride_wd,
                                       const uint16_t stride_ht,
                                       const int8_t *filter_data,
                                       const uint16_t filter_wd,
                                       const uint16_t filter_ht,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_wd,
                                       const uint16_t out_ht,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max)
{
    int out_idx = 0;
    for (int out_y = 0; out_y < out_ht; out_y++) { //height loop
        const int16_t base_y = (out_y * stride_ht) - pad_ht;
        for (int out_x = 0; out_x < out_wd; out_x++) { //width_loop
            const int16_t base_x = (out_x * stride_wd) - pad_wd;
            for (int ch_idx = 0; ch_idx < channels; ch_idx++) {//channel_loop
                int32_t result = 0;
                /* Select filter so as the point doesn't lie outside block */
                int filter_y_start = max(0, -base_y);
                int filter_x_start = max(0, -base_x);
                int filter_y_end = min(filter_ht, input_ht - base_y);
                int filter_x_end = min(filter_wd, input_wd - base_x);

                for (int filter_y_idx = filter_y_start; filter_y_idx < filter_y_end; filter_y_idx++) {
                    const int32_t idx_y = base_y + filter_y_idx;
                    for (int filter_x_idx = filter_x_start; filter_x_idx < filter_x_end; filter_x_idx++) {
                        const int32_t idx_x = base_x + filter_x_idx;
                        int32_t input_index = (idx_y * input_wd + idx_x) * channels + ch_idx;
                        int32_t filter_index = (filter_y_idx * filter_wd + filter_x_idx) * channels + ch_idx;
                        int32_t input_val = input_data[input_index] + input_offset;
                        int32_t filter_val = filter_data[filter_index];
                        result += input_val * filter_val;
                    }
                }
                if (bias) {
                    result += bias[ch_idx];
                }
                result = esp_nn_multiply_by_quantized_mult(result, out_mult[ch_idx], out_shift[ch_idx]);
                result += out_offset;
                result = max(result, activation_min);
                result = min(result, activation_max);

                out_data[out_idx++] = result;
            }
        }
    }
}

int esp_nn_get_depthwise_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                                   const data_dims_t *filter_dims,
                                                   const data_dims_t *output_dims,
                                                   const dw_conv_params_t *conv_params)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t ch_mult = conv_params->ch_mult;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    int filter_size = filter_wd * filter_ht * channels * ch_mult;
    int pad_width = 0, pad_height = 0;

    if ((ch_mult == 1) && (channels % 8 == 0)) {
        if(filter_wd == 3 && filter_ht == 3) {
            if (channels % 16 == 0) {
                if (pad_wd || pad_ht) {
                    pad_width = pad_wd * 2;
                    pad_height = pad_ht * 2;
                } else {
                    pad_width = (out_wd * stride_wd + filter_wd - 1) - input_wd;
                    pad_height = (out_ht * stride_ht + filter_ht - 1) - input_ht;
                }
                if (pad_width || pad_height) {
                    int full_input = (input_wd + pad_width) * (input_ht + pad_height) * channels;
                    if (full_input <= 40 * 1024) {
                        return filter_size + full_input + 16;
                    } else {
                        /* Tiled: only need filter + strip buffer (filter_ht rows) */
                        int strip = (input_wd + pad_width) * filter_ht * channels;
                        return filter_size + strip + 16;
                    }
                } else {
                    return filter_size + 16;
                }
            } else if (channels >= 12) {
                /* ch % 8 == 0, not % 16, ch >= 12: pad channels to 16, s8 path + compaction */
                int new_ch = (channels + 15) & ~15;
                int new_filter_size = 9 * new_ch;
                int total_pad_wd = pad_wd * 2 + max(0, (out_wd * stride_wd + 2) - input_wd);
                int total_pad_ht = pad_ht * 2 + max(0, (out_ht * stride_ht + 2) - input_ht);
                int new_input_size = (input_wd + total_pad_wd) * (input_ht + total_pad_ht) * new_ch;
                int out_buf_size = out_wd * out_ht * new_ch;
                return new_filter_size + new_input_size + out_buf_size + 64;
            } else {
                /* ch=8: s16 path is more efficient (no channel padding overhead) */
                int input_s = input_wd * input_ht * channels;
                return  2 * (filter_size + input_s) + 32;
            }
        } else {
            int input_size = input_wd * input_ht * channels;
            int total_s16 = 2 * (filter_size + input_size);
            if (total_s16 <= 48 * 1024) {
                return total_s16 + 32;
            } else {
                /* Tiled: only need filter_s16 + tile buffer (filter_ht rows of input s16) */
                int tile_rows = filter_ht;
                int tile_s16 = 2 * input_wd * tile_rows * channels;
                return 2 * filter_size + tile_s16 + 32;
            }
        }
    } else if ((ch_mult == 1) && (channels > 3)) {
        // ch_mult=1, channels>3 case: pad channels to multiple of 8 for mult1
        int padded_channels = (channels + 7) & ~7;
        int padded_input_size = input_wd * input_ht * padded_channels;
        int padded_filter_size = filter_wd * filter_ht * padded_channels;

        // Calculate actual memory layout with 16-byte alignments (matching usage)
        size_t filter_bytes = padded_filter_size * sizeof(int16_t);
        size_t input_start = (filter_bytes + 15) & ~15;
        size_t input_bytes = padded_input_size * sizeof(int16_t);
        size_t out_start = (input_start + input_bytes + 15) & ~15;
        size_t out_bytes = out_wd * out_ht * padded_channels * sizeof(int8_t);
        size_t bias_start = (out_start + out_bytes + 15) & ~15;
        size_t bias_bytes = padded_channels * sizeof(int32_t);
        size_t shift_bytes = padded_channels * sizeof(int32_t);
        size_t mult_bytes = padded_channels * sizeof(int32_t);
        size_t total_size = bias_start + bias_bytes + shift_bytes + mult_bytes;

        return total_size + 16; // 16 for margin
    } else if (ch_mult % 4 == 0) {
        int input_size = input_wd * input_ht * channels;
        return  2 * (filter_size + input_size) + 32; // 32 for alignment
    }

    // Default fallback
    return 32;
}

void esp_nn_set_depthwise_conv_scratch_buf_esp32s3(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

/**
 * ESP32-S3 optimized depthwise convolution implementation.
 *
 * This function dispatches to various optimized implementations based on:
 * - Channel multiplier (ch_mult)
 * - Number of channels
 * - Filter dimensions
 * - Padding requirements
 *
 * For cases that don't have direct optimized implementations, the function
 * uses data padding techniques to leverage existing optimized functions:
 * - ch_mult % 4 != 0: Pad ch_mult to next multiple of 4, use mult4 functions
 * - ch_mult == 1, channels % 8 != 0: Fallback to C implementation for correctness
 *
 * Assumption 1: i/p channels == o/p channels
 * Assumption 2: Pointers are valid
 * Assumption 3: dilation width = 1
 */

#include "esp_nn_generic_opt.h"

void esp_nn_depthwise_conv_s8_esp32s3(const data_dims_t *input_dims,
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
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;
    const uint16_t ch_mult = conv_params->ch_mult;

    int filter_size = filter_wd * filter_ht * channels * ch_mult;
    int align_len = 16 - (filter_size & 15);
    int input_size = input_wd * input_ht * channels;
    int16_t *filter_data16 = scratch_buffer;
    int16_t *input_data16 = scratch_buffer + filter_size + align_len;
    if (scratch_buffer == NULL) {
        printf("esp_nn_depthwise_conv error! scratch_buffer not set!\n");
        return;
    }

    if ((ch_mult == 1) && (channels % 8 == 0)) {
        if ((filter_wd == 3) && (filter_ht == 3)) {
            if ((channels % 16 == 0) && (pad_wd == 1) && (pad_ht == 1)) {
                /* process in 8 bits with s8 padded assembly */
                int8_t *filter_aligned = (int8_t *) scratch_buffer;
                int8_t *input_padded = (int8_t *) scratch_buffer + filter_size + align_len;
                memcpy(filter_aligned, filter_data, filter_size);

                int padded_input_size = (input_wd + 2*pad_wd) * (input_ht + 2*pad_ht) * channels;
                if (padded_input_size <= 40 * 1024) {
                    /* Small enough — full padding, single assembly call */
                    esp_nn_aligned_s8_pad_with_value(input_data, input_padded, input_wd, input_ht, channels,
                                                     -input_offset, pad_wd, pad_ht);
                    esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(input_padded, input_wd + 2 * pad_wd,
                                                                      input_ht + 2 * pad_ht, channels, input_offset,
                                                                      stride_wd, stride_ht, filter_aligned, bias,
                                                                      out_data, out_wd, out_ht, out_offset, out_shift,
                                                                      out_mult, activation_min, activation_max);
                } else {
                    /* Large input: row-tiled processing to reduce cache pressure.
                     * Pad and process a strip of output rows at a time. */
                    int padded_wd = input_wd + 2 * pad_wd;
                    int8_t pad_val = (int8_t)(-input_offset);

                    for (int out_y = 0; out_y < out_ht; out_y++) {
                        int in_y_start = out_y * stride_ht; /* in padded coords (pad_ht already accounted) */
                        /* Pad filter_ht rows of input into scratch */
                        int8_t *tile = input_padded;
                        for (int fy = 0; fy < filter_ht; fy++) {
                            int src_y = in_y_start + fy - pad_ht; /* original input row */
                            if (src_y < 0 || src_y >= input_ht) {
                                /* Padding row */
                                memset(tile, pad_val, padded_wd * channels);
                            } else {
                                /* Left pad */
                                memset(tile, pad_val, pad_wd * channels);
                                /* Copy input row */
                                memcpy(tile + pad_wd * channels,
                                       input_data + src_y * input_wd * channels,
                                       input_wd * channels);
                                /* Right pad */
                                memset(tile + (pad_wd + input_wd) * channels, pad_val, pad_wd * channels);
                            }
                            tile += padded_wd * channels;
                        }
                        /* Process one output row */
                        esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
                            input_padded, padded_wd, filter_ht, channels, input_offset,
                            stride_wd, 1, filter_aligned, bias,
                            out_data + out_y * out_wd * channels,
                            out_wd, 1, out_offset, out_shift,
                            out_mult, activation_min, activation_max);
                    }
                }
            } else if ((channels % 16 == 0) && (pad_wd == 0) && (pad_ht == 0)) {
                /* process in 8 bits */
                int8_t *filter_aligned = (int8_t *) scratch_buffer;
                int8_t *input_padded = (int8_t *) scratch_buffer + filter_size + align_len;

                // check if we need to pad additionally
                int pad_right = (out_wd * stride_wd + filter_wd - 1) - input_wd;
                int pad_bottom = (out_ht * stride_ht + filter_ht - 1) - input_ht;
                if (pad_right || pad_bottom) { // pad right and bottom
                    esp_nn_aligned_s8_pad_end_with_value(input_data, input_padded, input_wd, input_ht,
                                                         channels, -input_offset, pad_right, pad_bottom);
                } else {
                    input_padded = (int8_t *) input_data;
                }
                memcpy(filter_aligned, filter_data, filter_size);
                esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(input_padded, input_wd + pad_right,
                                                                  input_ht + pad_bottom, channels, input_offset,
                                                                  stride_wd, stride_ht, filter_aligned, bias,
                                                                  out_data, out_wd, out_ht, out_offset, out_shift,
                                                                  out_mult, activation_min, activation_max);
            } else if (channels >= 12) {
                /* channels % 8 == 0, not % 16, channels >= 12: pad to 16 is worthwhile
                 * (overhead <= 33%). For ch=8, padding to 16 doubles data — use s16 instead */
                int new_ch = (channels + 15) & ~15;
                int8_t pad_val = (int8_t)(-input_offset);

                /* Pad filter: 3x3 x new_ch */
                int new_filter_size = 9 * new_ch;
                int8_t *filter_padded = (int8_t *) scratch_buffer;
                memset(filter_padded, 0, new_filter_size);
                for (int f = 0; f < 9; f++) {
                    memcpy(filter_padded + f * new_ch, filter_data + f * channels, channels);
                }

                /* Pad input: (input_wd + 2*pad) x (input_ht + 2*pad) x new_ch */
                int new_input_wd = input_wd + 2 * pad_wd;
                int new_input_ht = input_ht + 2 * pad_ht;
                int pad_right = max(0, (out_wd * stride_wd + 3 - 1) - (input_wd + 2 * pad_wd));
                int pad_bottom = max(0, (out_ht * stride_ht + 3 - 1) - (input_ht + 2 * pad_ht));
                new_input_wd += pad_right;
                new_input_ht += pad_bottom;

                int8_t *input_padded = filter_padded + new_filter_size + 16;
                int padded_input_total = new_input_wd * new_input_ht * new_ch;
                /* Fill entire padded input with pad_val first */
                memset(input_padded, pad_val, padded_input_total);
                /* Copy actual input data into correct positions */
                for (int y = 0; y < input_ht; y++) {
                    for (int x = 0; x < input_wd; x++) {
                        int dst_y = y + pad_ht;
                        int dst_x = x + pad_wd;
                        memcpy(input_padded + (dst_y * new_input_wd + dst_x) * new_ch,
                               input_data + (y * input_wd + x) * channels, channels);
                    }
                }

                /* Padded output buffer */
                int8_t *out_padded = input_padded + padded_input_total;

                /* Pad quant arrays */
                int32_t shift_pad[new_ch], mult_pad[new_ch], bias_pad[new_ch];
                memcpy(shift_pad, out_shift, channels * sizeof(int32_t));
                memcpy(mult_pad, out_mult, channels * sizeof(int32_t));
                memset(shift_pad + channels, 0, (new_ch - channels) * sizeof(int32_t));
                memset(mult_pad + channels, 0, (new_ch - channels) * sizeof(int32_t));
                if (bias) {
                    memcpy(bias_pad, bias, channels * sizeof(int32_t));
                    memset(bias_pad + channels, 0, (new_ch - channels) * sizeof(int32_t));
                }

                esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
                    input_padded, new_input_wd, new_input_ht, new_ch, input_offset,
                    stride_wd, stride_ht, filter_padded,
                    bias ? bias_pad : NULL, out_padded,
                    out_wd, out_ht, out_offset, shift_pad, mult_pad,
                    activation_min, activation_max);

                /* Compact output: strip padding channels */
                for (int pos = 0; pos < out_wd * out_ht; pos++) {
                    memcpy(out_data + pos * channels,
                           out_padded + pos * new_ch, channels);
                }
            } else {
                /* ch < 12 (e.g., ch=8), 3x3: use s16 mult1 3x3 path */
                esp_nn_s8_to_s16_esp32s3(filter_data, filter_data16, filter_size);
                esp_nn_aligned_s8_to_s16_with_offset_esp32s3(input_data, input_data16, input_size, input_offset);
                esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3(input_data16, input_wd, input_ht, channels,
                                                            pad_wd, pad_ht, stride_wd, stride_ht, filter_data16,
                                                            bias, out_data, out_wd, out_ht, out_offset, out_shift,
                                                            out_mult, activation_min, activation_max);
            }
        } else { // all other ch_mult == 1, channels % 8 == 0
            /* Tiled s16 processing: convert filter once, process input in row strips
             * to keep working set within DCache (64KB) */
            esp_nn_s8_to_s16_esp32s3(filter_data, filter_data16, filter_size);

            /* Check if full conversion fits comfortably in cache */
            int total_s16_size = 2 * (filter_size + input_size);
            if (total_s16_size <= 48 * 1024) {
                /* Small enough — full conversion is fine */
                esp_nn_aligned_s8_to_s16_with_offset_esp32s3(input_data, input_data16, input_size, input_offset);
                esp_nn_depthwise_conv_s16_mult1_esp32s3(input_data16, input_wd, input_ht, channels,
                                                        pad_wd, pad_ht, stride_wd, stride_ht, filter_data16,
                                                        filter_wd, filter_ht, bias, out_data, out_wd, out_ht, out_offset, out_shift,
                                                        out_mult, activation_min, activation_max);
            } else {
                /* Large input: process in row tiles to reduce cache pressure.
                 * Convert only the input rows needed for each output row strip. */
                int16_t *tile_buf = input_data16; /* reuse scratch for tile */

                for (int out_row = 0; out_row < out_ht; out_row++) {
                    int in_row_start = out_row * stride_ht - pad_ht;
                    int in_row_end = in_row_start + filter_ht;

                    /* Fill tile: pad rows that are outside input bounds */
                    int16_t *dst = tile_buf;
                    for (int r = in_row_start; r < in_row_end; r++) {
                        if (r < 0 || r >= input_ht) {
                            /* Padding row: fill with input_offset */
                            for (int i = 0; i < input_wd * channels; i++) {
                                dst[i] = (int16_t)input_offset;
                            }
                        } else {
                            /* Valid row: convert s8 to s16 with offset */
                            const int8_t *src = input_data + r * input_wd * channels;
                            for (int i = 0; i < input_wd * channels; i++) {
                                dst[i] = (int16_t)src[i] + (int16_t)input_offset;
                            }
                        }
                        dst += input_wd * channels;
                    }

                    /* Process one output row */
                    esp_nn_depthwise_conv_s16_mult1_esp32s3(tile_buf, input_wd, filter_ht, channels,
                                                            pad_wd, 0, stride_wd, 1, filter_data16,
                                                            filter_wd, filter_ht, bias,
                                                            out_data + out_row * out_wd * channels,
                                                            out_wd, 1, out_offset, out_shift,
                                                            out_mult, activation_min, activation_max);
                }
            }
        }
    } else if ((ch_mult == 1) && (channels > 3)) {
        // For ch_mult=1, pad channels to multiple of 8 for optimized mult1 function
        int padded_channels = (channels + 7) & ~7; // Round up to multiple of 8
        int padded_input_size = input_wd * input_ht * padded_channels;
        int padded_filter_size = filter_wd * filter_ht * padded_channels;

        // Use scratch buffer for padded data (ensure 16-byte alignment for SIMD)
        int16_t *padded_filter_data16 = (int16_t*)scratch_buffer;
        size_t input_start = (size_t)(padded_filter_data16 + padded_filter_size);
        int16_t *padded_input_data16 = (int16_t*)((input_start + 15) & ~15);
        size_t out_start = (size_t)(padded_input_data16 + padded_input_size);
        int8_t *padded_out_data = (int8_t*)((out_start + 15) & ~15);

        // Create padded parameter arrays
        size_t bias_start = (size_t)(padded_out_data + out_wd * out_ht * padded_channels);
        int32_t *padded_bias = (int32_t*)((bias_start + 15) & ~15);
        int32_t *padded_shift = padded_bias + padded_channels;
        int32_t *padded_mult = padded_shift + padded_channels;

        // Initialize padded parameters - copy valid values, set padded ones to safe defaults
        memset(padded_bias, 0, padded_channels * sizeof(int32_t));
        memset(padded_shift, 0, padded_channels * sizeof(int32_t));
        memset(padded_mult, 0, padded_channels * sizeof(int32_t));

        if (bias) {
            memcpy(padded_bias, bias, channels * sizeof(int32_t));
        }
        if (out_shift) {
            memcpy(padded_shift, out_shift, channels * sizeof(int32_t));
        }
        if (out_mult) {
            memcpy(padded_mult, out_mult, channels * sizeof(int32_t));
        }

        // Convert filter data to padded layout (zero out extra channels)
        memset(padded_filter_data16, 0, padded_filter_size * sizeof(int16_t));
        for (int c = 0; c < channels; c++) {
            for (int fy = 0; fy < filter_ht; fy++) {
                for (int fx = 0; fx < filter_wd; fx++) {
                    int orig_idx = (fy * filter_wd + fx) * channels + c;
                    int padded_idx = (fy * filter_wd + fx) * padded_channels + c;
                    padded_filter_data16[padded_idx] = (int16_t) filter_data[orig_idx];
                }
            }
        }

        // Convert input data to padded layout (zero out extra channels, apply offset)
        memset(padded_input_data16, 0, padded_input_size * sizeof(int16_t));
        for (int h = 0; h < input_ht; h++) {
            for (int w = 0; w < input_wd; w++) {
                for (int c = 0; c < channels; c++) {
                    int orig_idx = (h * input_wd + w) * channels + c;
                    int padded_idx = (h * input_wd + w) * padded_channels + c;
                    padded_input_data16[padded_idx] = (int16_t) input_data[orig_idx] + input_offset;
                }
            }
        }

        // Call mult1 with padded data
        esp_nn_depthwise_conv_s16_mult1_esp32s3(padded_input_data16, input_wd, input_ht, padded_channels,
                                                pad_wd, pad_ht, stride_wd, stride_ht, padded_filter_data16,
                                                filter_wd, filter_ht, padded_bias, padded_out_data, out_wd, out_ht, out_offset, padded_shift,
                                                padded_mult, activation_min, activation_max);

        // Copy back only valid channels
        for (int h = 0; h < out_ht; h++) {
            for (int w = 0; w < out_wd; w++) {
                for (int c = 0; c < channels; c++) {
                    int out_idx = (h * out_wd + w) * channels + c;
                    int padded_idx = (h * out_wd + w) * padded_channels + c;
                    out_data[out_idx] = padded_out_data[padded_idx];
                }
            }
        }
    } else if (ch_mult % 8 == 0) {
        // Channel multiplier is optimized multiple - use direct s16 functions
        esp_nn_s8_to_s16_esp32s3(filter_data, filter_data16, filter_size);
        esp_nn_aligned_s8_to_s16_with_offset_esp32s3(input_data, input_data16, input_size, input_offset);
        if (filter_wd == 3 && filter_ht == 3) {
            esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3(input_data16, input_wd, input_ht, channels,
                                                        pad_wd, pad_ht, stride_wd, stride_ht, ch_mult,
                                                        filter_data16, bias,
                                                        out_data, out_wd, out_ht, out_offset, out_shift,
                                                        out_mult, activation_min, activation_max);
        } else {
            esp_nn_depthwise_conv_s16_mult8_esp32s3(input_data16, input_wd, input_ht, channels,
                                                    pad_wd, pad_ht, stride_wd, stride_ht, ch_mult,
                                                    filter_data16, filter_wd, filter_ht, bias,
                                                    out_data, out_wd, out_ht, out_offset, out_shift,
                                                    out_mult, activation_min, activation_max);
        }
    } else if (ch_mult % 4 == 0) {
        esp_nn_s8_to_s16_esp32s3(filter_data, filter_data16, filter_size);
        esp_nn_aligned_s8_to_s16_with_offset_esp32s3(input_data, input_data16, input_size, input_offset);
        esp_nn_depthwise_conv_s16_mult4_esp32s3(input_data16, input_wd, input_ht, channels,
                                                pad_wd, pad_ht, stride_wd, stride_ht, ch_mult,
                                                filter_data16, filter_wd, filter_ht, bias,
                                                out_data, out_wd, out_ht, out_offset, out_shift,
                                                out_mult, activation_min, activation_max);
    } else {
        esp_nn_depthwise_conv_s8_opt(input_dims, input_data, filter_dims, filter_data, bias,
                                     output_dims, out_data, conv_params, quant_data);
    }
}
