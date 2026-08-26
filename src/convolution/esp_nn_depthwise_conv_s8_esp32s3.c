/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <esp_nn_defs.h>

#include <common_functions.h>
#include <esp_nn_multicore.h>

static int16_t *scratch_buffer = NULL;
static uint8_t *preferred_scratch_buffer = NULL;
static size_t preferred_scratch_size = 0;

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

/* int8 depthwise for (K,1) filters on width-1 tensors, the shape a 1-D
 * convolutional encoder produces. Measured on ESP32-S3 with a 16-layer speech
 * encoder: 461-489 ms against 478-507 ms for the int16 path it replaces, and
 * scratch for the affected layers falls from 99 kB to about 1 kB. */
#ifndef ESP_NN_DW_S8_KX1
#define ESP_NN_DW_S8_KX1 1
#endif

/* Everything past the sixth argument goes by pointer: see the note in
 * esp_nn_depthwise_conv_s8_kx1_esp32s3.S about stack-argument placement. */
typedef struct {
    int8_t        *out;
    const int32_t *out_mult;
    const int32_t *out_shift;
    int32_t        out_offset;
    int32_t        activation_min;
    int32_t        activation_max;
} dw_kx1_params_t;

/* Everything from the multiply-accumulate through the activation clamp;
 * results come back as 16 int32 and the caller truncates them to int8. */
extern void esp_nn_dw_s8_kx1_esp32s3(const int8_t *in, int tap_stride,
                                     int n_taps, const int8_t *filter,
                                     const int32_t *bias16,
                                     const dw_kx1_params_t *p);

extern void esp_nn_aligned_s8_to_s16_with_offset_esp32s3(const int8_t *src, int16_t *dst,
                                                         const int size, const int32_t offset);

/* Unaligned-source variant. Reads up to 32 bytes past src + size and 16
 * before src, so the row must be INTERIOR to a larger readable object. */
extern void esp_nn_s8_to_s16_with_offset_row_esp32s3(const int8_t *src, int16_t *dst,
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

typedef struct {
    const int8_t *input_data;
    uint16_t input_wd, input_ht, channels;
    int32_t input_offset;
    /* pad_wd/pad_ht are the leading (left/top) pads; pad_right/pad_bottom are
     * derived from the output extent and may differ, as TFLite's "SAME"
     * padding is asymmetric whenever the total pad is odd. */
    uint16_t pad_wd, pad_ht, pad_right, pad_bottom, stride_wd, stride_ht;
    const int8_t *filter_aligned;
    const int32_t *bias;
    int8_t *out_data;
    uint16_t out_wd;
    int out_y_begin, out_y_end;
    int32_t out_offset;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min, activation_max;
    int8_t *tile_buf;
    int tile_bytes;
} dw3x3_strip_job_t;

/* Strip-tiled processing of output rows [out_y_begin, out_y_end): copy a
 * strip of padded input rows into tile_buf and run the padded 3x3 kernel
 * per strip. Used directly and as the per-core body of the dual-core split. */
static void dw3x3_strip_rows(const dw3x3_strip_job_t *j)
{
    const int padded_wd = j->input_wd + j->pad_wd + j->pad_right;
    const int8_t pad_val = (int8_t)(-j->input_offset);
    const int row_bytes = padded_wd * j->channels;
    int strip_rows = j->tile_bytes / row_bytes;
    if (strip_rows < 3) {
        strip_rows = 3; /* caller guarantees at least filter_ht rows fit */
    }
    const int out_rows_per_strip = (strip_rows - 3) / j->stride_ht + 1;

    for (int out_y = j->out_y_begin; out_y < j->out_y_end; ) {
        int n_out = out_rows_per_strip;
        if (n_out > j->out_y_end - out_y) {
            n_out = j->out_y_end - out_y;
        }
        const int in_rows = (n_out - 1) * j->stride_ht + 3;
        const int in_y_start = out_y * j->stride_ht; /* padded coords */
        int8_t *tile = j->tile_buf;
        for (int r = 0; r < in_rows; r++) {
            const int src_y = in_y_start + r - j->pad_ht;
            if (src_y < 0 || src_y >= j->input_ht) {
                memset(tile, pad_val, row_bytes);
            } else {
                memset(tile, pad_val, j->pad_wd * j->channels);
                memcpy(tile + j->pad_wd * j->channels,
                       j->input_data + src_y * j->input_wd * j->channels,
                       j->input_wd * j->channels);
                memset(tile + (j->pad_wd + j->input_wd) * j->channels, pad_val,
                       j->pad_right * j->channels);
            }
            tile += row_bytes;
        }
        esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
                j->tile_buf, padded_wd, in_rows, j->channels, j->input_offset,
                j->stride_wd, j->stride_ht, j->filter_aligned, j->bias,
                j->out_data + out_y * j->out_wd * j->channels,
                j->out_wd, n_out, j->out_offset, j->out_shift,
                j->out_mult, j->activation_min, j->activation_max);
        out_y += n_out;
    }
}

static void dw3x3_strip_mt_worker(void *p)
{
    dw3x3_strip_rows((const dw3x3_strip_job_t *)p);
}

typedef struct {
    const int8_t *input_padded;
    uint16_t padded_wd;
    uint16_t in_rows;
    uint16_t channels;
    int32_t input_offset;
    uint16_t stride_wd, stride_ht;
    const int8_t *filter_aligned;
    const int32_t *bias;
    int8_t *out_data;
    uint16_t out_wd, out_rows;
    int32_t out_offset;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min, activation_max;
} dw3x3_mt_job_t;

static void dw3x3_mt_worker(void *p)
{
    const dw3x3_mt_job_t *j = (const dw3x3_mt_job_t *)p;
    esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
            j->input_padded, j->padded_wd, j->in_rows, j->channels,
            j->input_offset, j->stride_wd, j->stride_ht, j->filter_aligned,
            j->bias, j->out_data, j->out_wd, j->out_rows, j->out_offset,
            j->out_shift, j->out_mult, j->activation_min, j->activation_max);
}

/* Runs the padded 3x3 kernel with the output rows split across both cores.
 * The padded input is shared read-only; each core writes a disjoint row
 * range, so results are bit-identical to the single call. */
static void dw3x3_run_split(const int8_t *input_padded, uint16_t padded_wd,
                            uint16_t padded_ht, uint16_t channels,
                            int32_t input_offset, uint16_t stride_wd,
                            uint16_t stride_ht, const int8_t *filter_aligned,
                            const int32_t *bias, int8_t *out_data,
                            uint16_t out_wd, uint16_t out_ht,
                            int32_t out_offset, const int32_t *out_shift,
                            const int32_t *out_mult, int32_t activation_min,
                            int32_t activation_max)
{
    if (esp_nn_dual_core_active() && out_ht >= 4) {
        const uint16_t h0 = out_ht / 2;
        dw3x3_mt_job_t job = {
            .input_padded = input_padded, .padded_wd = padded_wd,
            .in_rows = (uint16_t)((h0 - 1) * stride_ht + 3),
            .channels = channels, .input_offset = input_offset,
            .stride_wd = stride_wd, .stride_ht = stride_ht,
            .filter_aligned = filter_aligned, .bias = bias,
            .out_data = out_data, .out_wd = out_wd, .out_rows = h0,
            .out_offset = out_offset, .out_shift = out_shift,
            .out_mult = out_mult, .activation_min = activation_min,
            .activation_max = activation_max,
        };
        if (esp_nn_dual_core_run(dw3x3_mt_worker, &job)) {
            esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
                    input_padded + (int32_t)h0 * stride_ht * padded_wd * channels,
                    padded_wd, (uint16_t)(padded_ht - h0 * stride_ht), channels,
                    input_offset, stride_wd, stride_ht, filter_aligned, bias,
                    out_data + (int32_t)h0 * out_wd * channels,
                    out_wd, (uint16_t)(out_ht - h0), out_offset,
                    out_shift, out_mult, activation_min, activation_max);
            esp_nn_dual_core_wait();
            return;
        }
    }
    esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3(
            input_padded, padded_wd, padded_ht, channels, input_offset,
            stride_wd, stride_ht, filter_aligned, bias, out_data, out_wd,
            out_ht, out_offset, out_shift, out_mult,
            activation_min, activation_max);
}


typedef struct {
    const int16_t *input_data16;
    uint16_t input_wd, input_ht, channels;
    uint16_t pad_wd, pad_ht, stride_wd, stride_ht, ch_mult;
    const int16_t *filter_data16;
    const int32_t *bias;
    int8_t *out_data;
    uint16_t out_wd, out_ht;
    int32_t out_offset;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min, activation_max;
} dw_s16m8_3x3_mt_job_t;

static void dw_s16m8_3x3_mt_worker(void *p)
{
    const dw_s16m8_3x3_mt_job_t *j = (const dw_s16m8_3x3_mt_job_t *)p;
    esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3(
            j->input_data16, j->input_wd, j->input_ht, j->channels,
            j->pad_wd, j->pad_ht, j->stride_wd, j->stride_ht, j->ch_mult,
            j->filter_data16, j->bias, j->out_data, j->out_wd, j->out_ht,
            j->out_offset, j->out_shift, j->out_mult,
            j->activation_min, j->activation_max);
}

/* Output-row split of the s16 mult8 3x3 kernel. The top slice keeps the top
 * padding; the bottom slice starts at an interior input row (offset
 * h0*stride - pad_ht >= 0) with pad_ht = 0, so both slices see exactly the
 * rows the single call would use and write disjoint output rows. */
static void dw_s16m8_3x3_run_split(const int16_t *input_data16,
                                   uint16_t input_wd, uint16_t input_ht,
                                   uint16_t channels, uint16_t pad_wd,
                                   uint16_t pad_ht, uint16_t stride_wd,
                                   uint16_t stride_ht, uint16_t ch_mult,
                                   const int16_t *filter_data16,
                                   const int32_t *bias, int8_t *out_data,
                                   uint16_t out_wd, uint16_t out_ht,
                                   int32_t out_offset,
                                   const int32_t *out_shift,
                                   const int32_t *out_mult,
                                   int32_t activation_min,
                                   int32_t activation_max)
{
    const uint16_t h0 = out_ht / 2;
    const int32_t in_row_off = (int32_t)h0 * stride_ht - pad_ht;
    if (esp_nn_dual_core_active() && out_ht >= 8 && in_row_off >= 0) {
        dw_s16m8_3x3_mt_job_t job = {
            .input_data16 = input_data16, .input_wd = input_wd,
            .input_ht = input_ht, .channels = channels, .pad_wd = pad_wd,
            .pad_ht = pad_ht, .stride_wd = stride_wd, .stride_ht = stride_ht,
            .ch_mult = ch_mult, .filter_data16 = filter_data16, .bias = bias,
            .out_data = out_data, .out_wd = out_wd, .out_ht = h0,
            .out_offset = out_offset, .out_shift = out_shift,
            .out_mult = out_mult, .activation_min = activation_min,
            .activation_max = activation_max,
        };
        if (esp_nn_dual_core_run(dw_s16m8_3x3_mt_worker, &job)) {
            esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3(
                    input_data16 + in_row_off * input_wd * channels, input_wd,
                    (uint16_t)(input_ht - in_row_off), channels, pad_wd, 0,
                    stride_wd, stride_ht, ch_mult, filter_data16, bias,
                    out_data + (int32_t)h0 * out_wd * channels * ch_mult,
                    out_wd, (uint16_t)(out_ht - h0), out_offset, out_shift,
                    out_mult, activation_min, activation_max);
            esp_nn_dual_core_wait();
            return;
        }
    }
    esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3(
            input_data16, input_wd, input_ht, channels, pad_wd, pad_ht,
            stride_wd, stride_ht, ch_mult, filter_data16, bias, out_data,
            out_wd, out_ht, out_offset, out_shift, out_mult,
            activation_min, activation_max);
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

    /* MUST mirror the kernel's dispatch. Now that the (K,1) path declines
     * nothing on alignment grounds, its requirement is knowable here:
     * bias_adj (channels int32) + a K-row window + an aligned filter copy.
     * For a 384x1x128 layer that is ~1.3 kB against 99 kB for the int16 path
     * it replaces. */
    if (ESP_NN_DW_S8_KX1 && (ch_mult == 1) && (input_wd == 1) &&
            (filter_wd == 1) && (out_wd == 1) && (stride_wd == 1) &&
            (pad_wd == 0) && (channels % 16 == 0)) {
        return channels * 4 + 2 * filter_ht * channels + 64;
    }

    int filter_size = filter_wd * filter_ht * channels * ch_mult;
    int pad_width = 0, pad_height = 0;

    if ((ch_mult == 1) && (channels % 8 == 0)) {
        if(filter_wd == 3 && filter_ht == 3) {
            if (channels % 16 == 0) {
                /* Mirrors the kernel: leading pad from the caller, trailing
                 * pad from whatever the output extent still needs. Covers
                 * TFLite's asymmetric "SAME" shapes on this path. */
                pad_width = pad_wd + max(0, (out_wd - 1) * stride_wd + filter_wd
                                            - pad_wd - input_wd);
                pad_height = pad_ht + max(0, (out_ht - 1) * stride_ht + filter_ht
                                             - pad_ht - input_ht);
                if (pad_width || pad_height) {
                    int full_input = (input_wd + pad_width) * (input_ht + pad_height) * channels;
                    if (full_input <= 40 * 1024) {
                        return filter_size + full_input + 16;
                    } else {
                        /* Strip-tiled: filter + as many padded input rows as
                         * fit the same 40 KB budget the monolithic path uses,
                         * never fewer than filter_ht. The kernel derives its
                         * strip height from this size (minus its 16-byte
                         * alignment pad, hence +32 not +16), so reserving
                         * only filter_ht rows here would silently degrade it
                         * to one output row per assembly call. */
                        int row_bytes = (input_wd + pad_width) * channels;
                        int rows = (40 * 1024 - filter_size - 32) / row_bytes;
                        if (rows < filter_ht) {
                            rows = filter_ht;
                        }
                        if (rows > input_ht + pad_height) {
                            rows = input_ht + pad_height;
                        }
                        return filter_size + rows * row_bytes + 32;
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
                /* + channel-padded shift/mult/bias arrays (see the kernel) */
                int quant_pad_size = 3 * new_ch * (int)sizeof(int32_t);
                return new_filter_size + new_input_size + out_buf_size + quant_pad_size + 64;
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

void esp_nn_set_depthwise_conv_preferred_scratch_buf_esp32s3(void *buf,
                                                             size_t size)
{
    preferred_scratch_buffer = (uint8_t *)buf;
    preferred_scratch_size = size;
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


/**
 * int8 depthwise for a (filter_ht, 1) filter on a width-1 tensor.
 *
 * Reads int8 in place instead of converting the tensor to int16. Padding is
 * materialized only for the few edge output rows.
 */
static void esp_nn_depthwise_conv_s8_mult1_kx1(const int8_t *input,
                                              const uint16_t input_ht,
                                              const uint16_t channels,
                                              const int32_t input_offset,
                                              const uint16_t pad_ht,
                                              const uint16_t stride_ht,
                                              const int8_t *filter,
                                              const uint16_t filter_ht,
                                              const int32_t *bias,
                                              int8_t *out_data,
                                              const uint16_t out_ht,
                                              const int32_t out_offset,
                                              const int32_t *out_shift,
                                              const int32_t *out_mult,
                                              const int32_t activation_min,
                                              const int32_t activation_max)
{
    /* offset folded into the bias once per layer: sum((q+off)*w) =
     * sum(q*w) + off*sum(w), because ee.vmulas.s8.qacc has no offset operand.
     * A padded tap must then hold -offset so its share of off*sum(w) cancels,
     * which is what the edge buffer below is filled with. */
    int32_t *bias_adj = (int32_t *)((((uintptr_t)scratch_buffer) + 15) & ~(uintptr_t)15);
    int8_t *edge = (int8_t *)(bias_adj + channels);
    int8_t *filt_aligned = edge + filter_ht * channels;
    for (int ch = 0; ch < channels; ch++) {
        int32_t sum_w = 0;
        for (int k = 0; k < filter_ht; k++) {
            sum_w += filter[k * channels + ch];
        }
        bias_adj[ch] = (bias ? bias[ch] : 0) + input_offset * sum_w;
    }

    /* The kernel loads the filter with ee.vld.128, so it must be aligned.
     * TFLM hands out aligned tensors, so the copy (K*channels bytes, once
     * per layer) only runs for misaligned callers. */
    if (((uintptr_t)filter & 15) != 0) {
        memcpy(filt_aligned, filter, (size_t)filter_ht * channels);
        filter = filt_aligned;
    }

    /* A misaligned input is read through the window for every row, not just
     * the edges. In TFLM the arena hands out aligned tensors so this stays
     * cold, but it means the path never has to be declined. */
    const bool input_unaligned = (((uintptr_t)input) & 15) != 0;

    const int8_t pad_val = (int8_t)(-input_offset);
    /* The kernel does everything through the clamp; only the truncation to
     * int8 is left, one store per channel. Staging is hoisted: only the
     * per-channel-group mult/shift pointers change between calls. */
    int32_t acc[16] __attribute__((aligned(16)));
    dw_kx1_params_t pp = {
        .out = (int8_t *)acc,
        .out_offset = out_offset,
        .activation_min = activation_min,
        .activation_max = activation_max,
    };
    for (int out_y = 0; out_y < out_ht; out_y++) {
        const int base_y = out_y * stride_ht - pad_ht;
        const int8_t *rows = input + base_y * channels;
        int tap_stride = channels;
        if (input_unaligned || base_y < 0 || base_y + filter_ht > input_ht) {
            /* only the few edge rows need a materialized window */
            for (int k = 0; k < filter_ht; k++) {
                const int y = base_y + k;
                if (y < 0 || y >= input_ht) {
                    memset(edge + k * channels, pad_val, channels);
                } else {
                    memcpy(edge + k * channels, input + y * channels, channels);
                }
            }
            rows = edge;
        }
        for (int ch = 0; ch < channels; ch += 16) {
            pp.out_mult = out_mult + ch;
            pp.out_shift = out_shift + ch;
            esp_nn_dw_s8_kx1_esp32s3(rows + ch, tap_stride, filter_ht,
                                     filter + ch, bias_adj + ch, &pp);
            for (int i = 0; i < 16; i++) {
                out_data[out_y * channels + ch + i] = (int8_t)acc[i];
            }
        }
    }
}

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

    if (scratch_buffer == NULL) {
        printf("esp_nn_depthwise_conv error! scratch_buffer not set!\n");
        return;
    }
    int required_scratch = esp_nn_get_depthwise_conv_scratch_size_esp32s3(
            input_dims, filter_dims, output_dims, conv_params);
    int16_t *active_scratch = scratch_buffer;
    if (preferred_scratch_buffer != NULL &&
            (((uintptr_t)preferred_scratch_buffer & 15) == 0) &&
            required_scratch <= preferred_scratch_size) {
        active_scratch = (int16_t *)preferred_scratch_buffer;
    }
    int filter_size = filter_wd * filter_ht * channels * ch_mult;
    int align_len = 16 - (filter_size & 15);
    int input_size = input_wd * input_ht * channels;
    int16_t *filter_data16 = active_scratch;
    int16_t *input_data16 = active_scratch + filter_size + align_len;

    /* Width-1 tensor with a (K,1) filter: int8 in place. Alignment is
     * arranged inside the helper, not demanded of the caller. */
    if (ESP_NN_DW_S8_KX1 && (ch_mult == 1) && (input_wd == 1) &&
            (filter_wd == 1) &&
            (out_wd == 1) && (stride_wd == 1) && (pad_wd == 0) &&
            (channels % 16 == 0)) {
        esp_nn_depthwise_conv_s8_mult1_kx1(input_data, input_ht, channels,
                                           input_offset, pad_ht, stride_ht,
                                           filter_data, filter_ht, bias,
                                           out_data, out_ht, out_offset,
                                           out_shift, out_mult,
                                           activation_min, activation_max);
        return;
    }

    if ((ch_mult == 1) && (channels % 8 == 0)) {
        if ((filter_wd == 3) && (filter_ht == 3)) {
            if (channels % 16 == 0) {
                /* process in 8 bits with s8 padded assembly. Leading padding
                 * comes from the caller; trailing padding is whatever the
                 * output extent still needs, so TFLite's asymmetric "SAME"
                 * shapes stay on this path instead of falling through to the
                 * channel-padding one. */
                int8_t *filter_aligned = (int8_t *) active_scratch;
                int8_t *input_padded = (int8_t *) active_scratch + filter_size + align_len;
                const int pad_right = max(0, (out_wd - 1) * stride_wd + filter_wd
                                             - pad_wd - input_wd);
                const int pad_bottom = max(0, (out_ht - 1) * stride_ht + filter_ht
                                              - pad_ht - input_ht);
                const int padded_wd_full = input_wd + pad_wd + pad_right;
                const int padded_ht_full = input_ht + pad_ht + pad_bottom;

                if ((pad_wd | pad_ht | pad_right | pad_bottom) == 0) {
                    /* nothing to pad: run straight off the caller's buffer */
                    memcpy(filter_aligned, filter_data, filter_size);
                    dw3x3_run_split(input_data, input_wd, input_ht, channels,
                                    input_offset, stride_wd, stride_ht,
                                    filter_aligned, bias, out_data, out_wd, out_ht,
                                    out_offset, out_shift, out_mult,
                                    activation_min, activation_max);
                    return;
                }
                memcpy(filter_aligned, filter_data, filter_size);

                int padded_input_size = padded_wd_full * padded_ht_full * channels;
                if (padded_input_size <= 40 * 1024) {
                    /* Small enough — full padding, single assembly call */
                    esp_nn_aligned_s8_pad_asymmetric(input_data, input_padded,
                                                     input_wd, input_ht, channels,
                                                     -input_offset, pad_wd, pad_ht,
                                                     pad_right, pad_bottom);
                    dw3x3_run_split(input_padded, padded_wd_full,
                                    padded_ht_full, channels, input_offset,
                                    stride_wd, stride_ht, filter_aligned, bias,
                                    out_data, out_wd, out_ht, out_offset, out_shift,
                                    out_mult, activation_min, activation_max);
                } else {
                    /* Large input: strip-tiled processing. Copy a strip of
                     * input rows once and produce several output rows per
                     * assembly call; per-row tiling would re-copy every
                     * input row filter_ht times from the (PSRAM) input.
                     * With the dual-core worker active, each core runs its
                     * own half of the output rows with its own tile. */
                    int padded_wd = padded_wd_full;
                    int row_bytes = padded_wd * channels;
                    int avail = (active_scratch == (int16_t *)preferred_scratch_buffer)
                                ? (int)preferred_scratch_size : required_scratch;
                    dw3x3_strip_job_t job = {
                        .input_data = input_data, .input_wd = input_wd,
                        .input_ht = input_ht, .channels = channels,
                        .input_offset = input_offset, .pad_wd = pad_wd,
                        .pad_ht = pad_ht, .pad_right = pad_right,
                        .pad_bottom = pad_bottom, .stride_wd = stride_wd,
                        .stride_ht = stride_ht, .filter_aligned = filter_aligned,
                        .bias = bias, .out_data = out_data, .out_wd = out_wd,
                        .out_y_begin = 0, .out_y_end = out_ht,
                        .out_offset = out_offset, .out_shift = out_shift,
                        .out_mult = out_mult, .activation_min = activation_min,
                        .activation_max = activation_max,
                        .tile_buf = input_padded,
                        .tile_bytes = avail - filter_size - align_len - 16,
                    };
                    int done = 0;
                    if (esp_nn_dual_core_active() && out_ht >= 4) {
                        int want = 3 * row_bytes + 16;
                        if (want < 24 * 1024 && job.tile_bytes > want) {
                            want = (job.tile_bytes < 24 * 1024) ? job.tile_bytes : 24 * 1024;
                        }
                        int8_t *wtile = (int8_t *)esp_nn_dual_core_scratch(want);
                        if (wtile != NULL && want >= 3 * row_bytes) {
                            dw3x3_strip_job_t wjob = job;
                            wjob.out_y_end = out_ht / 2;
                            wjob.tile_buf = wtile;
                            wjob.tile_bytes = want;
                            if (esp_nn_dual_core_run(dw3x3_strip_mt_worker, &wjob)) {
                                job.out_y_begin = out_ht / 2;
                                dw3x3_strip_rows(&job);
                                esp_nn_dual_core_wait();
                                done = 1;
                            }
                        }
                    }
                    if (!done) {
                        dw3x3_strip_rows(&job);
                    }
                }
            } else if (channels >= 12) {
                /* channels % 8 == 0, not % 16, channels >= 12: pad to 16 is worthwhile
                 * (overhead <= 33%). For ch=8, padding to 16 doubles data — use s16 instead */
                int new_ch = (channels + 15) & ~15;
                int8_t pad_val = (int8_t)(-input_offset);

                /* Pad filter: 3x3 x new_ch */
                int new_filter_size = 9 * new_ch;
                int8_t *filter_padded = (int8_t *) active_scratch;
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

                /* Channel-padded quant arrays: in scratch after the output
                 * buffer (getter reserves 3 * new_ch int32), not on the stack. */
                int32_t *shift_pad = (int32_t *)(out_padded + out_wd * out_ht * new_ch);
                int32_t *mult_pad = shift_pad + new_ch;
                int32_t *bias_pad = mult_pad + new_ch;
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
                /* Row tiles to limit cache pressure: convert only the rows
                 * each output row needs, in SIMD. A row is reconverted per
                 * output row; caching across rows needs a rotated window the
                 * kernel does not accept. */
                int16_t *tile_buf = input_data16; /* reuse scratch for tile */
                const int row_len = input_wd * channels;

                for (int out_row = 0; out_row < out_ht; out_row++) {
                    int in_row_start = out_row * stride_ht - pad_ht;
                    int in_row_end = in_row_start + filter_ht;

                    int16_t *dst = tile_buf;
                    for (int r = in_row_start; r < in_row_end; r++) {
                        if (r < 0 || r >= input_ht) {
                            /* Padding row. Valid rows are stored as
                             * (q + in_offset), so a real zero is 0 here. */
                            memset(dst, 0, sizeof(int16_t) * (size_t)row_len);
                        } else {
                            const int8_t *src = input_data + r * row_len;
                            if ((((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
                                esp_nn_aligned_s8_to_s16_with_offset_esp32s3(
                                        src, dst, row_len, input_offset);
                            } else if (r > 0 && r + 1 < input_ht &&
                                       (((uintptr_t)dst & 15) == 0)) {
                                /* row_len is the BYTE stride of an int8 row:
                                 * at channels % 16 != 0 alternate rows start
                                 * 8-byte aligned and the aligned converter's
                                 * 128-bit loads need 16. Interior rows have
                                 * neighbours covering the QUP window's
                                 * over-read. Cases 21/22 cover this path. */
                                esp_nn_s8_to_s16_with_offset_row_esp32s3(
                                        src, dst, row_len, input_offset);
                            } else {
                                /* first/last row: no slack for the QUP window */
                                for (int i = 0; i < row_len; i++) {
                                    dst[i] = (int16_t)(src[i] + input_offset);
                                }
                            }
                        }
                        dst += row_len;
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
        int16_t *padded_filter_data16 = active_scratch;
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
            dw_s16m8_3x3_run_split(input_data16, input_wd, input_ht, channels,
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
