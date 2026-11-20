/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sdkconfig.h"
#include <stddef.h>
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include <esp_nn_fully_connected_mt.h>

#if defined(CONFIG_NN_OPTIMIZED) && (defined(CONFIG_IDF_TARGET_ESP32S3) || \
    defined(CONFIG_IDF_TARGET_ESP32P4))

#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define fc_single esp_nn_fully_connected_s8_esp32s3
#define fc_per_ch_single esp_nn_fully_connected_per_ch_s8_esp32s3
#else
#define fc_single esp_nn_fully_connected_s8_riscv_pie
#define fc_per_ch_single esp_nn_fully_connected_per_ch_s8_riscv_pie
#endif

#if ESP_NN_DUAL_CORE_SUPPORTED
typedef struct {
    const int8_t *input, *filter;
    const int32_t *bias, *shifts, *mults;
    int8_t *output;
    int32_t input_offset, filter_offset, output_offset;
    int32_t shift, mult, activation_min, activation_max;
    uint16_t rows, outputs, split;
    bool per_channel;
} fc_job_t;

/* Use the original kernels directly: no recursive dispatch, convolution,
 * tensor padding or shared scratch. Each core computes complete outputs. */
static void fc_slice(const fc_job_t *c, uint16_t start, uint16_t count)
{
    const int8_t *filter = c->filter + (size_t) start * c->rows;
    const int32_t *bias = c->bias ? c->bias + start : NULL;
    if (c->per_channel) {
        fc_per_ch_single(c->input, c->input_offset, c->rows, filter,
            c->filter_offset, bias, c->output + start, count, c->output_offset,
            c->shifts + start, c->mults + start,
            c->activation_min, c->activation_max);
    } else {
        fc_single(c->input, c->input_offset, c->rows, filter,
            c->filter_offset, bias, c->output + start, count, c->output_offset,
            c->shift, c->mult, c->activation_min, c->activation_max);
    }
}

static void fc_worker(void *arg)
{
    const fc_job_t *c = arg;
    fc_slice(c, 0, c->split);
}

/* The caller already owns the worker. No output is written on refusal. */
static bool fc_execute(fc_job_t *c)
{
    /* Round only the split, never the tensor dimensions. */
    c->split = (c->outputs / 2) & ~7;
    if (!esp_nn_dual_core_run(fc_worker, c)) {
        esp_nn_dual_core_end();
        return false;
    }
    fc_slice(c, c->split, c->outputs - c->split);
    esp_nn_dual_core_wait();
    esp_nn_dual_core_end();
    return true;
}

void __attribute__((noinline)) esp_nn_fc_split_s8(
    const int8_t *input_data, int32_t input_offset, uint16_t row_len,
    const int8_t *filter_data, int32_t filter_offset, const int32_t *bias,
    int8_t *out_data, uint16_t out_channels, int32_t out_offset,
    int32_t out_shift, int32_t out_mult, int32_t activation_min, int32_t activation_max)
{
    if (!esp_nn_dual_core_begin()) {
        fc_single(input_data, input_offset, row_len, filter_data, filter_offset,
                bias, out_data, out_channels, out_offset, out_shift, out_mult,
                activation_min, activation_max);
        return;
    }
    fc_job_t c = {
        .input = input_data, .filter = filter_data, .bias = bias,
        .output = out_data, .input_offset = input_offset,
        .filter_offset = filter_offset, .output_offset = out_offset,
        .rows = row_len, .outputs = out_channels,
        .activation_min = activation_min, .activation_max = activation_max,
        .shift = out_shift, .mult = out_mult, .per_channel = false,
    };
    if (!fc_execute(&c)) {
        fc_single(input_data, input_offset, row_len, filter_data, filter_offset,
                bias, out_data, out_channels, out_offset, out_shift, out_mult,
                activation_min, activation_max);
    }
}

void __attribute__((noinline)) esp_nn_fc_split_per_ch_s8(
    const int8_t *input_data, int32_t input_offset, uint16_t row_len,
    const int8_t *filter_data, int32_t filter_offset, const int32_t *bias,
    int8_t *out_data, uint16_t out_channels, int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult, int32_t activation_min, int32_t activation_max)
{
    if (!esp_nn_dual_core_begin()) {
        fc_per_ch_single(input_data, input_offset, row_len, filter_data,
                filter_offset, bias, out_data, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max);
        return;
    }
    fc_job_t c = {
        .input = input_data, .filter = filter_data, .bias = bias,
        .output = out_data, .input_offset = input_offset,
        .filter_offset = filter_offset, .output_offset = out_offset,
        .rows = row_len, .outputs = out_channels,
        .activation_min = activation_min, .activation_max = activation_max,
        .shifts = out_shift, .mults = out_mult, .per_channel = true,
    };
    if (!fc_execute(&c)) {
        fc_per_ch_single(input_data, input_offset, row_len, filter_data,
                filter_offset, bias, out_data, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max);
    }
}

#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
#endif
