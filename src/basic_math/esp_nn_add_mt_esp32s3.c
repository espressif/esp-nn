/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Dual-core front for the elementwise s8 add: splits the array at a
 * 16-byte boundary so both halves keep the SIMD alignment fast path and
 * every output byte is written by exactly one core (bit-identical to the
 * single call). Falls through to the assembly kernel when the worker is
 * not running or the array is small. */

#include <stdint.h>

#include <esp_nn_esp32s3.h>
#include <esp_nn_multicore.h>

typedef struct {
    const int8_t *input1_data;
    const int8_t *input2_data;
    int32_t input1_offset, input2_offset;
    int32_t input1_mult, input2_mult;
    int32_t input1_shift, input2_shift;
    int32_t left_shift;
    int8_t *output;
    int32_t out_offset, out_mult, out_shift;
    int32_t activation_min, activation_max;
    int32_t size;
} add_s8_mt_job_t;

static void add_s8_mt_worker(void *p)
{
    const add_s8_mt_job_t *j = (const add_s8_mt_job_t *)p;
    esp_nn_add_elementwise_s8_esp32s3(
            j->input1_data, j->input2_data, j->input1_offset, j->input2_offset,
            j->input1_mult, j->input2_mult, j->input1_shift, j->input2_shift,
            j->left_shift, j->output, j->out_offset, j->out_mult, j->out_shift,
            j->activation_min, j->activation_max, j->size);
}

void esp_nn_add_elementwise_s8_mt_esp32s3(const int8_t *input1_data,
                                          const int8_t *input2_data,
                                          const int32_t input1_offset,
                                          const int32_t input2_offset,
                                          const int32_t input1_mult,
                                          const int32_t input2_mult,
                                          const int32_t input1_shift,
                                          const int32_t input2_shift,
                                          const int32_t left_shift,
                                          int8_t *output,
                                          const int32_t out_offset,
                                          const int32_t out_mult,
                                          const int32_t out_shift,
                                          const int32_t activation_min,
                                          const int32_t activation_max,
                                          const int32_t size)
{
    if (esp_nn_dual_core_active() && size >= 8192) {
        const int32_t half = (size / 2) & ~15;
        add_s8_mt_job_t job = {
            .input1_data = input1_data, .input2_data = input2_data,
            .input1_offset = input1_offset, .input2_offset = input2_offset,
            .input1_mult = input1_mult, .input2_mult = input2_mult,
            .input1_shift = input1_shift, .input2_shift = input2_shift,
            .left_shift = left_shift, .output = output,
            .out_offset = out_offset, .out_mult = out_mult,
            .out_shift = out_shift, .activation_min = activation_min,
            .activation_max = activation_max, .size = half,
        };
        if (half > 0 && esp_nn_dual_core_run(add_s8_mt_worker, &job)) {
            esp_nn_add_elementwise_s8_esp32s3(
                    input1_data + half, input2_data + half, input1_offset,
                    input2_offset, input1_mult, input2_mult, input1_shift,
                    input2_shift, left_shift, output + half, out_offset,
                    out_mult, out_shift, activation_min, activation_max,
                    size - half);
            esp_nn_dual_core_wait();
            return;
        }
    }
    esp_nn_add_elementwise_s8_esp32s3(
            input1_data, input2_data, input1_offset, input2_offset,
            input1_mult, input2_mult, input1_shift, input2_shift, left_shift,
            output, out_offset, out_mult, out_shift,
            activation_min, activation_max, size);
}
