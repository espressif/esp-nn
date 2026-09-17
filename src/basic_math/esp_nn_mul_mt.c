/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Dual-core front for the elementwise s8 multiply: splits the array at a
 * 16-byte boundary so both halves keep the SIMD alignment fast path and
 * every output byte is written by exactly one core (bit-identical to the
 * single call). Falls through to the target kernel when the worker is not
 * running or the array is small. */

#include <stdint.h>

#include <sdkconfig.h>
#include <esp_nn_multicore.h>

#if CONFIG_IDF_TARGET_ESP32S3
#include <esp_nn_esp32s3.h>
#define MUL_S8_TARGET esp_nn_mul_elementwise_s8_esp32s3
#elif CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_ESP32S31
#include <esp_nn_riscv_pie.h>
#define MUL_S8_TARGET esp_nn_mul_elementwise_s8_riscv_pie
#endif

#ifdef MUL_S8_TARGET

typedef struct {
    const int8_t *input1_data;
    const int8_t *input2_data;
    int32_t input1_offset, input2_offset;
    int8_t *output;
    int32_t out_offset, out_mult, out_shift;
    int32_t activation_min, activation_max;
    int32_t size;
} mul_s8_mt_job_t;

static void mul_s8_mt_worker(void *p)
{
    const mul_s8_mt_job_t *j = (const mul_s8_mt_job_t *)p;
    MUL_S8_TARGET(j->input1_data, j->input2_data, j->input1_offset,
                  j->input2_offset, j->output, j->out_offset, j->out_mult,
                  j->out_shift, j->activation_min, j->activation_max, j->size);
}

void esp_nn_mul_elementwise_s8_mt(const int8_t *input1_data,
                                  const int8_t *input2_data,
                                  const int32_t input1_offset,
                                  const int32_t input2_offset,
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
        mul_s8_mt_job_t job = {
            .input1_data = input1_data, .input2_data = input2_data,
            .input1_offset = input1_offset, .input2_offset = input2_offset,
            .output = output, .out_offset = out_offset, .out_mult = out_mult,
            .out_shift = out_shift, .activation_min = activation_min,
            .activation_max = activation_max, .size = half,
        };
        if (half > 0 && esp_nn_dual_core_run(mul_s8_mt_worker, &job)) {
            MUL_S8_TARGET(input1_data + half, input2_data + half,
                          input1_offset, input2_offset, output + half,
                          out_offset, out_mult, out_shift,
                          activation_min, activation_max, size - half);
            esp_nn_dual_core_wait();
            return;
        }
    }
    MUL_S8_TARGET(input1_data, input2_data, input1_offset, input2_offset,
                  output, out_offset, out_mult, out_shift,
                  activation_min, activation_max, size);
}

#endif /* MUL_S8_TARGET */
