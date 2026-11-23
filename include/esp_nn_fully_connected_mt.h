/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <stdint.h>
#include <esp_nn_multicore.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(CONFIG_NN_OPTIMIZED) && (defined(CONFIG_IDF_TARGET_ESP32S3) || \
    defined(CONFIG_IDF_TARGET_ESP32P4))
/* Batch-1 direct FC. Where dual-core FC is supported, the public names are
 * static inline gates in the target header. At or above the floor, with the
 * worker claimed, each core runs the original kernel on its own outputs and
 * the gate does not run again. Below the floor the gate does not claim the
 * worker. A missing or busy worker stays on the original kernel. With
 * esp_nn_dual_core_enable(), split when K >= 24 && N >= 192 on S3 and
 * K >= 384 && N >= 512 on P4.
 * No upper bounds or dimension divisibility requirements. S31 retains its original
 * public APIs and is not routed through this gate.
 * Both zero and nonzero input offsets are supported. The scratch-enabled
 * 1x1 route separately requires a zero filter offset, not a zero input offset.
 * S3/P4 no-scratch batch APIs also use this path for batches == 1.
 * Scratch-enabled APIs retain production 1x1 eligibility for all batches.
 * When 1x1 is rejected, a batch of 1 reaches this path from the S3 and P4
 * batch kernels. Other no-scratch batches keep their original dispatch.
 * Existing buffer/alignment requirements of the underlying FC still apply.
 */
#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define ESP_NN_FC_DUAL_MIN_ROW 24
#define ESP_NN_FC_DUAL_MIN_OUT 192
#elif defined(CONFIG_IDF_TARGET_ESP32P4)
/* P4 gate measured on rev v1.3 and checked on rev v3. S31 retains its original dispatch. */
#define ESP_NN_FC_DUAL_MIN_ROW 384
#define ESP_NN_FC_DUAL_MIN_OUT 512
#endif

#if ESP_NN_DUAL_CORE_SUPPORTED
/* Caller already passed the floor. Claims the worker and splits, or runs
 * the single-core kernel when the worker is missing, busy, or the launch
 * is refused. */
void esp_nn_fc_split_s8(
    const int8_t *input_data, int32_t input_offset, uint16_t row_len,
    const int8_t *filter_data, int32_t filter_offset, const int32_t *bias,
    int8_t *out_data, uint16_t out_channels, int32_t out_offset,
    int32_t out_shift, int32_t out_mult, int32_t activation_min, int32_t activation_max);
void esp_nn_fc_split_per_ch_s8(
    const int8_t *input_data, int32_t input_offset, uint16_t row_len,
    const int8_t *filter_data, int32_t filter_offset, const int32_t *bias,
    int8_t *out_data, uint16_t out_channels, int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult, int32_t activation_min, int32_t activation_max);
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */

#endif /* optimized S3/P4 */

#ifdef __cplusplus
}
#endif
