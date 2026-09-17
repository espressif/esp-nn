/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_nn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#if !defined(CONFIG_FREERTOS_UNICORE) && \
    (CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4)
#define ESP_NN_DUAL_CORE_SUPPORTED 1
#endif
#endif
#ifndef ESP_NN_DUAL_CORE_SUPPORTED
#define ESP_NN_DUAL_CORE_SUPPORTED 0
#endif

/**
 * Opt in to dual-core kernel splitting. Spawns a worker task pinned to the
 * other core; kernels that support it then split their output range in two.
 * Results are bit-identical to single-core execution (disjoint outputs, same
 * arithmetic). Call once from the core that will run inference.
 */
void esp_nn_dual_core_enable(void);

/** True once the worker task exists. */
bool esp_nn_dual_core_active(void);

/** Dispatch fn(arg) to the worker core. Pair every successful call with
 * esp_nn_dual_core_wait(). Returns false when the worker is not running. */
bool esp_nn_dual_core_run(void (*fn)(void *), void *arg);

/** Block until the dispatched job completes. */
void esp_nn_dual_core_wait(void);

/* Dual-core output-row split around the reference conv (grouped-conv and
 * other fallback shapes); bit-identical to a single esp_nn_conv_s8_ansi call. */
void esp_nn_conv_s8_ansi_mt_split(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const conv_params_t *conv_params,
                                  const quant_data_t *quant_data);

/** Kernel-private scratch for the worker's slice (grown on demand,
 * internal-SRAM first). Returns NULL if allocation fails. */
void *esp_nn_dual_core_scratch(int size);

#ifdef __cplusplus
}
#endif
