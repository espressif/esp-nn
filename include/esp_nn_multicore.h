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
#if !defined(CONFIG_FREERTOS_UNICORE) && defined(CONFIG_NN_OPTIMIZED) && \
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
 * arithmetic). Call once from the core that will run inference, early: the
 * worker's stack (CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE bytes) must come from
 * SIMD-capable memory: internal RAM first, or PSRAM when
 * CONFIG_FREERTOS_TASK_CREATE_ALLOW_EXT_MEM allows task stacks there; only PSRAM
 * with CONFIG_NN_DUAL_CORE_WORKER_STACK_PSRAM. If none is left, esp-nn stays
 * single core; esp_nn_dual_core_active() reports which.
 *
 * On ESP32-P4:
 * - This call pins the calling task to the core it runs on (the first PIE
 *   instruction in a task does that), and the worker goes to the other core.
 *   Run inference from this task, or from another task pinned to this core.
 * - Every task that runs esp-nn kernels needs its stack in SIMD-capable
 *   memory, not LP RAM, like the worker: the kernels and IDF's PIE context
 *   save store 128-bit vectors to it. For example, create it with
 *   xTaskCreatePinnedToCoreWithCaps(..., MALLOC_CAP_SIMD | MALLOC_CAP_8BIT)
 *   (MALLOC_CAP_SIMD needs IDF v5.3.3, v5.4.1 or v5.5; use MALLOC_CAP_DMA
 *   before).
 */
void esp_nn_dual_core_enable(void);

/**
 * Stop the worker and free its stack, TCB, scratch and semaphores; esp-nn
 * runs single core afterwards and esp_nn_dual_core_enable() may start it
 * again. Not thread-safe: call it only while no other task is in esp-nn,
 * including esp_nn_dual_core_*() calls and another disable().
 */
void esp_nn_dual_core_disable(void);

/** True once the worker task exists. */
bool esp_nn_dual_core_active(void);

/**
 * Claim the worker for one split kernel. Never blocks: returns false when
 * dual core is off, the caller runs on the worker's core, or another task
 * holds the worker; the kernel then runs single core. Only the worker is
 * arbitrated: kernels whose scratch is set with esp_nn_set_*_scratch_buf()
 * (conv, depthwise conv, softmax, mean, hard swish) share it, so those must
 * not run from two tasks at once. Pair every true return with
 * esp_nn_dual_core_end(), on every path. The worker scratch
 * (esp_nn_dual_core_scratch()) belongs to the holder.
 */
bool esp_nn_dual_core_begin(void);

/** Release the claim taken by esp_nn_dual_core_begin(). */
void esp_nn_dual_core_end(void);

/** Dispatch fn(arg) to the worker core; the caller must hold the claim
 * (esp_nn_dual_core_begin()). Pair every successful call with
 * esp_nn_dual_core_wait(). Returns false when the worker is not running;
 * asserts when the caller does not hold the claim. */
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

typedef void (*esp_nn_conv_s8_target_fn_t)(const data_dims_t *,
                                           const int8_t *,
                                           const data_dims_t *,
                                           const int8_t *,
                                           const int32_t *,
                                           const data_dims_t *,
                                           int8_t *,
                                           const conv_params_t *,
                                           const quant_data_t *);

/* Grouped conv as G standard convs through the target's optimized path
 * (repack input slice, contiguous group filters, scatter staged output).
 * Bit-identical to the reference grouped loop. Returns false when shapes
 * or scratch don't fit. */
bool esp_nn_conv_s8_grouped_repack(esp_nn_conv_s8_target_fn_t conv_fn,
                                   const data_dims_t *input_dims,
                                   const int8_t *input_data,
                                   const data_dims_t *filter_dims,
                                   const int8_t *filter_data,
                                   const int32_t *bias,
                                   const data_dims_t *output_dims,
                                   int8_t *out_data,
                                   const conv_params_t *conv_params,
                                   const quant_data_t *quant_data,
                                   void *scratch, int scratch_size,
                                   int inner_scratch_size);

/** Kernel-private scratch for the worker's slice (grown on demand,
 * internal-SRAM first). Returns NULL if allocation fails. */
void *esp_nn_dual_core_scratch(int size);

#ifdef __cplusplus
}
#endif
