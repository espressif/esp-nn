// Copyright 2020-2026 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(CONFIG_NN_OPTIMIZED)
// select apt optimisations
// ESP32-S31 shares the P4 PIE/SIMD ISA, so it reuses the ESP32-P4 kernels
#if defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31)
#define ARCH_ESP_RISCV_PIE 1
#endif
#ifdef CONFIG_IDF_TARGET_ESP32S3
#define ARCH_ESP32_S3 1
#endif
#ifdef CONFIG_IDF_TARGET_ESP32
#define ARCH_ESP32 1
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* reference kernels included by default */
#include "esp_nn_ansi_headers.h"

#if defined(CONFIG_NN_OPTIMIZED)
#if defined(ARCH_ESP_RISCV_PIE)
#include "esp_nn_riscv_pie.h"
#elif defined(ARCH_ESP32_S3)
#include "esp_nn_esp32s3.h"
#else // for other platforms use generic optimisations
#include "esp_nn_generic_opt.h"
#endif // #if defined(ARCH_ESP32_S3)
#else
#include "esp_nn_ansi_c.h"
#endif

/* Optional per-kernel APIs. An arch header that has an optimised
 * implementation aliases the public name to it above; every other target
 * falls back to the portable ANSI version here, so the API and its feature
 * flag are present on all targets and consumers need only one code path. */
#ifndef esp_nn_get_mean_scratch_size
#define esp_nn_get_mean_scratch_size esp_nn_get_mean_scratch_size_ansi
#define esp_nn_set_mean_scratch_buf esp_nn_set_mean_scratch_buf_ansi
#endif
#ifndef esp_nn_fully_connected_per_ch_s8_batch
#define esp_nn_fully_connected_per_ch_s8_batch esp_nn_fully_connected_per_ch_s8_batch_ansi
#endif
/* Feature flag: consumers can detect the batched per-channel FC API. */
#define ESP_NN_HAS_FC_S8_BATCH 1

#ifdef __cplusplus
}
#endif
