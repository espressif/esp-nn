/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Private to the fully connected sources. Not part of esp_nn.h. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sdkconfig.h"

#if defined(CONFIG_NN_OPTIMIZED) && (defined(CONFIG_IDF_TARGET_ESP32S3) || \
        defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
#define ESP_NN_FC_BATCH_VIA_CONV 1

/* Install buf as the convolution scratch pointer for one call, and remember
 * the pointer that was already there. The S3 preferred buffer stays set. */
#if defined(CONFIG_IDF_TARGET_ESP32S3)
void esp_nn_conv_scratch_borrow_esp32s3(void *buf, void **scratch);
void esp_nn_conv_scratch_restore_esp32s3(void *scratch);
#define esp_nn_conv_scratch_borrow esp_nn_conv_scratch_borrow_esp32s3
#define esp_nn_conv_scratch_restore esp_nn_conv_scratch_restore_esp32s3
#else
void esp_nn_conv_scratch_borrow_riscv_pie(void *buf, void **scratch);
void esp_nn_conv_scratch_restore_riscv_pie(void *scratch);
#define esp_nn_conv_scratch_borrow esp_nn_conv_scratch_borrow_riscv_pie
#define esp_nn_conv_scratch_restore esp_nn_conv_scratch_restore_riscv_pie
#endif

/* Returns 1 when the batch was evaluated as a 1x1 convolution.
 * A nonzero weight zero-point, a shape below the per-chip performance floor,
 * batches above UINT16_MAX, an unaligned input or filter (or output on
 * S3), or a scratch buffer that is missing, unaligned or smaller than this
 * 1x1 shape needs returns 0 and leaves out_data untouched. The convolution
 * scratch pointer is not consulted. */
int esp_nn_fc_per_ch_s8_batch_via_conv(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches,
                                    void *scratch,
                                    size_t scratch_size);
#endif
