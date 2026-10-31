/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Private to the fully connected sources. Not part of esp_nn.h. */
#pragma once

#include <stdint.h>

#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
int esp_nn_conv_scratch_fits_riscv_pie(int bytes);
#define esp_nn_conv_scratch_fits esp_nn_conv_scratch_fits_riscv_pie
#elif defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
int esp_nn_conv_scratch_fits_esp32s3(int bytes);
#define esp_nn_conv_scratch_fits esp_nn_conv_scratch_fits_esp32s3
#else
int esp_nn_conv_scratch_fits_ansi(int bytes);
#define esp_nn_conv_scratch_fits esp_nn_conv_scratch_fits_ansi
#endif

/* Returns 1 when the batch was evaluated as a 1x1 convolution.
 * A nonzero weight zero-point, a small shape, an unaligned input or filter,
 * or a conv scratch buffer that is missing or smaller than this 1x1 shape
 * returns 0 and leaves out_data untouched. */
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
                                    const int32_t batches);
