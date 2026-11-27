/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file        Header definitions to include for esp_nn optimisations on ESP32
 *              Conv and depthwise run on the MAC16 unit. Every other kernel
 *              uses the generic optimisations.
 */

#pragma once

#include "esp_nn_defs.h"
#include "esp_nn_generic_opt.h"

/**
 * @brief       2d-convolution channelwise, on MAC16
 *
 * @note        These run the generic kernel (esp_nn_conv_s8_opt):
 *              - grouped conv, and convs without input or output channels;
 *              - filter windows too large for the scratch budget;
 *              - calls without a scratch buffer;
 *              - builds with CONFIG_NN_SKIP_NUDGE;
 *              - on builds with CONFIG_SPIRAM_CACHE_WORKAROUND, calls with
 *                the input, filter, output, scratch, bias, mult or shift
 *                in PSRAM.
 *              Calls with an output window that has no input run the
 *              reference kernel.
 */
void esp_nn_conv_s8_esp32(const data_dims_t *input_dims,
                          const int8_t *input_data,
                          const data_dims_t *filter_dims,
                          const int8_t *filter_data,
                          const int32_t *bias,
                          const data_dims_t *output_dims,
                          int8_t *out_data,
                          const conv_params_t *conv_params,
                          const quant_data_t *quant_data);

/* Pure C: also builds on a host (esp_nn_scratch_sizes_esp32.c). */
int esp_nn_get_conv_scratch_size_esp32(const data_dims_t *input_dims,
                                       const data_dims_t *filter_dims,
                                       const data_dims_t *output_dims,
                                       const conv_params_t *conv_params);
void esp_nn_set_conv_scratch_buf_esp32(const void *buf);

/**
 * @brief       depthwise convolution per channel, 3x3 filters on MAC16
 *
 * @note        These run the generic kernel (esp_nn_depthwise_conv_s8_opt):
 *              - other filters, and shapes whose band does not fit the
 *                scratch budget;
 *              - calls without a scratch buffer;
 *              - builds with CONFIG_NN_SKIP_NUDGE;
 *              - on builds with CONFIG_SPIRAM_CACHE_WORKAROUND, calls with
 *                the input, filter, output or scratch in PSRAM.
 */
void esp_nn_depthwise_conv_s8_esp32(const data_dims_t *input_dims,
                                    const int8_t *input_data,
                                    const data_dims_t *filter_dims,
                                    const int8_t *filter_data,
                                    const int32_t *bias,
                                    const data_dims_t *output_dims,
                                    int8_t *out_data,
                                    const dw_conv_params_t *conv_params,
                                    const quant_data_t *quant_data);

/* Pure C: also builds on a host (esp_nn_scratch_sizes_esp32.c). */
int esp_nn_get_depthwise_conv_scratch_size_esp32(const data_dims_t *input_dims,
                                                 const data_dims_t *filter_dims,
                                                 const data_dims_t *output_dims,
                                                 const dw_conv_params_t *conv_params);
void esp_nn_set_depthwise_conv_scratch_buf_esp32(const void *buf);

#undef esp_nn_conv_s8
#undef esp_nn_get_conv_scratch_size
#undef esp_nn_set_conv_scratch_buf
#undef esp_nn_depthwise_conv_s8
#undef esp_nn_get_depthwise_conv_scratch_size
#undef esp_nn_set_depthwise_conv_scratch_buf

#define esp_nn_conv_s8 esp_nn_conv_s8_esp32
#define esp_nn_get_conv_scratch_size esp_nn_get_conv_scratch_size_esp32
#define esp_nn_set_conv_scratch_buf esp_nn_set_conv_scratch_buf_esp32

#define esp_nn_depthwise_conv_s8 esp_nn_depthwise_conv_s8_esp32
#define esp_nn_get_depthwise_conv_scratch_size esp_nn_get_depthwise_conv_scratch_size_esp32
#define esp_nn_set_depthwise_conv_scratch_buf esp_nn_set_depthwise_conv_scratch_buf_esp32
