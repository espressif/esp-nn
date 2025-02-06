/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file        Header definitions to include for esp_nn optimized functions for
 *              the ESP32-P4 platform
 */

#pragma once

#include "esp_nn_defs.h"
#include "esp_nn_ansi_headers.h"

/**
 * @brief       2d - convolution channelwise
 *
 * @note        operation: result += (input + offset) * filter
 *
 *              inputs type: int8_t, output: int8_t
 *              input offsets: although int32_t, they are contained in 8 bits [-128, 127]
 */
void esp_nn_conv_s8_esp32p4(const data_dims_t *input_dims,
                            const int8_t *input_data,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *output_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data);

int esp_nn_get_conv_scratch_size_esp32p4(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params);
void esp_nn_set_conv_scratch_buf_esp32p4(const void *buf);

/********************** function defines ***************************/



#define esp_nn_add_elementwise_s8 esp_nn_add_elementwise_s8_ansi
#define esp_nn_mul_elementwise_s8 esp_nn_mul_elementwise_s8_ansi

#define esp_nn_depthwise_conv_s8 esp_nn_depthwise_conv_s8_opt

#define esp_nn_conv_s8 esp_nn_conv_s8_esp32p4

#define esp_nn_get_conv_scratch_size esp_nn_get_conv_scratch_size_esp32p4
#define esp_nn_set_conv_scratch_buf esp_nn_set_conv_scratch_buf_esp32p4

#define esp_nn_get_depthwise_conv_scratch_size esp_nn_get_depthwise_conv_scratch_size_opt
#define esp_nn_set_depthwise_conv_scratch_buf esp_nn_set_depthwise_conv_scratch_buf_opt

#define esp_nn_relu6_s8 esp_nn_relu6_s8_ansi

#define esp_nn_avg_pool_s8 esp_nn_avg_pool_s8_ansi
#define esp_nn_max_pool_s8 esp_nn_max_pool_s8_ansi

#define esp_nn_fully_connected_s8 esp_nn_fully_connected_s8_ansi
#define esp_nn_fully_connected_per_ch_s8 esp_nn_fully_connected_per_ch_s8_ansi

#define esp_nn_get_softmax_scratch_size esp_nn_get_softmax_scratch_size_opt
#define esp_nn_set_softmax_scratch_buf esp_nn_set_softmax_scratch_buf_opt
#define esp_nn_softmax_s8 esp_nn_softmax_s8_opt
