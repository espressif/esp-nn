/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file        Header definitions to include for ANSI C versions.
 *              These are just typedefs to pick up ANSI versions.
 */

#pragma once

#include "esp_nn_defs.h"
#include "esp_nn_ansi_headers.h"

#define esp_nn_add_elementwise_s8 esp_nn_add_elementwise_s8_ansi
#define esp_nn_mul_elementwise_s8 esp_nn_mul_elementwise_s8_ansi

#define esp_nn_depthwise_conv_s8 esp_nn_depthwise_conv_s8_ansi

#define esp_nn_conv_s8 esp_nn_conv_s8_ansi

#define esp_nn_get_conv_scratch_size esp_nn_get_conv_scratch_size_ansi
#define esp_nn_set_conv_scratch_buf esp_nn_set_conv_scratch_buf_ansi

#define esp_nn_get_depthwise_conv_scratch_size esp_nn_get_depthwise_conv_scratch_size_ansi
#define esp_nn_set_depthwise_conv_scratch_buf esp_nn_set_depthwise_conv_scratch_buf_ansi

#define esp_nn_relu6_s8 esp_nn_relu6_s8_ansi
#define esp_nn_hard_swish_s8 esp_nn_hard_swish_s8_ansi
#define esp_nn_mean_nhwc_s8 esp_nn_mean_nhwc_s8_ansi

#define esp_nn_avg_pool_s8 esp_nn_avg_pool_s8_ansi
#define esp_nn_max_pool_s8 esp_nn_max_pool_s8_ansi

#define esp_nn_fully_connected_s8 esp_nn_fully_connected_s8_ansi
#define esp_nn_fully_connected_per_ch_s8 esp_nn_fully_connected_per_ch_s8_ansi

#define esp_nn_get_softmax_scratch_size esp_nn_get_softmax_scratch_size_ansi
#define esp_nn_set_softmax_scratch_buf esp_nn_set_softmax_scratch_buf_ansi
#define esp_nn_softmax_s8 esp_nn_softmax_s8_ansi
