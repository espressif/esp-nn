/*
 * SPDX-FileCopyrightText: 2024-2026 Espressif Systems (Shanghai) CO LTD
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



#define esp_nn_mul_broadcast_channel_s8 esp_nn_mul_broadcast_channel_s8_ansi

void esp_nn_add_elementwise_s8_esp32p4(const int8_t *input1_data,
                                        const int8_t *input2_data,
                                        const int32_t input1_offset,
                                        const int32_t input2_offset,
                                        const int32_t input1_mult,
                                        const int32_t input2_mult,
                                        const int32_t input1_shift,
                                        const int32_t input2_shift,
                                        const int32_t left_shift,
                                        int8_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const int32_t size);
#define esp_nn_add_elementwise_s8 esp_nn_add_elementwise_s8_esp32p4

void esp_nn_mul_elementwise_s8_esp32p4(const int8_t *input1_data,
                                        const int8_t *input2_data,
                                        const int32_t input1_offset,
                                        const int32_t input2_offset,
                                        int8_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const int32_t size);
#define esp_nn_mul_elementwise_s8 esp_nn_mul_elementwise_s8_esp32p4

void esp_nn_depthwise_conv_s8_esp32p4(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data);
int esp_nn_get_depthwise_conv_scratch_size_esp32p4(const data_dims_t *input_dims,
                                                    const data_dims_t *filter_dims,
                                                    const data_dims_t *output_dims,
                                                    const dw_conv_params_t *conv_params);
void esp_nn_set_depthwise_conv_scratch_buf_esp32p4(const void *buf);
#define esp_nn_depthwise_conv_s8 esp_nn_depthwise_conv_s8_esp32p4

#define esp_nn_conv_s8 esp_nn_conv_s8_esp32p4

#define esp_nn_get_conv_scratch_size esp_nn_get_conv_scratch_size_esp32p4
#define esp_nn_set_conv_scratch_buf esp_nn_set_conv_scratch_buf_esp32p4

#define esp_nn_get_depthwise_conv_scratch_size esp_nn_get_depthwise_conv_scratch_size_esp32p4
#define esp_nn_set_depthwise_conv_scratch_buf esp_nn_set_depthwise_conv_scratch_buf_esp32p4

/* Functions not yet optimized for P4 - use ANSI fallback */
#define esp_nn_hard_swish_s8 esp_nn_hard_swish_s8_ansi
#define esp_nn_mean_nhwc_s8 esp_nn_mean_nhwc_s8_ansi

void esp_nn_relu6_s8_esp32p4(int8_t *data, uint16_t size);
#define esp_nn_relu6_s8 esp_nn_relu6_s8_esp32p4

#define esp_nn_hard_swish_s8 esp_nn_hard_swish_s8_ansi
#define esp_nn_get_hard_swish_scratch_size() 0
#define esp_nn_set_hard_swish_scratch_buf(buf)

#define esp_nn_mean_nhwc_s8 esp_nn_mean_nhwc_s8_ansi

void esp_nn_avg_pool_s8_esp32p4(const int8_t *input,
                                 const uint16_t input_wd,
                                 const uint16_t input_ht,
                                 int8_t *output,
                                 const uint16_t output_wd,
                                 const uint16_t output_ht,
                                 const uint16_t stride_wd,
                                 const uint16_t stride_ht,
                                 const uint16_t filter_wd,
                                 const uint16_t filter_ht,
                                 const uint16_t pad_wd,
                                 const uint16_t pad_ht,
                                 const int32_t activation_min,
                                 const int32_t activation_max,
                                 const uint16_t channels);
#define esp_nn_avg_pool_s8 esp_nn_avg_pool_s8_esp32p4
void esp_nn_max_pool_s8_esp32p4(const int8_t *input,
                                 const uint16_t input_wd,
                                 const uint16_t input_ht,
                                 int8_t *output,
                                 const uint16_t output_wd,
                                 const uint16_t output_ht,
                                 const uint16_t stride_wd,
                                 const uint16_t stride_ht,
                                 const uint16_t filter_wd,
                                 const uint16_t filter_ht,
                                 const uint16_t pad_wd,
                                 const uint16_t pad_ht,
                                 const int32_t activation_min,
                                 const int32_t activation_max,
                                 const uint16_t channels);
#define esp_nn_max_pool_s8 esp_nn_max_pool_s8_esp32p4

void esp_nn_fully_connected_s8_esp32p4(const int8_t *input_data,
                                        const int32_t input_offset,
                                        const uint16_t row_len,
                                        const int8_t *filter_data,
                                        const int32_t filter_offset,
                                        const int32_t *bias,
                                        int8_t *out_data,
                                        const uint16_t out_channels,
                                        const int32_t out_offset,
                                        const int32_t out_shift,
                                        const int32_t out_mult,
                                        const int32_t activation_min,
                                        const int32_t activation_max);
void esp_nn_fully_connected_per_ch_s8_esp32p4(const int8_t *input_data,
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
                                        const int32_t activation_max);
#define esp_nn_fully_connected_s8 esp_nn_fully_connected_s8_esp32p4
#define esp_nn_fully_connected_per_ch_s8 esp_nn_fully_connected_per_ch_s8_esp32p4

int32_t esp_nn_get_softmax_scratch_size_esp32p4(const int32_t width, const int32_t height);
void esp_nn_set_softmax_scratch_buf_esp32p4(void *buffer);
void esp_nn_softmax_s8_esp32p4(const int8_t *input_data,
                                const int32_t height,
                                const int32_t width,
                                const int32_t mult,
                                const int32_t shift,
                                const int32_t diff_min,
                                int8_t *output_data);
#define esp_nn_get_softmax_scratch_size esp_nn_get_softmax_scratch_size_esp32p4
#define esp_nn_set_softmax_scratch_buf esp_nn_set_softmax_scratch_buf_esp32p4
#define esp_nn_softmax_s8 esp_nn_softmax_s8_esp32p4
