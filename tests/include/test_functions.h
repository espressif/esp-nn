/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */


/* int8_t ops tests */
void esp_nn_add_elementwise_s8_test();
void esp_nn_mul_elementwise_s8_test();

void esp_nn_depthwise_conv_s8_test();
void esp_nn_conv_s8_test();

void esp_nn_avg_pool_s8_test();
void esp_nn_max_pool_s8_test();

void esp_nn_fully_connected_s8_test();
void esp_nn_fully_connected_per_ch_s8_test();

void esp_nn_relu6_s8_test();

void esp_nn_softmax_s8_test();

void esp_nn_hard_swish_s8_test();
void esp_nn_mean_nhwc_s8_test();

/* uint8_t ops tests */
void esp_nn_add_elementwise_u8_test();

void esp_nn_depthwise_conv_u8_test();
void esp_nn_conv_u8_test();

void esp_nn_avg_pool_u8_test();
void esp_nn_max_pool_u8_test();

void esp_nn_fully_connected_u8_test();

/* instructions test functions */
void compare_instructions_test();
void arith_instructions_test();
void min_max_instructions_test();
void bitwise_instructions_test();
void load_store_instructions_test();
