/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_log.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <test_functions.h>
#include <esp_timer.h>


#if __has_include("esp_idf_version.h")
#include <esp_idf_version.h>
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#define esp_cpu_get_ccount esp_cpu_get_cycle_count
#endif
#endif

static const char *TAG = "test_app";
static uint32_t start_c, start_opt, total_c, total_opt;

void profile_c_start()
{
    /* initiate profiling */
    start_c = esp_cpu_get_ccount();
}

uint32_t profile_c_end()
{
    /* record profile number */
    total_c = esp_cpu_get_ccount() - start_c;
    return total_c;
}

void profile_opt_start()
{
    /* initiate profiling */
    start_opt = esp_cpu_get_ccount();
}

uint32_t profile_opt_end()
{
    /* record profile number */
    total_opt = esp_cpu_get_ccount() - start_opt;
    return total_opt;
}

static void print_profile(const char *kernel)
{
    float speedup = (total_c > 0 && total_opt > 0) ? (float)total_c / (float)total_opt : 0.0f;
    printf("PROFILE: %s, ansi=%"PRIu32", opt=%"PRIu32", speedup=%.2fx\n",
           kernel, total_c, total_opt, speedup);
}

void app_main()
{
    /* s8 tests */
    ESP_LOGI(TAG, "Running s8 tests...");
    esp_nn_add_elementwise_s8_test();
    print_profile("add_s8");
    esp_nn_mul_elementwise_s8_test();
    print_profile("mul_s8");
    esp_nn_depthwise_conv_s8_test();
    print_profile("depthwise_conv_s8");
    esp_nn_conv_s8_test();
    print_profile("conv_s8");
    esp_nn_relu6_s8_test();
    print_profile("relu6_s8");
    esp_nn_avg_pool_s8_test();
    print_profile("avg_pool_s8");
    esp_nn_max_pool_s8_test();
    print_profile("max_pool_s8");
    esp_nn_fully_connected_s8_test();
    print_profile("fc_s8");
    esp_nn_fully_connected_per_ch_s8_test();
    print_profile("fc_per_ch_s8");
    esp_nn_softmax_s8_test();
    print_profile("softmax_s8");
    ESP_LOGI(TAG, "s8 tests done!\n");

    /* u8 tests */
    //ESP_LOGI(TAG, "Running u8 tests...");
    //esp_nn_add_elementwise_u8_test();
    //esp_nn_depthwise_conv_u8_test();
    //esp_nn_conv_u8_test();
    //esp_nn_avg_pool_u8_test();
    //esp_nn_max_pool_u8_test();
    //esp_nn_fully_connected_u8_test();
    //ESP_LOGI(TAG, "u8 tests done!\n");
}
