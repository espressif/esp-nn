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
#include <esp_nn_multicore.h>
#include <esp_timer.h>


#if __has_include("esp_idf_version.h")
#include <esp_idf_version.h>
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#define esp_cpu_get_ccount esp_cpu_get_cycle_count
#endif
#endif

static const char *TAG = "test_app";

/* The tests call these hooks around every kernel call. test_app reports no
 * cycles (bench_app measures them); on ESP32-P4 and ESP32-S31 the hooks clear
 * the PIE CFG before each optimized call, so a kernel that does not set it
 * itself fails its bit-exact check. */
static uint32_t start_c, start_opt;

void profile_c_start()
{
    /* initiate profiling */
    start_c = esp_cpu_get_ccount();
}

uint32_t profile_c_end()
{
    return esp_cpu_get_ccount() - start_c;
}

#if CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_ESP32S31
/* Clear the PIE CFG unaligned-access bit (bit 1) on this core. With it clear,
 * esp.vld.128 on P4 silently loads from addr & ~15. */
static void clear_pie_cfg_unaligned(void *unused)
{
    (void)unused;
    asm volatile (
        "esp.movx.r.cfg x29        \n\t"
        "andi   x29, x29, ~0b10    \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );
}

/* IDF does not save CFG per task, so anything else on the core can leave it
 * clear. Clear it on both cores before every optimized call: a kernel that
 * relies on an earlier setup call (or the worker's start-up) to have set it
 * then fails the bit-exact checks against the reference. */
static void clobber_pie_cfg(void)
{
    clear_pie_cfg_unaligned(NULL);
    if (esp_nn_dual_core_begin()) {
        if (esp_nn_dual_core_run(clear_pie_cfg_unaligned, NULL)) {
            esp_nn_dual_core_wait();
        }
        esp_nn_dual_core_end();
    }
}
#endif

void profile_opt_start()
{
#if CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_ESP32S31
    clobber_pie_cfg();
#endif
    /* initiate profiling */
    start_opt = esp_cpu_get_ccount();
}

uint32_t profile_opt_end()
{
    return esp_cpu_get_ccount() - start_opt;
}

/* Run one test, then report its own pass/fail tally. */
#define RUN_TEST(fn, name) do {                                     \
    uint32_t passed_before = esp_nn_test_passed;                    \
    uint32_t failed_before = esp_nn_test_failed;                    \
    uint32_t skipped_before = esp_nn_test_skipped;                  \
    fn();                                                           \
    printf("TEST: %s, passed=%"PRIu32", failed=%"PRIu32", skipped=%"PRIu32"\n", name, \
           esp_nn_test_passed - passed_before,                      \
           esp_nn_test_failed - failed_before,                      \
           esp_nn_test_skipped - skipped_before);                   \
} while (0)

void app_main()
{
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
    /* Exercise the dual-core split paths in the bit-exactness sweeps. */
    esp_nn_dual_core_enable();
    printf("DUAL_CORE: %s\n", esp_nn_dual_core_active() ? "active" : "inactive");
#endif
    /* s8 tests */
    ESP_LOGI(TAG, "Running s8 tests...");
    RUN_TEST(esp_nn_add_elementwise_s8_test, "add_s8");
    RUN_TEST(esp_nn_mul_elementwise_s8_test, "mul_s8");
    RUN_TEST(esp_nn_mul_broadcast_channel_s8_test, "mul_broadcast_ch_s8");
    RUN_TEST(esp_nn_depthwise_conv_s8_test, "depthwise_conv_s8");
    RUN_TEST(esp_nn_conv_s8_test, "conv_s8");
    RUN_TEST(esp_nn_relu6_s8_test, "relu6_s8");
    RUN_TEST(esp_nn_avg_pool_s8_test, "avg_pool_s8");
    RUN_TEST(esp_nn_max_pool_s8_test, "max_pool_s8");
    RUN_TEST(esp_nn_fully_connected_s8_test, "fc_s8");
    RUN_TEST(esp_nn_fully_connected_per_ch_s8_test, "fc_per_ch_s8");
    RUN_TEST(esp_nn_fully_connected_per_ch_s8_batch_test, "fc_per_ch_s8_batch");
    RUN_TEST(esp_nn_fully_connected_per_ch_s8_batch_1x1_test, "fc_per_ch_s8_batch_1x1");
    RUN_TEST(esp_nn_fully_connected_align_s8_test, "fc_align_s8");
    RUN_TEST(esp_nn_alignment_s8_test, "alignment_s8");
    RUN_TEST(esp_nn_fully_connected_large_s8_test, "fc_large_s8");
    RUN_TEST(esp_nn_softmax_s8_test, "softmax_s8");
    RUN_TEST(esp_nn_hard_swish_s8_test, "hard_swish_s8");
    RUN_TEST(esp_nn_mean_nhwc_s8_test, "mean_nhwc_s8");
    /* Last: it disables and re-enables the worker. */
    RUN_TEST(esp_nn_dual_core_test, "dual_core");
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

#ifdef CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE
    /* Lowest free stack the dual-core worker reached over all the suites. */
    TaskHandle_t worker = xTaskGetHandle("esp_nn_worker");
    if (worker != NULL) {
        printf("WORKER_STACK: min_free=%u of %u bytes\n",
               (unsigned)uxTaskGetStackHighWaterMark(worker),
               (unsigned)CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE);
    }
#endif
    printf("TEST_SUMMARY: passed=%"PRIu32", failed=%"PRIu32", skipped=%"PRIu32"\n",
        esp_nn_test_passed, esp_nn_test_failed, esp_nn_test_skipped);
    printf("TEST_RESULT: %s\n", esp_nn_test_failed == 0 ? "PASS" : "FAIL");
}
