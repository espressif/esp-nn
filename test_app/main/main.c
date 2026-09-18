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
static uint32_t start_c, start_opt, total_c, total_opt;
/* Per-suite sums of every profiled section, reset by RUN_TEST. The per-call
 * return values above stay per call for the tests' own cycle prints. */
static uint64_t suite_c, suite_opt;

void profile_c_start()
{
    /* initiate profiling */
    start_c = esp_cpu_get_ccount();
}

uint32_t profile_c_end()
{
    /* record profile number */
    total_c = esp_cpu_get_ccount() - start_c;
    suite_c += total_c;
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
    suite_opt += total_opt;
    return total_opt;
}

/* nano-printf (the default C library for ESP32-C2 on IDF v5.x) implements
 * neither 64-bit nor floating-point conversions, so format both by hand. */
static const char *u64_dec(uint64_t v, char *buf, size_t len)
{
    char *p = buf + len - 1;
    *p = '\0';
    do {
        *--p = (char) ('0' + (v % 10));
        v /= 10;
    } while (v);
    return p;
}

/* Silent for a test that does not profile, which leaves the counters at 0. */
static void print_profile(const char *kernel)
{
    if (suite_c == 0 && suite_opt == 0) {
        return;
    }
    /* Whole-suite sums, not the last case. Cycle counts are wherever the
     * app runs (emulator or silicon); the CI report labels the source. */
    uint32_t whole = 0, hundredths = 0;
    if (suite_c > 0 && suite_opt > 0) {
        uint64_t ratio_x100 = (suite_c * 100u) / suite_opt;
        whole = (uint32_t) (ratio_x100 / 100);
        hundredths = (uint32_t) (ratio_x100 % 100);
    }
    char ansi_buf[24], opt_buf[24];
    printf("PROFILE: %s, ansi=%s, opt=%s, speedup=%u.%02ux\n", kernel,
           u64_dec(suite_c, ansi_buf, sizeof(ansi_buf)),
           u64_dec(suite_opt, opt_buf, sizeof(opt_buf)),
           (unsigned) whole, (unsigned) hundredths);
}


/* Run one test, then report its own pass/fail tally and its cycle counts. */
#define RUN_TEST(fn, name) do {                                     \
    uint32_t passed_before = esp_nn_test_passed;                    \
    uint32_t failed_before = esp_nn_test_failed;                    \
    uint32_t skipped_before = esp_nn_test_skipped;                  \
    total_c = 0;                                                    \
    total_opt = 0;                                                  \
    suite_c = 0;                                                    \
    suite_opt = 0;                                                  \
    fn();                                                           \
    printf("TEST: %s, passed=%"PRIu32", failed=%"PRIu32", skipped=%"PRIu32"\n", name, \
           esp_nn_test_passed - passed_before,                      \
           esp_nn_test_failed - failed_before,                      \
           esp_nn_test_skipped - skipped_before);                   \
    print_profile(name);                                            \
} while (0)

void app_main()
{
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
    /* Exercise the dual-core split paths in the bit-exactness sweeps. */
    esp_nn_dual_core_enable();
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
    RUN_TEST(esp_nn_fully_connected_align_s8_test, "fc_align_s8");
    RUN_TEST(esp_nn_fully_connected_perf_test, "fc_perf");
    RUN_TEST(esp_nn_softmax_s8_test, "softmax_s8");
    RUN_TEST(esp_nn_hard_swish_s8_test, "hard_swish_s8");
    RUN_TEST(esp_nn_mean_nhwc_s8_test, "mean_nhwc_s8");
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

    printf("TEST_SUMMARY: passed=%"PRIu32", failed=%"PRIu32", skipped=%"PRIu32"\n",
        esp_nn_test_passed, esp_nn_test_failed, esp_nn_test_skipped);
    printf("TEST_RESULT: %s\n", esp_nn_test_failed == 0 ? "PASS" : "FAIL");
}
