/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Cycle benchmark of the esp-nn kernels against their ANSI C reference.
 * Output: one BENCH line per case and variant, between BENCH_START and
 * BENCH_DONE. See README.md. */

#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_rom_sys.h"
#include "esp_idf_version.h"
#include "sdkconfig.h"

#include "bench_ops.h"

#ifndef BENCH_SEED
#define BENCH_SEED 0xC0FFEE42u
#endif

static void bench_task(void *arg)
{
    (void) arg;
    printf("BENCH_START target=%s idf=%s cpu_mhz=%u psram=%d iters=%d seed=0x%08x\n",
           CONFIG_IDF_TARGET, esp_get_idf_version(), (unsigned) esp_rom_get_cpu_ticks_per_us(),
           bench_has_psram(), BENCH_ITERS, (unsigned) BENCH_SEED);

    bench_seed(BENCH_SEED);
    bench_conv();
    bench_depthwise();
    bench_fully_connected();
    bench_basic_math();
    bench_pooling();
    bench_activations();
    bench_softmax();
    bench_mean();

    printf("BENCH_DONE cases=%d failures=%d skipped=%d\n", bench_cases_run(), bench_failures(),
           bench_skipped());
    /* A task created WithCaps must not delete itself. */
    vTaskSuspend(NULL);
}

void app_main(void)
{
    /* One core, the worker off: the numbers are single-core kernel cycles.
     * The stack is large because some kernels keep small arrays on it.
     * ESP32-S31 has PIE on core 1 only. */
#if CONFIG_IDF_TARGET_ESP32S31
    const BaseType_t core = 1;
#else
    const BaseType_t core = 0;
#endif
#if CONFIG_IDF_TARGET_ESP32P4
    /* ESP32-P4: the stack must be in SIMD-capable memory, not LP RAM. */
    const BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(bench_task, "bench", 16 * 1024, NULL, 5,
                                                          NULL, core,
                                                          MALLOC_CAP_SIMD | MALLOC_CAP_8BIT);
#else
    const BaseType_t ok = xTaskCreatePinnedToCore(bench_task, "bench", 16 * 1024, NULL, 5, NULL,
                                                  core);
#endif
    if (ok != pdPASS) {
        printf("BENCH_DONE cases=0 failures=1 skipped=0 (no memory for the bench task)\n");
    }
}
