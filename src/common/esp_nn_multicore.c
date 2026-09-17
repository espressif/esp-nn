/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Opt-in dual-core work splitting for esp-nn kernels.
 *
 * The application enables it once with esp_nn_dual_core_enable(); kernels
 * that support splitting then hand half of their disjoint output range to a
 * worker task pinned on the other core and compute the rest inline. Results
 * are bit-identical to the single-core path: the split is by output
 * channels/rows only, every output element is computed by exactly one core,
 * and the ESP32-S3 cores share one data cache (no coherency management
 * needed). Each core has its own PIE state (QACC/ACCX/SAR).
 */

#include "esp_nn_multicore.h"

#if ESP_NN_DUAL_CORE_SUPPORTED

#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"

/* MALLOC_CAP_SIMD (IDF >= 5.5) marks memory that SIMD loads may access:
 * L2MEM and PSRAM on ESP32-P4, not the LP/RTC RAM. On older IDFs, which
 * have no such cap, MALLOC_CAP_DMA excludes the RTC RAM on every target. */
#ifndef MALLOC_CAP_SIMD
#define MALLOC_CAP_SIMD MALLOC_CAP_DMA
#endif

static TaskHandle_t s_worker;
static BaseType_t s_worker_core = -1;
static SemaphoreHandle_t s_done;
static void (*volatile s_fn)(void *);
static void *volatile s_arg;

/* Private scratch for the worker's kernel slice, grown on demand. */
static void *s_worker_scratch;
static int s_worker_scratch_size;

static void esp_nn_worker_main(void *unused)
{
    (void)unused;
#if CONFIG_IDF_TARGET_ESP32P4
    /* The RISC-V PIE state is per core and some kernels rely on it being
     * enabled already; enable it once for this core. */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );
#endif
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        void (*fn)(void *) = s_fn;
        void *arg = s_arg;
        if (fn) {
            fn(arg);
        }
        xSemaphoreGive(s_done);
    }
}

void esp_nn_dual_core_enable(void)
{
    if (s_worker != NULL) {
        return;
    }
#if !CONFIG_FREERTOS_UNICORE
    s_done = xSemaphoreCreateBinary();
    if (s_done == NULL) {
        return;
    }
    const BaseType_t core = (xPortGetCoreID() == 0) ? 1 : 0;
    if (xTaskCreatePinnedToCore(esp_nn_worker_main, "esp_nn_worker", 8 * 1024,
                                NULL, configMAX_PRIORITIES - 2, &s_worker,
                                core) != pdPASS) {
        vSemaphoreDelete(s_done);
        s_done = NULL;
        s_worker = NULL;
    } else {
        s_worker_core = core;
    }
#endif
}

bool esp_nn_dual_core_active(void)
{
    return s_worker != NULL;
}

bool esp_nn_dual_core_run(void (*fn)(void *), void *arg)
{
    if (s_worker == NULL) {
        return false;
    }
    /* A kernel slice already running on the worker must not hand work to
     * the worker: it would notify itself and then block in wait() on a
     * semaphore only it can give. A caller that merely runs on the worker's
     * core would gain nothing but two context switches. Both fall back to
     * the single-core path; every seam checks this return value. */
    if (xTaskGetCurrentTaskHandle() == s_worker || xPortGetCoreID() == s_worker_core) {
        return false;
    }
    s_fn = fn;
    s_arg = arg;
    xTaskNotifyGive(s_worker);
    return true;
}

void esp_nn_dual_core_wait(void)
{
    if (s_worker != NULL) {
        xSemaphoreTake(s_done, portMAX_DELAY);
    }
}

void *esp_nn_dual_core_scratch(int size)
{
    if (size <= s_worker_scratch_size) {
        return s_worker_scratch;
    }
    /* The kernels read this buffer with SIMD loads; MALLOC_CAP_SIMD keeps it
     * out of the LP/RTC RAM that plain MALLOC_CAP_INTERNAL falls back to on
     * ESP32-P4 when L2MEM runs short (esp.vld.128 faults there). */
    void *p = heap_caps_aligned_alloc(16, size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_SIMD);
    if (p == NULL) {
        p = heap_caps_aligned_alloc(16, size,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_SIMD);
    }
    if (p == NULL) {
        return NULL;
    }
    if (s_worker_scratch != NULL) {
        heap_caps_free(s_worker_scratch);
    }
    s_worker_scratch = p;
    s_worker_scratch_size = size;
    return s_worker_scratch;
}

#else /* !ESP_NN_DUAL_CORE_SUPPORTED */

void esp_nn_dual_core_enable(void) {}
bool esp_nn_dual_core_active(void) { return false; }
bool esp_nn_dual_core_run(void (*fn)(void *), void *arg)
{
    (void)fn;
    (void)arg;
    return false;
}
void esp_nn_dual_core_wait(void) {}
void *esp_nn_dual_core_scratch(int size)
{
    (void)size;
    return (void *)0;
}

#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
