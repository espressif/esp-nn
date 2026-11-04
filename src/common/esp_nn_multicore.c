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
#include <common_functions.h>

#if ESP_NN_DUAL_CORE_SUPPORTED

#include <assert.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#include "esp_idf_version.h"

/* MALLOC_CAP_SIMD (IDF v5.3.3, v5.4.1, v5.5 and later) marks memory that
 * SIMD loads may access: L2MEM and PSRAM on ESP32-P4, not the LP/RTC RAM.
 * On older IDFs, which have no such cap, MALLOC_CAP_DMA excludes the RTC RAM
 * on every target. */
#ifndef MALLOC_CAP_SIMD
#define MALLOC_CAP_SIMD MALLOC_CAP_DMA
#endif

/* Worker stack in bytes (StackType_t is one byte in IDF). */
#ifndef CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE
#define CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE 4096
#endif
#define ESP_NN_WORKER_STACK_SIZE CONFIG_NN_DUAL_CORE_WORKER_STACK_SIZE

/* Above application tasks, so a dispatched slice starts at once; only the
 * IPC task (configMAX_PRIORITIES - 1) ranks higher. */
#define ESP_NN_WORKER_PRIORITY (configMAX_PRIORITIES - 2)

/* Whether PSRAM may hold a task stack. The option was renamed in IDF 5.4;
 * sdkconfig.h still defines the old name as an alias. */
#if CONFIG_FREERTOS_TASK_CREATE_ALLOW_EXT_MEM || CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY
#define ESP_NN_WORKER_STACK_EXT_OK 1
#else
#define ESP_NN_WORKER_STACK_EXT_OK 0
#endif

#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 2, 0)
#define xTaskGetCurrentTaskHandleForCore xTaskGetCurrentTaskHandleForCPU
#endif

static TaskHandle_t s_worker;
static BaseType_t s_worker_core = -1;
static SemaphoreHandle_t s_done;
/* One split kernel at a time: the job slot, the completion semaphore and the
 * worker scratch are single. Held from esp_nn_dual_core_begin() to _end(). */
static SemaphoreHandle_t s_claim;
/* The worker's buffers, created static so they can be freed in disable(). */
static StackType_t *s_stack;
static StaticTask_t *s_tcb;
static void (*volatile s_fn)(void *);
static void *volatile s_arg;

/* Private scratch for the worker's kernel slice, grown on demand. */
static void *s_worker_scratch;
static int s_worker_scratch_size;

static void esp_nn_worker_main(void *unused)
{
    (void)unused;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        void (*fn)(void *) = s_fn;
        void *arg = s_arg;
        if (fn) {
#if CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_ESP32S31
            /* Jobs run sub-kernels directly, past their public entry, and
             * the PIE CFG is per core and unsaved across tasks: set it for
             * every job on this core (see ESP_NN_PIE_ENABLE). */
            ESP_NN_PIE_ENABLE();
#endif
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
    s_claim = xSemaphoreCreateMutex();
    if (s_done == NULL || s_claim == NULL) {
        goto fail;
    }
#if CONFIG_IDF_TARGET_ESP32P4
    /* A task's first PIE instruction traps, and IDF pins the task to the core
     * it runs on. Take that trap here, so the calling task is pinned to this
     * core and the worker goes to the other one. */
    ESP_NN_PIE_ENABLE();
#endif
    const BaseType_t core = (xPortGetCoreID() == 0) ? 1 : 0;
    /* The kernels store 128-bit vectors to the stack too, so it needs the
     * same memory as the scratch below; a plain xTaskCreate stack can land in
     * the ESP32-P4's LP/RTC RAM and fault on the first split kernel. Created
     * static, so the buffers stay put. */
#if CONFIG_NN_DUAL_CORE_WORKER_STACK_PSRAM
    /* PSRAM takes 128-bit accesses on both targets. No MALLOC_CAP_SIMD: on
     * IDF before v5.3.3 and v5.4.1 it stands in for MALLOC_CAP_DMA, which
     * PSRAM lacks. */
    StackType_t *stack = heap_caps_malloc(ESP_NN_WORKER_STACK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    /* Internal SIMD-capable RAM first, then PSRAM where the config allows
     * task stacks there. No MALLOC_CAP_SIMD on the PSRAM side, as in the
     * PSRAM-only case: PSRAM takes 128-bit accesses on both targets. */
    StackType_t *stack = heap_caps_malloc_prefer(ESP_NN_WORKER_STACK_SIZE, 1 + ESP_NN_WORKER_STACK_EXT_OK,
                                                 MALLOC_CAP_INTERNAL | MALLOC_CAP_SIMD | MALLOC_CAP_8BIT,
                                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#endif
    StaticTask_t *tcb = heap_caps_malloc(sizeof(StaticTask_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (stack != NULL && tcb != NULL) {
        s_worker = xTaskCreateStaticPinnedToCore(esp_nn_worker_main, "esp_nn_worker",
                                                 ESP_NN_WORKER_STACK_SIZE, NULL, ESP_NN_WORKER_PRIORITY,
                                                 stack, tcb, core);
    }
    if (s_worker != NULL) {
        s_worker_core = core;
        s_stack = stack;
        s_tcb = tcb;
        return;
    }
    /* No suitable memory: stay single core rather than risk the fault. */
    heap_caps_free(stack);
    heap_caps_free(tcb);
fail:
    if (s_done != NULL) {
        vSemaphoreDelete(s_done);
        s_done = NULL;
    }
    if (s_claim != NULL) {
        vSemaphoreDelete(s_claim);
        s_claim = NULL;
    }
#endif
}

void esp_nn_dual_core_disable(void)
{
    if (s_worker == NULL) {
        return;
    }
    /* No other task may be in esp-nn here (see the header), so the claim is
     * free and this take does not wait. */
    xSemaphoreTake(s_claim, portMAX_DELAY);
    TaskHandle_t worker = s_worker;
    s_worker = NULL;
    s_worker_core = -1;
    /* No job is pending, so the worker is idle or about to block. Suspend
     * it and wait until it is off both cores: only then does vTaskDelete()
     * run the port's cleanup (its PIE/FPU save area) synchronously, after
     * which its stack and TCB are ours to free. */
    vTaskSuspend(worker);
    for (BaseType_t c = 0; c < portNUM_PROCESSORS; c++) {
        while (xTaskGetCurrentTaskHandleForCore(c) == worker) {
            vTaskDelay(1);
        }
    }
    vTaskDelete(worker);
    heap_caps_free(s_stack);
    heap_caps_free(s_tcb);
    s_stack = NULL;
    s_tcb = NULL;
    heap_caps_free(s_worker_scratch);
    s_worker_scratch = NULL;
    s_worker_scratch_size = 0;
    vSemaphoreDelete(s_done);
    s_done = NULL;
    xSemaphoreGive(s_claim);
    vSemaphoreDelete(s_claim);
    s_claim = NULL;
}

bool esp_nn_dual_core_active(void)
{
    return s_worker != NULL;
}

bool esp_nn_dual_core_begin(void)
{
    if (s_worker == NULL) {
        return false;
    }
    /* A kernel slice already running on the worker must not hand work to
     * the worker: it would notify itself and then block in wait() on a
     * semaphore only it can give. A caller that merely runs on the worker's
     * core would gain nothing but two context switches. Both fall back to
     * the single-core path. */
    if (xTaskGetCurrentTaskHandle() == s_worker || xPortGetCoreID() == s_worker_core) {
        return false;
    }
    /* Never block: while another task holds the worker, this kernel runs
     * single core (bit-identical). A nested begin() from the holder fails
     * the same way, the mutex not being recursive. */
    return xSemaphoreTake(s_claim, 0) == pdTRUE;
}

void esp_nn_dual_core_end(void)
{
    xSemaphoreGive(s_claim);
}

bool esp_nn_dual_core_run(void (*fn)(void *), void *arg)
{
    if (s_worker == NULL) {
        return false;
    }
    if (xSemaphoreGetMutexHolder(s_claim) != xTaskGetCurrentTaskHandle()) {
        assert(!"esp_nn_dual_core_run() without esp_nn_dual_core_begin()");
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
void esp_nn_dual_core_disable(void) {}
bool esp_nn_dual_core_active(void) { return false; }
bool esp_nn_dual_core_begin(void) { return false; }
void esp_nn_dual_core_end(void) {}
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
