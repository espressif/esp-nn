/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* The dual-core worker: kernels stay bit-exact while another task holds or
 * uses the worker, and disable() frees everything enable() allocated. */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "test_functions.h"
#include "test_utils.h"

#if ESP_NN_DUAL_CORE_SUPPORTED

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"

/* The contender runs kernels, so its stack must take 128-bit stores: not
 * the ESP32-P4's LP RAM. MALLOC_CAP_SIMD is missing before IDF v5.3.3 and
 * v5.4.1; MALLOC_CAP_DMA also excludes LP RAM there. */
#ifndef MALLOC_CAP_SIMD
#define MALLOC_CAP_SIMD MALLOC_CAP_DMA
#endif

#define CONTENDER_STACK 4096
#define CONTENDER_LEN   8192

static volatile bool contender_stop;
static volatile bool contender_done;
static volatile uint32_t contender_claims;
static volatile uint32_t contender_runs;
static volatile uint32_t contender_mismatches;

static void contender_mul(int8_t *out, const int8_t *a, const int8_t *b)
{
    esp_nn_mul_elementwise_s8(a, b, 3, -5, out, 7, 1 << 30, -6, -128, 127, CONTENDER_LEN);
}

/* Takes the worker and holds it across a tick, then runs an elementwise
 * multiply that splits when the worker is free. The main task's kernels
 * meet a busy worker, and both sides must stay bit-exact. */
static void contender(void *arg)
{
    int8_t *buf = (int8_t *) arg;
    int8_t *a = buf, *b = buf + CONTENDER_LEN;
    int8_t *ref = buf + 2 * CONTENDER_LEN, *out = buf + 3 * CONTENDER_LEN;

    esp_nn_mul_elementwise_s8_ansi(a, b, 3, -5, ref, 7, 1 << 30, -6, -128, 127, CONTENDER_LEN);
    while (!contender_stop) {
        if (esp_nn_dual_core_begin()) {
            contender_claims++;
            vTaskDelay(1);
            esp_nn_dual_core_end();
        }
        contender_mul(out, a, b);
        if (memcmp(out, ref, CONTENDER_LEN) != 0) {
            contender_mismatches++;
        }
        contender_runs++;
        taskYIELD();
    }
    contender_done = true;
    /* Parked rather than deleted: its static stack and TCB stay valid. */
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    }
}

void esp_nn_dual_core_test(void)
{
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    if (!esp_nn_dual_core_active()) {
        TEST_SKIP("dual core: the worker is not running\n");
        return;
    }

    /* 1. Contention. A second task on this core takes the worker and runs
     * split kernels while the conv and depthwise sweeps run here. */
    int8_t *buf = heap_caps_aligned_alloc(16, 4 * CONTENDER_LEN,
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    StackType_t *stack = heap_caps_malloc(CONTENDER_STACK,
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_SIMD | MALLOC_CAP_8BIT);
    StaticTask_t *tcb = heap_caps_malloc(sizeof(StaticTask_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf == NULL || stack == NULL || tcb == NULL) {
        TEST_SKIP("dual core: allocation failed\n");
        heap_caps_free(buf);
        heap_caps_free(stack);
        heap_caps_free(tcb);
        return;
    }
    for (int i = 0; i < 2 * CONTENDER_LEN; i++) {
        buf[i] = (int8_t) (rand() & 0xff);
    }
    contender_stop = false;
    contender_done = false;
    contender_claims = contender_runs = contender_mismatches = 0;
    TaskHandle_t task = xTaskCreateStaticPinnedToCore(contender, "nn_contender", CONTENDER_STACK,
                                                      buf, uxTaskPriorityGet(NULL), stack, tcb,
                                                      xPortGetCoreID());
    esp_nn_depthwise_conv_s8_test();
    esp_nn_conv_s8_test();
    contender_stop = true;
    while (!contender_done) {
        vTaskDelay(1);
    }
    printf("dual core: contender took the worker %" PRIu32 " times in %" PRIu32 " runs\n",
           contender_claims, contender_runs);
    if (contender_mismatches != 0) {
        TEST_FAIL("dual core: contender's multiply differed %" PRIu32 " times\n",
                  contender_mismatches);
    } else if (contender_claims == 0) {
        TEST_FAIL("dual core: the contender never got the worker\n");
    } else {
        TEST_PASS("dual core: contender bit-exact\n");
    }
    (void) task;

    /* 2. disable() frees everything; enable() starts the worker again. */
    esp_nn_dual_core_disable();
    if (esp_nn_dual_core_active()) {
        TEST_FAIL("dual core: still active after disable()\n");
    }
    const size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    esp_nn_dual_core_enable();
    if (!esp_nn_dual_core_active()) {
        TEST_FAIL("dual core: enable() after disable() did not start the worker\n");
        return;
    }
    esp_nn_depthwise_conv_s8_test();   /* allocates the worker scratch */
    esp_nn_dual_core_disable();
    const size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    if (heap_after != heap_before) {
        TEST_FAIL("dual core: disable() left %d bytes allocated\n",
                  (int) heap_before - (int) heap_after);
    } else {
        TEST_PASS("dual core: disable() freed the worker\n");
    }
    esp_nn_dual_core_enable();
}

#else /* !ESP_NN_DUAL_CORE_SUPPORTED */

void esp_nn_dual_core_test(void)
{
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    TEST_SKIP("dual core: not supported on this target\n");
}

#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
