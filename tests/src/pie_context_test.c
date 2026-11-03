/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Two PIE tasks on one core (ESP32-P4, esp-nn#47).
 *
 * FreeRTOS saves a task's PIE registers lazily: PIE is off in every task that
 * does not own it, and the first PIE instruction traps so the port can save
 * the previous owner's registers. Code that turns PIE on by hand skips that
 * trap, and two PIE tasks on one core then overwrite each other's registers.
 * Separately, a hardware loop (esp.lp.setup) that breaks the P4 placement
 * rules can lose or repeat iterations when an interrupt lands on its
 * boundary.
 *
 * Two tasks of the same priority are pinned to the caller's core, so the tick
 * time-slices them, and an optional GPTimer interrupt on that core adds
 * frequent interrupts. Each task repeatedly:
 *   1. loads every PIE register with its own random pattern, runs a hardware
 *      loop that follows the placement rules (and gets preempted inside it),
 *      reads the registers back and compares them bit by bit; the loop's own
 *      iteration count is checked too;
 *   2. runs esp_nn_fully_connected_s8() (PIE kernel with hardware loops) and
 *      compares the output bit by bit with esp_nn_fully_connected_s8_ansi().
 * Every mismatch is counted per register / per check; any mismatch fails.
 *
 * CFG is not checked: ESP-IDF does not keep it in the task context.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"

#if CONFIG_IDF_TARGET_ESP32P4

#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "driver/gptimer.h"

#ifndef CONFIG_NN_TEST_PIE_CONTEXT_SECONDS
#define CONFIG_NN_TEST_PIE_CONTEXT_SECONDS 10
#endif
#ifndef CONFIG_NN_TEST_PIE_CONTEXT_IRQ_US
#define CONFIG_NN_TEST_PIE_CONTEXT_IRQ_US 10
#endif

/* ESP-IDF v5.3 - v5.5 do not save q3 and swap SAR / FFT_BIT_WIDTH when they
 * save and restore a task's PIE context (IDF-16495). Leave those three alone
 * there, so the test checks esp-nn and not that IDF issue; set to 1 on an IDF
 * that has the fix. */
#ifndef NN_TEST_PIE_CONTEXT_FULL
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define NN_TEST_PIE_CONTEXT_FULL 1
#else
#define NN_TEST_PIE_CONTEXT_FULL 0
#endif
#endif

/* MALLOC_CAP_SIMD exists from IDF v5.3.3 / v5.4.1 / v5.5; MALLOC_CAP_DMA
 * also keeps buffers out of the LP RAM (as in esp_nn_multicore.c). */
#ifndef MALLOC_CAP_SIMD
#define MALLOC_CAP_SIMD MALLOC_CAP_DMA
#endif

typedef struct __attribute__((aligned(16))) {
    uint8_t q[8][16];
    uint8_t qacc[4][16];
    uint8_t ua[16];
    uint32_t xacc_l, xacc_h, sar, sar_bytes, fft_bit_width;
    uint32_t pad[3];
} nn_test_pie_ctx_t;

/* Shared with pie_context_test_esp32p4.S. */
_Static_assert(sizeof(nn_test_pie_ctx_t) == 240, "layout");
_Static_assert(offsetof(nn_test_pie_ctx_t, qacc) == 128, "layout");
_Static_assert(offsetof(nn_test_pie_ctx_t, ua) == 192, "layout");
_Static_assert(offsetof(nn_test_pie_ctx_t, xacc_l) == 208, "layout");
_Static_assert(offsetof(nn_test_pie_ctx_t, fft_bit_width) == 224, "layout");

uint32_t nn_test_pie_hold(const nn_test_pie_ctx_t *in, nn_test_pie_ctx_t *out,
                          uint32_t spins, uint32_t full);

enum {
    PIE_SPINS = 2000,       /* hardware-loop iterations per probe (8 addi each) */
    FC_ROW = 256,
    FC_OUT = 32,
    TASK_STACK = 4096,
};

typedef struct {
    uint32_t q[8], qacc, ua, xacc, sar, sar_bytes, fft_bit_width;
    uint32_t hwlp, fc;
} pie_bad_t;

typedef struct {
    int id;
    uint32_t seed;
    TickType_t ticks;
    unsigned irq_us;
    SemaphoreHandle_t done;
    /* results */
    bool setup_ok;      /* buffers allocated */
    bool probe_ok;      /* undisturbed probe reads back what it wrote */
    bool kernel_ok;     /* undisturbed PIE kernel matches the ANSI reference */
    uint32_t iters, irqs;
    pie_bad_t bad;
} pie_job_t;

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
/* Static: a task that never finishes must not point into a dead stack frame. */
static pie_job_t s_jobs[2];
static volatile uint32_t s_irq_hits;

static uint32_t rnd(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

static void *simd_alloc(size_t n)
{
    return heap_caps_aligned_alloc(16, n, MALLOC_CAP_SIMD | MALLOC_CAP_8BIT);
}

static bool irq_cb(gptimer_handle_t t, const gptimer_alarm_event_data_t *e, void *u)
{
    (void)t;
    (void)e;
    (void)u;
    s_irq_hits++;
    return false;
}

/* The GPTimer interrupt is allocated on the core that registers the callback,
 * so call this from a task pinned to the core under test. */
static gptimer_handle_t irq_start(unsigned us)
{
    if (us == 0) {
        return NULL;
    }
    gptimer_handle_t t = NULL;
    const gptimer_config_t cfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    if (gptimer_new_timer(&cfg, &t) != ESP_OK) {
        return NULL;
    }
    const gptimer_alarm_config_t alarm = {
        .alarm_count = us,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    const gptimer_event_callbacks_t cbs = { .on_alarm = irq_cb };
    if (gptimer_set_alarm_action(t, &alarm) != ESP_OK ||
        gptimer_register_event_callbacks(t, &cbs, NULL) != ESP_OK ||
        gptimer_enable(t) != ESP_OK) {
        gptimer_del_timer(t);
        return NULL;
    }
    if (gptimer_start(t) != ESP_OK) {
        gptimer_disable(t);
        gptimer_del_timer(t);
        return NULL;
    }
    return t;
}

static void irq_stop(gptimer_handle_t t)
{
    if (t != NULL) {
        gptimer_stop(t);
        gptimer_disable(t);
        gptimer_del_timer(t);
    }
}

static void ctx_fill(nn_test_pie_ctx_t *p, uint32_t seed)
{
    uint8_t *b = (uint8_t *)p;
    for (size_t i = 0; i < offsetof(nn_test_pie_ctx_t, xacc_l); i++) {
        b[i] = (uint8_t)rnd(&seed);
    }
    p->xacc_l = rnd(&seed);
    p->xacc_h = rnd(&seed) & 0xffu;     /* XACC is 40 bits */
    p->sar = rnd(&seed) & 0x3fu;        /* 6 bits */
    p->sar_bytes = rnd(&seed) & 0xfu;   /* 4 bits */
    p->fft_bit_width = rnd(&seed) & 0xfu;
    p->pad[0] = p->pad[1] = p->pad[2] = 0;
}

/* Count every register that differs (only the bits each register holds). */
static void ctx_compare(const nn_test_pie_ctx_t *want, const nn_test_pie_ctx_t *got, pie_bad_t *bad)
{
    for (int i = 0; i < 8; i++) {
        if (i == 3 && !NN_TEST_PIE_CONTEXT_FULL) {
            continue;
        }
        bad->q[i] += memcmp(want->q[i], got->q[i], 16) != 0;
    }
    bad->qacc += memcmp(want->qacc, got->qacc, sizeof(want->qacc)) != 0;
    bad->ua += memcmp(want->ua, got->ua, sizeof(want->ua)) != 0;
    bad->xacc += want->xacc_l != got->xacc_l || ((want->xacc_h ^ got->xacc_h) & 0xffu) != 0;
    bad->sar_bytes += ((want->sar_bytes ^ got->sar_bytes) & 0xfu) != 0;
    if (NN_TEST_PIE_CONTEXT_FULL) {
        bad->sar += ((want->sar ^ got->sar) & 0x3fu) != 0;
        bad->fft_bit_width += ((want->fft_bit_width ^ got->fft_bit_width) & 0xfu) != 0;
    }
}

static uint32_t bad_total(const pie_bad_t *b)
{
    uint32_t n = b->qacc + b->ua + b->xacc + b->sar + b->sar_bytes + b->fft_bit_width + b->hwlp + b->fc;
    for (int i = 0; i < 8; i++) {
        n += b->q[i];
    }
    return n;
}

static void fc_run(const int8_t *in, const int8_t *filter, const int32_t *bias, int8_t *out, bool ansi)
{
    /* input_offset 3 also exercises the filter-sum PIE path */
    if (ansi) {
        esp_nn_fully_connected_s8_ansi(in, 3, FC_ROW, filter, 0, bias, out, FC_OUT,
                                       -5, -7, 0x60000000, -128, 127);
    } else {
        esp_nn_fully_connected_s8(in, 3, FC_ROW, filter, 0, bias, out, FC_OUT,
                                  -5, -7, 0x60000000, -128, 127);
    }
}

static void pie_task_body(pie_job_t *job)
{
    nn_test_pie_ctx_t *in = simd_alloc(sizeof(*in));
    nn_test_pie_ctx_t *ref = simd_alloc(sizeof(*ref));
    nn_test_pie_ctx_t *got = simd_alloc(sizeof(*got));
    int8_t *fc_in = simd_alloc(FC_ROW);
    int8_t *fc_filter = simd_alloc(FC_ROW * FC_OUT);
    int32_t *fc_bias = simd_alloc(FC_OUT * sizeof(int32_t));
    int8_t *fc_ref = simd_alloc(FC_OUT);
    int8_t *fc_out = simd_alloc(FC_OUT);
    if (!in || !ref || !got || !fc_in || !fc_filter || !fc_bias || !fc_ref || !fc_out) {
        goto done;
    }
    job->setup_ok = true;

    uint32_t seed = job->seed;
    ctx_fill(in, rnd(&seed));
    for (int i = 0; i < FC_ROW; i++) {
        fc_in[i] = (int8_t)rnd(&seed);
    }
    for (int i = 0; i < FC_ROW * FC_OUT; i++) {
        fc_filter[i] = (int8_t)rnd(&seed);
    }
    for (int i = 0; i < FC_OUT; i++) {
        fc_bias[i] = (int32_t)(rnd(&seed) % 4096u) - 2048;
    }
    fc_run(fc_in, fc_filter, fc_bias, fc_ref, true);

    /* Take this task's first PIE trap outside the critical section, then get
     * an undisturbed run of the probe and of the kernel with this core's
     * interrupts off. */
    (void)nn_test_pie_hold(in, ref, 1, NN_TEST_PIE_CONTEXT_FULL);
    portENTER_CRITICAL(&s_mux);
    const uint32_t ref_cnt = nn_test_pie_hold(in, ref, PIE_SPINS, NN_TEST_PIE_CONTEXT_FULL);
    fc_run(fc_in, fc_filter, fc_bias, fc_out, false);
    portEXIT_CRITICAL(&s_mux);
    pie_bad_t probe = { 0 };
    ctx_compare(in, ref, &probe);
    job->probe_ok = ref_cnt == PIE_SPINS * 8u && bad_total(&probe) == 0;
    job->kernel_ok = memcmp(fc_out, fc_ref, FC_OUT) == 0;

    /* The first task owns the extra interrupt source on this core. */
    gptimer_handle_t irq = job->id == 0 ? irq_start(job->irq_us) : NULL;
    const uint32_t irq0 = s_irq_hits;
    const TickType_t start = xTaskGetTickCount();
    TickType_t rest = start;
    while (xTaskGetTickCount() - start < job->ticks) {
        const uint32_t cnt = nn_test_pie_hold(in, got, PIE_SPINS, NN_TEST_PIE_CONTEXT_FULL);
        ctx_compare(in, got, &job->bad);
        job->bad.hwlp += cnt != PIE_SPINS * 8u;

        fc_run(fc_in, fc_filter, fc_bias, fc_out, false);
        job->bad.fc += memcmp(fc_out, fc_ref, FC_OUT) != 0;
        job->iters++;

        /* Let the idle task on this core run now and then. */
        if (xTaskGetTickCount() - rest >= pdMS_TO_TICKS(100)) {
            vTaskDelay(1);
            rest = xTaskGetTickCount();
        }
    }
    job->irqs = s_irq_hits - irq0;
    irq_stop(irq);

done:
    heap_caps_free(in);
    heap_caps_free(ref);
    heap_caps_free(got);
    heap_caps_free(fc_in);
    heap_caps_free(fc_filter);
    heap_caps_free(fc_bias);
    heap_caps_free(fc_ref);
    heap_caps_free(fc_out);
}

static void pie_task(void *arg)
{
    pie_job_t *job = arg;
    pie_task_body(job);
    xSemaphoreGive(job->done);
    /* Deleted by the test once both tasks are done (static task memory). */
    vTaskSuspend(NULL);
}

void esp_nn_pie_context_test(void)
{
    const unsigned secs = CONFIG_NN_TEST_PIE_CONTEXT_SECONDS;
    if (secs == 0) {
        TEST_SKIP("pie_context: disabled (CONFIG_NN_TEST_PIE_CONTEXT_SECONDS=0)\n");
        return;
    }
    const BaseType_t core = xPortGetCoreID();
    const UBaseType_t prio = uxTaskPriorityGet(NULL);
    printf("pie_context: 2 tasks on core %d, %u s, irq every %u us, q3/SAR/FFT_BIT_WIDTH %s\n",
           (int)core, secs, (unsigned)CONFIG_NN_TEST_PIE_CONTEXT_IRQ_US,
           NN_TEST_PIE_CONTEXT_FULL ? "checked" : "not checked (IDF-16495)");

    SemaphoreHandle_t done = xSemaphoreCreateCounting(2, 0);
    pie_job_t *jobs = s_jobs;
    TaskHandle_t tasks[2] = { NULL, NULL };
    StackType_t *stacks[2] = { NULL, NULL };
    StaticTask_t *tcbs[2] = { NULL, NULL };
    int started = 0;
    for (int i = 0; i < 2 && done != NULL; i++) {
        jobs[i] = (pie_job_t) {
            .id = i,
            .seed = 0x9E3779B9u * (uint32_t)(i + 1),
            .ticks = pdMS_TO_TICKS(secs * 1000u),
            .irq_us = CONFIG_NN_TEST_PIE_CONTEXT_IRQ_US,
            .done = done,
        };
        /* The PIE context is saved on the task's stack: SIMD-capable memory. */
        stacks[i] = heap_caps_malloc(TASK_STACK, MALLOC_CAP_INTERNAL | MALLOC_CAP_SIMD | MALLOC_CAP_8BIT);
        tcbs[i] = heap_caps_malloc(sizeof(StaticTask_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (stacks[i] == NULL || tcbs[i] == NULL) {
            break;
        }
        tasks[i] = xTaskCreateStaticPinnedToCore(pie_task, i ? "nn_pie_b" : "nn_pie_a", TASK_STACK,
                                                 &jobs[i], prio, stacks[i], tcbs[i], core);
        if (tasks[i] == NULL) {
            break;
        }
        started++;
    }

    bool timeout = false;
    for (int i = 0; i < started; i++) {
        if (xSemaphoreTake(done, pdMS_TO_TICKS((secs + 30u) * 1000u)) != pdTRUE) {
            timeout = true;
        }
    }
    for (int i = 0; i < 2; i++) {
        if (tasks[i] != NULL && !timeout) {
            vTaskDelete(tasks[i]);
        }
        if (!timeout) {
            heap_caps_free(stacks[i]);
            heap_caps_free(tcbs[i]);
        }
    }
    if (done != NULL && !timeout) {
        vSemaphoreDelete(done);
    }

    if (started < 2) {
        TEST_SKIP("pie_context: could not start both tasks\n");
        return;
    }
    if (timeout) {
        TEST_FAIL("pie_context: tasks did not finish\n");
        return;
    }
    for (int i = 0; i < 2; i++) {
        const pie_job_t *j = &jobs[i];
        const pie_bad_t *b = &j->bad;
        if (!j->setup_ok) {
            TEST_SKIP("pie_context: task %d: no memory\n", i);
            continue;
        }
        printf("pie_context: task %d iters=%" PRIu32 " irqs=%" PRIu32
               " bad q0-7=%" PRIu32 "/%" PRIu32 "/%" PRIu32 "/%" PRIu32 "/%" PRIu32 "/%" PRIu32 "/%" PRIu32 "/%" PRIu32
               " qacc=%" PRIu32 " ua=%" PRIu32 " xacc=%" PRIu32 " sar=%" PRIu32 " sar_bytes=%" PRIu32
               " fft_bit_width=%" PRIu32 " hwlp=%" PRIu32 " fc=%" PRIu32 "\n",
               i, j->iters, j->irqs, b->q[0], b->q[1], b->q[2], b->q[3], b->q[4], b->q[5], b->q[6], b->q[7],
               b->qacc, b->ua, b->xacc, b->sar, b->sar_bytes, b->fft_bit_width, b->hwlp, b->fc);
        if (!j->probe_ok || !j->kernel_ok) {
            TEST_FAIL("pie_context: task %d: undisturbed run wrong (probe %s, fc vs ansi %s)\n", i,
                      j->probe_ok ? "ok" : "BAD", j->kernel_ok ? "ok" : "BAD");
        } else if (bad_total(b) != 0 || j->iters == 0) {
            TEST_FAIL("pie_context: task %d: %" PRIu32 " mismatches\n", i, bad_total(b));
        } else {
            TEST_PASS("pie_context: task %d passed\n", i);
        }
    }
}

#else /* !CONFIG_IDF_TARGET_ESP32P4 */

void esp_nn_pie_context_test(void)
{
    /* The PIE context probe is ESP32-P4 assembly. */
}

#endif /* CONFIG_IDF_TARGET_ESP32P4 */
