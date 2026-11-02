/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"

#include "bench.h"

/* ------------------------------------------------------------------ PRNG -- */

static uint32_t s_seed = 0x12345678u;
static uint32_t s_rand = 0x12345678u;

void bench_seed(uint32_t seed)
{
    s_seed = seed ? seed : 0x12345678u;
    s_rand = s_seed;
}

void bench_seed_case(const char *op, const char *shape)
{
    /* FNV-1a over "op:shape", mixed with the run seed. */
    uint32_t h = 2166136261u ^ s_seed;
    for (const char *s = op; *s; s++) {
        h = (h ^ (uint8_t) *s) * 16777619u;
    }
    h = (h ^ ':') * 16777619u;
    for (const char *s = shape; *s; s++) {
        h = (h ^ (uint8_t) *s) * 16777619u;
    }
    s_rand = h ? h : 0x12345678u;
}

uint32_t bench_rand(void)
{
    /* xorshift32 */
    uint32_t x = s_rand;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_rand = x;
    return x;
}

int32_t bench_rand_range(int32_t lo, int32_t hi)
{
    return lo + (int32_t) (bench_rand() % (uint32_t) (hi - lo + 1));
}

void bench_fill(int8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        buf[i] = (int8_t) bench_rand();
    }
}

/* --------------------------------------------------------------- Buffers -- */

bool bench_has_psram(void)
{
    return heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0;
}

void *bench_alloc(size_t size, bench_mem_t mem)
{
#if CONFIG_IDF_TARGET_ESP32P4
    /* ESP32-P4: internal buffers must be SIMD-capable memory, not LP RAM. */
    const uint32_t int_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_SIMD;
#else
    const uint32_t int_caps = MALLOC_CAP_INTERNAL;
#endif
    const uint32_t caps = (mem == BENCH_MEM_EXT && bench_has_psram())
                          ? MALLOC_CAP_SPIRAM : int_caps;
    return heap_caps_aligned_alloc(16, size ? size : 1, caps | MALLOC_CAP_8BIT);
}

void bench_free(void *p)
{
    heap_caps_free(p);
}

/* Guard pattern, different on each side and per salt. */
static int8_t guard_byte(size_t i, int side, uint32_t salt)
{
    uint32_t s = (uint32_t) i * 2654435761u + (uint32_t) side * 40503u + salt;
    return (int8_t) (s >> 24);
}

static void guard_fill(int8_t *out, size_t size, uint32_t salt)
{
    for (size_t i = 0; i < BENCH_GUARD; i++) {
        out[-1 - (int) i] = guard_byte(i, 0, salt);
        out[size + i] = guard_byte(i, 1, salt);
    }
}

static bool guard_ok(const int8_t *out, size_t size, uint32_t salt)
{
    for (size_t i = 0; i < BENCH_GUARD; i++) {
        if (out[-1 - (int) i] != guard_byte(i, 0, salt) ||
                out[size + i] != guard_byte(i, 1, salt)) {
            return false;
        }
    }
    return true;
}

/* -------------------------------------------------------------- Running -- */

static int s_cases;
static int s_failures;
static int s_skipped;

/* Times fn: BENCH_ITERS calls, inputs alternating between the two variants,
 * minimum over all but the first call. With gold set, also checks every
 * output and the guard bytes; returns false on any mismatch. */
static bool time_calls(const bench_case_t *c, bench_fn fn, int8_t *out,
                       int8_t *const gold[2], uint32_t *min_cycles)
{
    bool ok = true;
    uint32_t best = UINT32_MAX;
    for (int it = 0; it < BENCH_ITERS; it++) {
        const int v = it & 1;
        memset(out, 0x5a ^ it, c->out_size);
        if (c->prep) {
            c->prep(c->ctx, v, out);
        }
        guard_fill(out, c->out_size, (uint32_t) it);

        portDISABLE_INTERRUPTS();
        const uint32_t t0 = esp_cpu_get_cycle_count();
        fn(c->ctx, v, out);
        const uint32_t t1 = esp_cpu_get_cycle_count();
        portENABLE_INTERRUPTS();

        if (it > 0 && t1 - t0 < best) {
            best = t1 - t0;
        }
        if (gold && (memcmp(out, gold[v], c->out_size) != 0 ||
                     !guard_ok(out, c->out_size, (uint32_t) it))) {
            ok = false;
        }
    }
    *min_cycles = best;
    return ok;
}

static const char *mem_name(bench_mem_t mem)
{
    return (mem == BENCH_MEM_EXT && bench_has_psram()) ? "ext" : "int";
}

int bench_run(const bench_case_t *c)
{
    int failures = 0;
    /* The output gets guard bytes on both sides; the expected outputs live in
     * internal RAM, out of the way of the timed accesses. */
    int8_t *out_raw = bench_alloc(c->out_size + 2 * BENCH_GUARD, c->mem);
    int8_t *gold[2] = {
        bench_alloc(c->out_size, BENCH_MEM_INT),
        bench_alloc(c->out_size, BENCH_MEM_INT),
    };
    if (!out_raw || !gold[0] || !gold[1]) {
        bench_skip(c->op, c->shape, c->mem);
        goto done;
    }
    int8_t *out = out_raw + BENCH_GUARD;

    for (int v = 0; v < 2; v++) {
        if (c->prep) {
            c->prep(c->ctx, v, gold[v]);
        }
        c->ref(c->ctx, v, gold[v]);
    }

    uint32_t ansi = 0;
    time_calls(c, c->ref, out, NULL, &ansi);

    for (int i = 0; i < c->n_variants; i++) {
        uint32_t opt = 0;
        const bool ok = time_calls(c, c->variants[i].call, out, gold, &opt);
        if (c->cycles) {
            c->cycles[i] = opt;
        }
        const uint32_t ratio_x100 = opt ? (uint32_t) (((uint64_t) ansi * 100 + opt / 2) / opt) : 0;
        printf("BENCH op=%s variant=%s mem=%s ansi=%" PRIu32 " opt=%" PRIu32
               " ratio=%" PRIu32 ".%02" PRIu32 " result=%s shape=\"%s\"\n",
               c->op, c->variants[i].name, mem_name(c->mem), ansi, opt,
               ratio_x100 / 100, ratio_x100 % 100, ok ? "ok" : "MISMATCH", c->shape);
        if (!ok) {
            failures++;
        }
    }
    s_cases++;

done:
    bench_free(out_raw);
    bench_free(gold[0]);
    bench_free(gold[1]);
    s_failures += failures;
    return failures;
}

void bench_skip(const char *op, const char *shape, bench_mem_t mem)
{
    printf("BENCH op=%s variant=- mem=%s result=skip shape=\"%s\" (no memory)\n",
           op, mem_name(mem), shape);
    s_skipped++;
}

int bench_cases_run(void)
{
    return s_cases;
}

int bench_failures(void)
{
    return s_failures;
}

int bench_skipped(void)
{
    return s_skipped;
}
