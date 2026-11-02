/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Cycle benchmark harness for esp-nn kernels.
 *
 * An op file describes one case at a time: its shape, where its tensors
 * live, the ANSI reference call, and one or more variants of the kernel
 * under test. bench_run() then does the same for every case:
 *  - computes the expected output of two input variants with the reference;
 *  - times the reference and each variant, BENCH_ITERS calls each, the
 *    inputs alternating between the two variants, and keeps the minimum;
 *  - checks every output of every variant against the reference, and the
 *    guard bytes on both sides of the output;
 *  - prints one BENCH line per variant.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Timed calls per kernel and case. The first call is a warm-up. */
#ifndef BENCH_ITERS
#define BENCH_ITERS 9
#endif

/* Guard bytes on each side of the output buffer. */
#define BENCH_GUARD 64

/* ------------------------------------------------------------------ PRNG -- */

/* Sets the run seed. bench_seed_case() then derives each case's seed from it,
 * so a case's data does not depend on the cases before it. */
void bench_seed(uint32_t seed);
/* Reseeds from the run seed, the op and the shape. Call before the fills. */
void bench_seed_case(const char *op, const char *shape);
uint32_t bench_rand(void);
/* Uniform in [lo, hi], both inclusive. */
int32_t bench_rand_range(int32_t lo, int32_t hi);
void bench_fill(int8_t *buf, size_t len);

/* --------------------------------------------------------------- Buffers -- */

typedef enum {
    BENCH_MEM_INT,      /* internal RAM */
    BENCH_MEM_EXT,      /* PSRAM; internal RAM when the board has none */
} bench_mem_t;

/* 16-byte aligned buffer of `size` bytes in the given memory. Returns NULL
 * on failure. Free with bench_free(). */
void *bench_alloc(size_t size, bench_mem_t mem);
void bench_free(void *p);
/* True when the board has PSRAM for BENCH_MEM_EXT buffers. */
bool bench_has_psram(void);

/* -------------------------------------------------------------- Running -- */

/* Calls the kernel once for input variant v (0 or 1), writing into out.
 * Only the kernel call goes here: everything in it is timed. */
typedef void (*bench_fn)(void *ctx, int v, int8_t *out);

/* One implementation of the op under test. Ops with several ways to run the
 * same computation (a different dispatch, a contributed kernel) list one
 * variant each; every variant is timed and checked against the reference. */
typedef struct {
    const char *name;
    bench_fn call;
} bench_variant_t;

typedef struct {
    const char *op;             /* e.g. "conv" */
    const char *shape;          /* free text, printed last on the line */
    bench_mem_t mem;            /* where the op put its tensors */
    size_t out_size;            /* output bytes per call */
    bench_fn ref;               /* the ANSI reference */
    /* Optional, untimed, before every call: for in-place kernels, copy the
     * input of variant v into out. */
    bench_fn prep;
    const bench_variant_t *variants;
    int n_variants;
    void *ctx;
    /* Optional: n_variants entries, set to each variant's minimum cycles. */
    uint32_t *cycles;
} bench_case_t;

/* Runs one case and prints its BENCH lines. Returns the number of variants
 * that failed (wrong output or damaged guard bytes). */
int bench_run(const bench_case_t *c);

/* Prints the skip line of a case that could not run, and counts it. */
void bench_skip(const char *op, const char *shape, bench_mem_t mem);

/* Totals over all bench_run() and bench_skip() calls so far. */
int bench_cases_run(void);
int bench_failures(void);
int bench_skipped(void);

#ifdef __cplusplus
}
#endif
