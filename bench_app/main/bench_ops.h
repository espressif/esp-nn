/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include "bench.h"

/* One function per op file. Each runs its case table through bench_run(). */
void bench_conv(void);
void bench_depthwise(void);
void bench_fully_connected(void);
void bench_basic_math(void);
void bench_pooling(void);
void bench_activations(void);
void bench_softmax(void);
void bench_mean(void);

/* Per-channel requantization typical of TFLite int8 models: a multiplier in
 * [2^30, 2^31) and a right shift of 6 to 9. */
static inline void bench_fill_quant(int32_t *mult, int32_t *shift, int n)
{
    for (int i = 0; i < n; i++) {
        mult[i] = (int32_t) (0x40000000u | (bench_rand() & 0x3fffffff));
        shift[i] = -bench_rand_range(6, 9);
    }
}

static inline void bench_fill_bias(int32_t *bias, int n)
{
    for (int i = 0; i < n; i++) {
        bias[i] = bench_rand_range(-1024, 1024);
    }
}
