/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Shared buffer for the layout sweeps: a guarded region whose data pointer can
 * be offset by an arbitrary number of bytes after the usual 16-byte rounding,
 * so the requested misalignment is exact on any allocator.
 *
 * Layout: [storage ... 16 bytes 0xA5][data, size bytes][16 bytes 0x5A ...]
 * The two patterns differ so an underrun and an overrun are distinguishable.
 */
#define TEST_ALIGN_GUARD 16

struct test_align_buf {
    int8_t *storage;
    int8_t *data;
    int size;
};

static inline int test_align_buf_init(struct test_align_buf *b, int size, int misalign)
{
    /* 15 for rounding + guard before + misalign + size + guard after */
    size_t total = (size_t)size + (size_t)misalign + TEST_ALIGN_GUARD * 2 + 15;
    b->storage = malloc(total);
    b->data = NULL;
    b->size = size;
    if (b->storage == NULL) {
        return -1;
    }
    uintptr_t base = ((uintptr_t)b->storage + TEST_ALIGN_GUARD + 15) & ~(uintptr_t)15;
    b->data = (int8_t *)base + misalign;
    memset(b->data - TEST_ALIGN_GUARD, 0xA5, TEST_ALIGN_GUARD);
    memset(b->data + size, 0x5A, TEST_ALIGN_GUARD);
    return 0;
}

static inline void test_align_buf_free(struct test_align_buf *b)
{
    free(b->storage);
    b->storage = NULL;
    b->data = NULL;
}

/* 1 when both bands are intact, 0 otherwise. */
static inline int test_align_guard_ok(const struct test_align_buf *b)
{
    for (int i = 1; i <= TEST_ALIGN_GUARD; i++) {
        if (b->data[-i] != (int8_t)0xA5 || b->data[b->size + i - 1] != (int8_t)0x5A) {
            return 0;
        }
    }
    return 1;
}

static inline void test_align_fill_random(int8_t *p, int size)
{
    for (int i = 0; i < size; i++) {
        p[i] = (int8_t)(rand() % 256 - 128);
    }
}
