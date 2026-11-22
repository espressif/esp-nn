/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 max pool.
 *
 * The C part clips each window to the input. It makes one leaf call for the
 * unclipped interior run and one per border pixel, so the SIMD leaf
 * (esp_nn_max_pool_s8_esp32s3.S) has no bounds checks. Rows whose windows
 * are not clipped vertically share each call, to save the call overhead.
 * When the output is one column wide, such a band is a single call.
 *
 * The leaf uses 64-bit loads when channels % 8 == 0, which need 8-byte
 * buffers, and 32-bit loads otherwise, which need 4-byte ones. With a 16-byte
 * aligned input and channels % 16 == 0 it uses 128-bit loads. These loads
 * ignore the low address bits, so a buffer below that alignment would be read
 * and written lane-shifted. Anything else goes to the reference: channels
 * % 4 != 0, lower alignment, or activation limits outside int8. Tensor data
 * is normally 16-byte aligned, so the vector paths stay the common ones.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <esp_nn_ansi_headers.h>
#include <esp_memory_utils.h>
#include <common_functions.h>
#include "esp_nn_max_pool_s8_esp32s3.h"

typedef struct {
    int32_t n32;        /* 32-lane chunks per pixel */
    int32_t n16;        /* 16-lane chunks per pixel, 0 or 1 */
    int32_t n8;         /* 8-lane chunks per pixel, 0 or 1 */
    int32_t n4;         /* 4-lane chunks per pixel */
    int32_t channels;
    int32_t in_step;    /* input bytes between output pixels of a row */
    int32_t row_bytes;  /* input bytes per input row */
    int8_t act_min;
    int8_t act_max;
    int32_t n16_full;   /* channels / 16 if channels % 16 == 0, else 0 */
    int32_t in_row_step;  /* input bytes between output rows */
    int32_t out_row_step; /* output bytes between output rows */
    int32_t rows;       /* rows for esp_nn_max_pool_s8_esp32s3_rows */
    int32_t rows_left;  /* esp_nn_max_pool_s8_esp32s3_rows scratch */
} s3mp_args_t;

/* The assembly reads these at the offsets in esp_nn_max_pool_s8_esp32s3.h */
_Static_assert(offsetof(s3mp_args_t, n32) == S3MP_N32, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, n16) == S3MP_N16, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, n8) == S3MP_N8, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, n4) == S3MP_N4, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, channels) == S3MP_CHANNELS, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, in_step) == S3MP_IN_STEP, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, row_bytes) == S3MP_ROW_BYTES, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, act_min) == S3MP_ACT_MIN, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, act_max) == S3MP_ACT_MAX, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, n16_full) == S3MP_N16_FULL, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, in_row_step) == S3MP_IN_ROW_STEP, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, out_row_step) == S3MP_OUT_ROW_STEP, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, rows) == S3MP_ROWS, "s3mp_args_t layout");
_Static_assert(offsetof(s3mp_args_t, rows_left) == S3MP_ROWS_LEFT, "s3mp_args_t layout");

typedef void s3mp_fn(s3mp_args_t *a, const int8_t *in, int8_t *out,
                     int n, int fw, int fh);
extern s3mp_fn esp_nn_max_pool_s8_esp32s3_run, esp_nn_max_pool_s8_esp32s3_rows;
extern void esp_nn_max_pool_s8_esp32s3_sliding_row_asm(
    const int8_t *input, int8_t *output, uint32_t width, uint32_t channels,
    uint32_t row_bytes, uint32_t rows, uint32_t pad,
    int32_t activation_min, int32_t activation_max);

/* An output pixel whose window misses the input entirely gets the clamped
 * INT8_MIN, as in the reference. That happens when the padding is as wide as
 * the filter, or when the output is larger than the padded input covers.
 * TFLite's SAME and VALID shapes never have one, so this stays out of line. */
static __attribute__((noinline, cold)) void fill_empty(int8_t *out, int n,
                                                      int32_t act_min, int32_t act_max)
{
    memset(out, min(max(INT8_MIN, act_min), act_max), (size_t) n);
}

/* The same for one border column of a band: n bytes in each of rows rows */
static __attribute__((noinline, cold)) void fill_empty_rows(int8_t *out, int n, int rows,
                                                           int row_step, int32_t act_min,
                                                           int32_t act_max)
{
    for (int r = 0; r < rows; r++) {
        fill_empty(out + r * row_step, n, act_min, act_max);
    }
}

void esp_nn_max_pool_s8_esp32s3(const int8_t *input,
                                const uint16_t input_wd,
                                const uint16_t input_ht,
                                int8_t *output,
                                const uint16_t output_wd,
                                const uint16_t output_ht,
                                const uint16_t stride_wd,
                                const uint16_t stride_ht,
                                const uint16_t filter_wd,
                                const uint16_t filter_ht,
                                const uint16_t pad_wd,
                                const uint16_t pad_ht,
                                const int32_t activation_min,
                                const int32_t activation_max,
                                const uint16_t channels)
{
    const uintptr_t bufs = (uintptr_t) input | (uintptr_t) output;
    const uintptr_t need = (channels % 8 == 0) ? 7u : 3u;
    /* The leaf clamps with int8 limits. Others, which TFLite never passes,
     * take the reference so the result stays exact. */
    const bool limits_int8 = activation_min >= INT8_MIN && activation_min <= INT8_MAX &&
                             activation_max >= INT8_MIN && activation_max <= INT8_MAX;
    if (channels % 4 != 0 || (bufs & need) != 0 || filter_wd == 0 || !limits_int8) {
        esp_nn_max_pool_s8_ansi(input, input_wd, input_ht, output, output_wd,
                                output_ht, stride_wd, stride_ht, filter_wd, filter_ht,
                                pad_wd, pad_ht, activation_min, activation_max,
                                channels);
        return;
    }

    /* Reuse column maxima for 3x3 and 5x5 stride-1 SAME pooling. */
    if (channels != 0 && ((bufs | channels) & 15u) == 0 &&
            (filter_wd == 3 || filter_wd == 5) && filter_ht == filter_wd &&
            stride_wd == 1 && stride_ht == 1 &&
            pad_wd == filter_wd / 2 && pad_ht == pad_wd &&
            output_wd == input_wd && output_ht == input_ht &&
            input_wd >= filter_wd && input_ht >= filter_ht) {
        const uint32_t row_bytes = (uint32_t) input_wd * channels;
        const uintptr_t in_addr = (uintptr_t) input;
        const uintptr_t out_addr = (uintptr_t) output;
        const uintptr_t distance = in_addr < out_addr ? out_addr - in_addr : in_addr - out_addr;
        /* Reuse requires disjoint input and output tensors of the same size. */
        if (distance >= (uint64_t) row_bytes * input_ht) {
            for (int y = 0; y < output_ht; y++) {
                const int top = y < pad_ht ? 0 : y - pad_ht;
                const int end = y + pad_ht + 1;
                const int bottom = end < input_ht ? end : input_ht;
                const int8_t *in_row = input + top * row_bytes;
                int8_t *out_row = output + y * row_bytes;
                for (int ch = 0; ch < channels; ch += 16) {
                    esp_nn_max_pool_s8_esp32s3_sliding_row_asm(
                        in_row + ch, out_row + ch, input_wd, channels, row_bytes,
                        bottom - top, pad_wd, activation_min, activation_max);
                }
            }
            return;
        }
    }

    const int ch = channels;
    /* Field by field: an initializer would also zero the row fields */
    s3mp_args_t a;
    a.n32 = (ch % 8 == 0) ? ch / 32 : 0;
    a.n16 = (ch % 8 == 0) ? (ch / 16) & 1 : 0;
    a.n8 = (ch % 8 == 0) ? (ch / 8) & 1 : 0;
    a.n4 = (ch % 8 == 0) ? 0 : ch / 4;
    a.channels = ch;
    a.row_bytes = input_wd * ch;
    a.in_row_step = stride_ht * a.row_bytes;
    a.out_row_step = output_wd * ch;
    /* One output column: a band is one pixel run down the column */
    a.in_step = (output_wd == 1) ? a.in_row_step : stride_wd * ch;
    a.act_min = (int8_t) activation_min;
    a.act_max = (int8_t) activation_max;
    a.n16_full = (ch % 16 == 0) ? ch / 16 : 0;

    /* Output columns [ox_lo, ox_hi) have windows fully inside the row */
    int ox_lo = 0;
    while (ox_lo < output_wd && ox_lo * stride_wd < pad_wd) {
        ox_lo++;
    }
    int ox_hi = output_wd;
    while (ox_hi > ox_lo && (ox_hi - 1) * stride_wd - pad_wd + filter_wd > input_wd) {
        ox_hi--;
    }

    /*
     * Unclipped rows share each leaf call, so each border column is walked
     * down the band. PSRAM goes through the cache, where that would fetch
     * the column's lines again, so there only rows without border columns
     * share.
     */
    const bool bands = (ox_lo == 0 && ox_hi == output_wd) || !esp_ptr_external_ram(input);
    int rows;
    for (int oy = 0; oy < output_ht; oy += rows) {
        const int y0 = oy * stride_ht - pad_ht;
        const int ys = max(y0, 0);
        const int fh = min(y0 + filter_ht, input_ht) - ys;
        if (__builtin_expect(fh <= 0, 0)) {
            fill_empty(output + oy * output_wd * ch, output_wd * ch, activation_min, activation_max);
            rows = 1;
            continue;
        }
        rows = 1;
        if (fh == filter_ht && oy + 1 < output_ht && bands) {
            while (oy + rows < output_ht &&
                   y0 + rows * stride_ht + filter_ht <= input_ht) {
                rows++;
            }
            a.rows = rows;
            if (output_wd == 1) {
                const int fw = min(filter_wd - pad_wd, input_wd);
                if (__builtin_expect(fw <= 0, 0)) {
                    fill_empty(output + oy * ch, rows * ch, activation_min, activation_max);
                    continue;
                }
                esp_nn_max_pool_s8_esp32s3_run(&a, input + ys * a.row_bytes, output + oy * ch, rows, fw, fh);
                continue;
            }
        }
        s3mp_fn *const run = (rows > 1) ? esp_nn_max_pool_s8_esp32s3_rows : esp_nn_max_pool_s8_esp32s3_run;
        const int8_t *in_row = input + ys * a.row_bytes;
        int8_t *out = output + oy * output_wd * ch;

        for (int ox = 0; ox < output_wd; ox++) {
            if (ox == ox_lo && ox_hi > ox_lo) {
                const int n = ox_hi - ox_lo;
                run(&a, in_row + (ox_lo * stride_wd - pad_wd) * ch, out, n,
                    filter_wd, fh);
                out += n * ch;
                ox = ox_hi - 1;
                continue;
            }
            const int x0 = ox * stride_wd - pad_wd;
            const int xs = max(x0, 0);
            const int fw = min(x0 + filter_wd, input_wd) - xs;
            if (__builtin_expect(fw <= 0, 0)) {
                /* In a band this pixel stands for its column in every row */
                fill_empty_rows(out, ch, rows, output_wd * ch, activation_min, activation_max);
            } else {
                run(&a, in_row + xs * ch, out, 1, fw, fh);
            }
            out += ch;
        }
    }
}
