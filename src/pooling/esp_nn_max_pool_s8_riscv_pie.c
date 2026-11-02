/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <common_functions.h>
#include <esp_nn_ansi_headers.h>

/* Scalar max over one clipped window, used only near the buffer ends. */
static void max_pool_pixel_scalar(const int8_t *win, int32_t fw, int32_t fh,
                                  int32_t row_stride, int32_t channels,
                                  int8_t act_min, int8_t act_max, int8_t *out)
{
    for (int32_t ch = 0; ch < channels; ch++) {
        int8_t result = act_min;
        for (int32_t fy = 0; fy < fh; fy++) {
            const int8_t *p = win + fy * row_stride + ch;
            for (int32_t fx = 0; fx < fw; fx++, p += channels) {
                result = max(*p, result);
            }
        }
        out[ch] = min(result, act_max);
    }
}

/*
 * Unclipped 2x2 and 3x3 windows, and 3x3 windows clipped to two rows. All
 * taps are unrolled, with a hardware loop over n 16-byte blocks. Each
 * iteration moves the window by in_step and the output by out_step. For the
 * blocks of one pixel both are 16. For one block offset across a row of
 * pixels they are stride * channels and channels.
 * x31 = channels, x29 = step to the next row, x27 = step from the last tap
 * to the first tap of the next iteration, x26 = out_step.
 */
#define LD_STEP(q, r) "esp.vld.128.xp " q ", x30, " r " \n\t"

static inline void max_pool_2x2_blocks(const int8_t *p, int8_t *o, int32_t n,
                                       int32_t channels, int32_t row_stride,
                                       int32_t in_step, int32_t out_step)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x31, %[ch]       \n\t"
        "mv              x29, %[next]     \n\t"
        "mv              x27, %[back]     \n\t"
        "mv              x26, %[ostep]    \n\t"
        "esp.lp.setup    0, %[n], 1f      \n\t"
        LD_STEP("q0", "x31")
        LD_STEP("q1", "x29")
        LD_STEP("q2", "x31")
        LD_STEP("q3", "x27")
        "esp.vmax.s8     q0, q0, q4       \n\t"
        "esp.vmax.s8     q1, q1, q2       \n\t"
        "esp.vmax.s8     q0, q0, q3       \n\t"
        "esp.vmax.s8     q0, q0, q1       \n\t"
        "esp.vmin.s8     q0, q0, q5       \n\t"
        ESP_NN_HWLOOP_LAST("esp.vst.128.xp  q0, x28, x26", 0)
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [n] "r"(n), [ch] "r"(channels),
          [next] "r"(row_stride - channels),
          [back] "r"(in_step - row_stride - channels), [ostep] "r"(out_step)
        : "x26", "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

static inline void max_pool_3x3_blocks(const int8_t *p, int8_t *o, int32_t n,
                                       int32_t channels, int32_t row_stride,
                                       int32_t in_step, int32_t out_step)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x31, %[ch]       \n\t"
        "mv              x29, %[next]     \n\t"
        "mv              x27, %[back]     \n\t"
        "mv              x26, %[ostep]    \n\t"
        "esp.lp.setup    0, %[n], 1f      \n\t"
        LD_STEP("q0", "x31")
        LD_STEP("q1", "x31")
        LD_STEP("q2", "x29")
        LD_STEP("q3", "x31")
        "esp.vmax.s8     q0, q0, q4       \n\t"
        LD_STEP("q6", "x31")
        "esp.vmax.s8     q1, q1, q2       \n\t"
        LD_STEP("q7", "x29")
        "esp.vmax.s8     q0, q0, q3       \n\t"
        LD_STEP("q2", "x31")
        "esp.vmax.s8     q1, q1, q6       \n\t"
        LD_STEP("q3", "x31")
        "esp.vmax.s8     q0, q0, q7       \n\t"
        LD_STEP("q6", "x27")
        "esp.vmax.s8     q1, q1, q2       \n\t"
        "esp.vmax.s8     q0, q0, q3       \n\t"
        "esp.vmax.s8     q1, q1, q6       \n\t"
        "esp.vmax.s8     q0, q0, q1       \n\t"
        "esp.vmin.s8     q0, q0, q5       \n\t"
        ESP_NN_HWLOOP_LAST("esp.vst.128.xp  q0, x28, x26", 0)
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [n] "r"(n), [ch] "r"(channels),
          [next] "r"(row_stride - 2 * channels),
          [back] "r"(in_step - 2 * row_stride - 2 * channels), [ostep] "r"(out_step)
        : "x26", "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

/* 3x3 filter clipped to two input rows, at the top or bottom row of SAME padding. */
static inline void max_pool_3x2_blocks(const int8_t *p, int8_t *o, int32_t n,
                                       int32_t channels, int32_t row_stride,
                                       int32_t in_step, int32_t out_step)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x31, %[ch]       \n\t"
        "mv              x29, %[next]     \n\t"
        "mv              x27, %[back]     \n\t"
        "mv              x26, %[ostep]    \n\t"
        "esp.lp.setup    0, %[n], 1f      \n\t"
        LD_STEP("q0", "x31")
        LD_STEP("q1", "x31")
        LD_STEP("q2", "x29")
        LD_STEP("q3", "x31")
        "esp.vmax.s8     q0, q0, q4       \n\t"
        LD_STEP("q6", "x31")
        "esp.vmax.s8     q1, q1, q2       \n\t"
        LD_STEP("q7", "x27")
        "esp.vmax.s8     q0, q0, q3       \n\t"
        "esp.vmax.s8     q1, q1, q6       \n\t"
        "esp.vmax.s8     q0, q0, q7       \n\t"
        "esp.vmax.s8     q0, q0, q1       \n\t"
        "esp.vmin.s8     q0, q0, q5       \n\t"
        ESP_NN_HWLOOP_LAST("esp.vst.128.xp  q0, x28, x26", 0)
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [n] "r"(n), [ch] "r"(channels),
          [next] "r"(row_stride - 2 * channels),
          [back] "r"(in_step - row_stride - 2 * channels), [ostep] "r"(out_step)
        : "x26", "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

/*
 * for (t = cnt; t; t--, ptr += step) q0 = max(q0, ptr[0..15]); cnt >= 1.
 * The body is two instructions. On the builds that pad hardware loops
 * (ESP_NN_HWLOOP_PADDED) a padded loop runs six per tap, so they use a
 * software loop, which runs four. Clobbers t0.
 */
#if ESP_NN_HWLOOP_PADDED
#define MAX_POOL_TAP_LOOP(cnt, ptr, step)                       \
    "mv              t0, " cnt "\n\t"                            \
    "1:                               \n\t"                     \
    "esp.vld.128.xp  q1, " ptr ", " step "\n\t"                  \
    "esp.vmax.s8     q0, q0, q1       \n\t"                     \
    "addi            t0, t0, -1       \n\t"                     \
    "bnez            t0, 1b           \n\t"
#else
#define MAX_POOL_TAP_LOOP(cnt, ptr, step)                       \
    "esp.lp.setup    0, " cnt ", 1f\n\t"                         \
    "esp.vld.128.xp  q1, " ptr ", " step "\n\t"                  \
    ESP_NN_HWLOOP_LAST("esp.vmax.s8     q0, q0, q1", 0)
#endif

/*
 * Window that is a single row or column of n_taps taps, step bytes apart.
 * MAX_POOL_TAP_LOOP over the taps, software loop over n iterations that move
 * the window by in_step and the output by out_step (as for 2x2/3x3).
 * Only loop 0 is used. The P4/S31 emulators reject esp.lp.setup 1.
 */
static inline void max_pool_line_blocks(const int8_t *p, int8_t *o, int32_t n,
                                        int32_t n_taps, int32_t step,
                                        int32_t in_step, int32_t out_step)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x31, %[step]     \n\t"
        "mv              x29, %[back]     \n\t"
        "mv              x27, %[n]        \n\t"
        "mv              x26, %[ostep]    \n\t"
        "2:                               \n\t"
        "esp.vmax.s8     q0, q4, q4       \n\t"  /* q0 = act_min */
        MAX_POOL_TAP_LOOP("%[taps]", "x30", "x31")
        "esp.vmin.s8     q0, q0, q5       \n\t"
        "add             x30, x30, x29    \n\t"
        "esp.vst.128.xp  q0, x28, x26     \n\t"
        "addi            x27, x27, -1     \n\t"
        "bnez            x27, 2b          \n\t"
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [n] "r"(n), [taps] "r"(n_taps),
          [step] "r"(step), [back] "r"(in_step - n_taps * step), [ostep] "r"(out_step)
        : "t0", "x26", "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

/* Two-tap line, a 1x2 or 2x1 window. Unrolled, with a hardware loop over n iterations. */
static inline void max_pool_2tap_blocks(const int8_t *p, int8_t *o, int32_t n, int32_t step,
                                        int32_t in_step, int32_t out_step)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x31, %[step]     \n\t"
        "mv              x29, %[back]     \n\t"
        "mv              x27, %[ostep]    \n\t"
        "esp.lp.setup    0, %[n], 1f      \n\t"
        LD_STEP("q0", "x31")
        LD_STEP("q1", "x29")
        "esp.vmax.s8     q0, q0, q4       \n\t"
        "esp.vmax.s8     q0, q0, q1       \n\t"
        "esp.vmin.s8     q0, q0, q5       \n\t"
        ESP_NN_HWLOOP_LAST("esp.vst.128.xp  q0, x28, x27", 0)
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [n] "r"(n), [step] "r"(step), [back] "r"(in_step - step),
          [ostep] "r"(out_step)
        : "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

static void max_pool_band_flat(const int8_t *win, int8_t *out, int32_t total,
                               int32_t filter_wd, int32_t filter_ht, int32_t fh,
                               int32_t channels, int32_t row_stride);
static void max_pool_window_wide(const int8_t *win, int8_t *out, int32_t n_blk,
                                 int32_t last_off, int32_t n_in, int32_t n_out,
                                 int32_t step_in, int32_t step_out);

/**
 * Max pooling for s8 using the P4/S31 PIE SIMD.
 *
 * Paths, in 16-channel blocks:
 * - Band: the unclipped pixels of a row in one asm call per block offset.
 *   It covers 2x2, 3x3, 3x3 clipped to two rows, and 1xK lines. With stride 1
 *   the band is one byte run (flat).
 * - Line: a window that is one row or column after clipping.
 * - Generic: any other clipped window, with a hardware loop along the longer
 *   side and a software loop along the other.
 * - Scalar: pixels near the buffer ends, see below.
 * The running max starts at activation_min, which folds in the lower clamp.
 *
 * Channel tails:
 * - channels >= 16: the last block overlaps the previous one, ending at the
 *   last channel. Max is idempotent, so the overlap is harmless.
 * - channels < 16: one 16-byte block that spills into the next pixel. Pixels
 *   go in increasing order, so the next pixel overwrites the spilled bytes.
 *   Pixels whose loads or store would leave the buffers use the scalar path.
 */
void esp_nn_max_pool_s8_riscv_pie(const int8_t *input,
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
    /* The vector clamp uses int8 limits. Others, which TFLite never passes,
     * take the reference so the result stays exact. */
    if (activation_min < INT8_MIN || activation_min > INT8_MAX ||
            activation_max < INT8_MIN || activation_max > INT8_MAX) {
        esp_nn_max_pool_s8_ansi(input, input_wd, input_ht, output, output_wd,
                                output_ht, stride_wd, stride_ht, filter_wd, filter_ht,
                                pad_wd, pad_ht, activation_min, activation_max,
                                channels);
        return;
    }

    /* A width-1 input is a single row in memory, and so is its output. Pool it as one row. */
    if (input_wd == 1 && output_wd == 1 && input_ht > 1) {
        esp_nn_max_pool_s8_riscv_pie(input, input_ht, 1, output, output_ht, 1,
                                     stride_ht, stride_wd, filter_ht, filter_wd, pad_ht, pad_wd,
                                     activation_min, activation_max, channels);
        return;
    }

    ESP_NN_PIE_ENABLE();

    int8_t act_min_val = (int8_t) activation_min;
    int8_t act_max_val = (int8_t) activation_max;

    asm volatile (
        "mv              x30, %0     \n\t"
        "esp.vldbc.8.ip  q4, x30, 0 \n\t"  /* q4 = broadcast(activation_min) */
        "mv              x30, %1     \n\t"
        "esp.vldbc.8.ip  q5, x30, 0 \n\t"  /* q5 = broadcast(activation_max) */
        :: "r"(&act_min_val), "r"(&act_max_val)
        : "x30", "memory"
    );

    const int32_t row_stride = input_wd * channels;
    /* Block count per pixel, and the offset of the last (possibly overlapping) block */
    const int32_t n_blk = (channels + 15) >> 4;
    const int32_t last_off = channels >= 16 ? channels - 16 : 0;
    /* Blocks done 16 apart, and whether one partial block follows */
    const int32_t has_tail = channels & 15;
    const int32_t n_full = channels >> 4;
    /* Loads must stay in the 16-byte blocks that hold input bytes */
    const uintptr_t in_lim = (uintptr_t) (input + input_ht * row_stride);
    const int8_t *out_end = output + output_wd * output_ht * channels;

    /* Output x range whose windows fit the row width, for the bands below */
    const int32_t fits_wd = input_wd + pad_wd >= filter_wd;
    const int32_t band_lo_x = (pad_wd + stride_wd - 1) / stride_wd;
    const int32_t band_hi_x = fits_wd ?
                              min(output_wd, (input_wd + pad_wd - filter_wd) / stride_wd + 1) : 0;
    /* With stride 1 the band's output is one byte run, done in 16-byte blocks */
    const int32_t flat_band = stride_wd == 1 && (band_hi_x - band_lo_x) * channels >= 16;

    /* A window that misses the input gives INT8_MIN, clamped, as in the
     * reference. That happens when the padding is as wide as the filter. */
    const int8_t empty_val = min(act_min_val, act_max_val);

    int8_t *out_ptr = output;
    int32_t base_y = -pad_ht;
    for (int32_t out_y = 0; out_y < output_ht; out_y++, base_y += stride_ht) {
        const int32_t fy_start = max(0, -base_y);
        const int32_t fy_end = min(filter_ht, input_ht - base_y);
        const int32_t fh = fy_end - fy_start;
        if (fh <= 0) {
            /* The hardware loops need a count above 0 */
            memset(out_ptr, empty_val, (size_t) output_wd * channels);
            out_ptr += output_wd * channels;
            continue;
        }

        /*
         * Square 2x2/3x3 filter on an unclipped row, or a 3x3 filter on a row
         * clipped to two input rows, or a 1xK line filter: the pixels whose
         * window fits the row width form one band. The band takes one asm call per
         * block offset, with the hardware loop over its pixels.
         * For channels < 16 the band ends before pixels whose 16-byte
         * accesses would leave the buffers (not needed for a flat band,
         * whose accesses stay inside the band's windows).
         */
        int32_t band_lo = 0, band_hi = 0, flat = 0;
        if (((filter_wd == filter_ht && (fh == filter_wd || (filter_wd == 3 && fh == 2)) &&
                (filter_wd == 2 || filter_wd == 3)) || (filter_ht == 1 && filter_wd >= 2)) &&
                fits_wd) {
            band_lo = band_lo_x;
            band_hi = band_hi_x;
            flat = flat_band;
            /* Integer offsets, so that no pointer is formed outside the buffers */
            const int32_t tap_row = (base_y + fy_end - 1) * row_stride +
                                    (filter_wd - 1 - pad_wd) * channels;
            const int32_t out_left = out_end - out_ptr;
            while (!flat && channels < 16 && band_hi > band_lo) {
                const int8_t *last_tap = input + (tap_row + (band_hi - 1) * stride_wd * channels);
                if ((band_hi - 1) * channels + 16 <= out_left &&
                        (((uintptr_t) last_tap + 15) & ~(uintptr_t) 15) < in_lim) {
                    break;
                }
                band_hi--;
            }
        }

        int32_t base_x = -pad_wd;
        for (int32_t out_x = 0; out_x < output_wd; out_x++, base_x += stride_wd,
                out_ptr += channels) {
            if (out_x == band_lo && band_hi > band_lo) {
                const int32_t n = band_hi - band_lo;
                const int8_t *win = input + (base_y + fy_start) * row_stride + base_x * channels;
                if (flat) {
                    max_pool_band_flat(win, out_ptr, n * channels, filter_wd, filter_ht, fh,
                                       channels, row_stride);
                    out_x += n - 1;
                    base_x += n - 1;
                    out_ptr += (n - 1) * channels;
                    continue;
                }
                /* The line filter is tested outside the block loop. Inside, it slows 2x2 bands. */
                for (int32_t blk = 0; blk < n_blk && filter_ht == 1; blk++) {
                    const int32_t off = blk == n_blk - 1 ? last_off : blk << 4;
                    if (filter_wd == 2) {
                        max_pool_2tap_blocks(win + off, out_ptr + off, n, channels,
                                             stride_wd * channels, channels);
                    } else {
                        max_pool_line_blocks(win + off, out_ptr + off, n, filter_wd, channels,
                                             stride_wd * channels, channels);
                    }
                }
                for (int32_t blk = 0; blk < n_blk && filter_ht > 1; blk++) {
                    const int32_t off = blk == n_blk - 1 ? last_off : blk << 4;
                    if (filter_wd == 2) {
                        max_pool_2x2_blocks(win + off, out_ptr + off, n, channels, row_stride,
                                            stride_wd * channels, channels);
                    } else if (fh == 3) {
                        max_pool_3x3_blocks(win + off, out_ptr + off, n, channels, row_stride,
                                            stride_wd * channels, channels);
                    } else {
                        max_pool_3x2_blocks(win + off, out_ptr + off, n, channels, row_stride,
                                            stride_wd * channels, channels);
                    }
                }
                out_x += n - 1;
                base_x += (n - 1) * stride_wd;
                out_ptr += (n - 1) * channels;
                continue;
            }
            const int32_t fx_start = max(0, -base_x);
            const int32_t fx_end = min(filter_wd, input_wd - base_x);
            const int32_t fw = fx_end - fx_start;
            if (fw <= 0) {
                memset(out_ptr, empty_val, channels);
                continue;
            }
            const int8_t *win = input + (base_y + fy_start) * row_stride
                                + (base_x + fx_start) * channels;

            if (channels < 16) {
                const int8_t *last_tap = win + (fh - 1) * row_stride + (fw - 1) * channels;
                if (out_end - out_ptr < 16 ||
                        (((uintptr_t) last_tap + 15) & ~(uintptr_t) 15) >= in_lim) {
                    max_pool_pixel_scalar(win, fw, fh, row_stride, channels,
                                          act_min_val, act_max_val, out_ptr);
                    continue;
                }
            }

            /* Inner hardware loop runs along the longer window side */
            int32_t n_in = fw, n_out = fh, step_in = channels, step_out = row_stride - fw * channels;
            if (fw < fh) {
                n_in = fh;
                n_out = fw;
                step_in = row_stride;
                step_out = channels - fh * row_stride;
            }
            if (n_out == 1 && n_in == 2) {
                if (n_full) {
                    max_pool_2tap_blocks(win, out_ptr, n_full, step_in, 16, 16);
                }
                if (has_tail) {
                    max_pool_2tap_blocks(win + last_off, out_ptr + last_off, 1, step_in, 16, 16);
                }
                continue;
            }
            if (n_out == 1) {
                if (n_full) {
                    max_pool_line_blocks(win, out_ptr, n_full, n_in, step_in, 16, 16);
                }
                if (has_tail) {
                    max_pool_line_blocks(win + last_off, out_ptr + last_off, 1, n_in, step_in, 16, 16);
                }
                continue;
            }

            if (n_blk >= 3) {
                max_pool_window_wide(win, out_ptr, n_blk, last_off, n_in, n_out,
                                     step_in, step_out);
                continue;
            }
            for (int32_t blk = 0; blk < n_blk; blk++) {
                const int32_t off = blk == n_blk - 1 ? last_off : blk << 4;
                asm volatile (
                    ESP_NN_HWLOOP_BEGIN
                    "mv              x30, %[p]        \n\t"
                    "mv              x31, %[ch]       \n\t"
                    "mv              x29, %[skip]     \n\t"
                    "mv              x28, %[o]        \n\t"
                    "mv              x27, %[n_out]    \n\t"
                    "esp.vmax.s8     q0, q4, q4       \n\t"  /* q0 = act_min */
                    "2:                               \n\t"
                    MAX_POOL_TAP_LOOP("%[n_in]", "x30", "x31")
                    "add             x30, x30, x29    \n\t"
                    "addi            x27, x27, -1     \n\t"
                    "bnez            x27, 2b          \n\t"
                    "esp.vmin.s8     q0, q0, q5       \n\t"
                    "esp.vst.128.ip  q0, x28, 0       \n\t"
                    ESP_NN_HWLOOP_END
                    :
                    : [p] "r"(win + off), [ch] "r"(step_in),
                      [skip] "r"(step_out), [o] "r"(out_ptr + off),
                      [n_out] "r"(n_out), [n_in] "r"(n_in)
                    : "t0", "x27", "x28", "x29", "x30", "x31", "memory"
                );
            }
        }
    }
}

/*
 * Generic clipped window for three blocks at offsets 0, 16 and d (16 < d <= 32).
 * Three accumulators hide the load-to-use latency of the one-block loop.
 * Hardware loop over n_in taps step_in apart, software loop over n_out lines.
 */
static inline void max_pool_window_3blk(const int8_t *p, int8_t *o, int32_t d,
                                        int32_t n_in, int32_t n_out,
                                        int32_t step_in, int32_t step_out)
{
    asm volatile (
        ESP_NN_HWLOOP_BEGIN
        "mv              x30, %[p]        \n\t"
        "mv              x28, %[o]        \n\t"
        "mv              x26, %[d16]      \n\t"
        "mv              x31, %[step]     \n\t"
        "mv              x29, %[skip]     \n\t"
        "mv              x27, %[n_out]    \n\t"
        "esp.vmax.s8     q0, q4, q4       \n\t"  /* accumulators = act_min */
        "esp.vmax.s8     q3, q4, q4       \n\t"
        "esp.vmax.s8     q7, q4, q4       \n\t"
        "2:                               \n\t"
        "esp.lp.setup    0, %[n_in], 1f   \n\t"
        "esp.vld.128.ip  q1, x30, 16      \n\t"
        "esp.vld.128.xp  q2, x30, x26     \n\t"
        "esp.vld.128.xp  q6, x30, x31     \n\t"
        "esp.vmax.s8     q0, q0, q1       \n\t"
        "esp.vmax.s8     q3, q3, q2       \n\t"
        ESP_NN_HWLOOP_LAST("esp.vmax.s8     q7, q7, q6", 0)
        "add             x30, x30, x29    \n\t"
        "addi            x27, x27, -1     \n\t"
        "bnez            x27, 2b          \n\t"
        "esp.vmin.s8     q0, q0, q5       \n\t"
        "esp.vmin.s8     q3, q3, q5       \n\t"
        "esp.vmin.s8     q7, q7, q5       \n\t"
        "esp.vst.128.ip  q0, x28, 16      \n\t"
        "esp.vst.128.xp  q3, x28, x26     \n\t"
        "esp.vst.128.ip  q7, x28, 0       \n\t"
        ESP_NN_HWLOOP_END
        :
        : [p] "r"(p), [o] "r"(o), [d16] "r"(d - 16), [step] "r"(step_in - d),
          [skip] "r"(step_out), [n_out] "r"(n_out), [n_in] "r"(n_in)
        : "x26", "x27", "x28", "x29", "x30", "x31", "memory"
    );
}

/*
 * Generic window over all blocks (three or more) of a pixel, three at a time.
 * The last group ends at the last block. It is kept out of line and after
 * the entry point. Otherwise the per-pixel code changes and small cases
 * slow down by ~2%.
 */
static __attribute__((noinline)) void max_pool_window_wide(const int8_t *win, int8_t *out,
                                                           int32_t n_blk, int32_t last_off,
                                                           int32_t n_in, int32_t n_out,
                                                           int32_t step_in, int32_t step_out)
{
    for (int32_t blk = 0; blk < n_blk; blk += 3) {
        if (blk > n_blk - 3) {
            blk = n_blk - 3;
        }
        const int32_t off = blk << 4;
        const int32_t d = blk == n_blk - 3 ? last_off - off : 32;
        max_pool_window_3blk(win + off, out + off, d, n_in, n_out, step_in, step_out);
    }
}

/*
 * Band with stride 1: output byte i of the band is the max over the taps of
 * input byte i at each tap offset, for any channel count. So the band is one
 * run of total bytes in 16-byte blocks, the last one overlapping.
 */
static __attribute__((noinline)) void max_pool_band_blocks(const int8_t *win, int8_t *out,
                                                           int32_t n, int32_t filter_wd,
                                                           int32_t filter_ht, int32_t fh,
                                                           int32_t channels, int32_t row_stride)
{
    if (filter_ht == 1) {
        if (filter_wd == 2) {
            max_pool_2tap_blocks(win, out, n, channels, 16, 16);
        } else {
            max_pool_line_blocks(win, out, n, filter_wd, channels, 16, 16);
        }
    } else if (filter_wd == 2) {
        max_pool_2x2_blocks(win, out, n, channels, row_stride, 16, 16);
    } else if (fh == 3) {
        max_pool_3x3_blocks(win, out, n, channels, row_stride, 16, 16);
    } else {
        max_pool_3x2_blocks(win, out, n, channels, row_stride, 16, 16);
    }
}

static void max_pool_band_flat(const int8_t *win, int8_t *out, int32_t total,
                               int32_t filter_wd, int32_t filter_ht, int32_t fh,
                               int32_t channels, int32_t row_stride)
{
    max_pool_band_blocks(win, out, total >> 4, filter_wd, filter_ht, fh, channels, row_stride);
    if (total & 15) {
        max_pool_band_blocks(win + total - 16, out + total - 16, 1, filter_wd, filter_ht, fh,
                             channels, row_stride);
    }
}
