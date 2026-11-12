/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Optimized 3x3 convolution for ESP32-S3.
 *
 * Key optimization vs the general aligned asm:
 * The general asm reloads input for each output channel (128× per pixel).
 * This version pre-loads the 3x3 input window into scratch (9 rows × in_ch bytes),
 * then iterates output channels with the input in L1 cache.
 *
 * For Conv[11] (26×26×128→12×12×128, 3×3 s2):
 * - Input window: 3 × 3 × 128 = 1,152 bytes (fits in L1)
 * - Filter per OC: 3 × 3 × 128 = 1,152 bytes
 * - Total for all 128 OC: 147,456 bytes (cycles through L1)
 * - Input loaded once vs 128× in the general asm
 */

#include <stdint.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_conv_1x1_panel_esp32s3.h"
#include <esp_nn_multicore.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <common_functions.h>

#include "esp_nn_conv_dispatch_esp32s3.h"

/*
 * 3x3 convolution: im2col per pixel, then dot product per output channel.
 * Uses ACCX dot product (ee.vmulas.s8.accx) for the 3×3×in_ch window.
 */
/* Build one output pixel's 3x3 window into `dst`. Out-of-bounds cells hold
 * -input_offset so that (cell + offset) * w is zero, which is what the
 * per-channel correction term already assumes. */
static inline void esp_nn_3x3_build_window(
        int8_t *dst, const int8_t *input, int in_y, int in_x,
        int input_wd, int input_ht, int in_channels, int in_row_stride,
        int8_t pad_val)
{
    for (int fy = 0; fy < 3; fy++) {
        const int y = in_y + fy;
        if (y < 0 || y >= input_ht) {
            memset(dst, pad_val, 3 * in_channels);
        } else if (in_x >= 0 && in_x + 3 <= input_wd) {
            memcpy(dst, input + y * in_row_stride + in_x * in_channels,
                   3 * in_channels);
        } else {
            for (int fx = 0; fx < 3; fx++) {
                const int x = in_x + fx;
                if (x < 0 || x >= input_wd) {
                    memset(dst + fx * in_channels, pad_val, in_channels);
                } else {
                    memcpy(dst + fx * in_channels,
                           input + y * in_row_stride + x * in_channels,
                           in_channels);
                }
            }
        }
        dst += 3 * in_channels;
    }
}

/* One filter row of window_len bytes, zero-padded to window_len_aligned:
 * the asm reads whole 16-byte blocks, and the padding multiplies zeros. */
static inline void esp_nn_3x3_copy_filter_row(int8_t *dst, const int8_t *src,
                                              int window_len, int window_len_aligned)
{
    memcpy(dst, src, window_len);
    memset(dst + window_len, 0, window_len_aligned - window_len);
}

/* Run output positions [pos_begin, pos_end) of the staged path: stage the
 * windows for a chunk of positions, then hand them to the 1x1 OC-panel
 * driver, for which a 3x3 conv with its windows built is a 1x1 conv over
 * 9 * in_channels (and the filter layout [oc][3][3][ic] already matches).
 * The filter is then streamed once per chunk instead of once per pixel.
 * Positions are contiguous in NHWC output, so a range is a disjoint slice. */
static void esp_nn_3x3_staged_range(
        const int8_t *input, int input_wd, int input_ht, int in_channels,
        int32_t input_offset, int pad_wd, int pad_ht, int stride_wd,
        int stride_ht, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, int out_wd, int out_channels, int32_t out_offset,
        const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max,
        int pos_begin, int pos_end, void *scratch)
{
    const int window_len_aligned = ((9 * in_channels) + 15) & ~15;
    const int in_row_stride = input_wd * in_channels;
    const int8_t pad_val = (int8_t)(-input_offset);
    /* batch and the panel size are recomputed per chunk (cheap) so they are
     * not live across the window build loop. */
    int8_t *staging = (int8_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);

    for (int pos0 = pos_begin; pos0 < pos_end;) {
        const int n = min(esp_nn_3x3_batch_positions(in_channels, out_channels),
                          pos_end - pos0);
        for (int k = 0; k < n; k++) {
            const int pos = pos0 + k;
            const int out_y = pos / out_wd;
            const int out_x = pos - out_y * out_wd;
            esp_nn_3x3_build_window(staging + k * window_len_aligned, input,
                                    out_y * stride_ht - pad_ht,
                                    out_x * stride_wd - pad_wd,
                                    input_wd, input_ht, in_channels,
                                    in_row_stride, pad_val);
        }
        const int batch = esp_nn_3x3_batch_positions(in_channels, out_channels);
        esp_nn_conv_s8_mult8_1x1_oc_panel_ex(
                staging, n, (uint16_t)window_len_aligned, input_offset,
                filter_data, bias, out_data + (int32_t)pos0 * out_channels,
                (uint16_t)out_channels, (uint16_t)out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max,
                staging + batch * window_len_aligned,
                esp_nn_3x3_panel_work_bytes(window_len_aligned, in_channels, out_channels));
        pos0 += n;
    }
}

#if ESP_NN_DUAL_CORE_SUPPORTED
typedef struct {
    const int8_t *input;
    int input_wd, input_ht, in_channels;
    int32_t input_offset;
    int pad_wd, pad_ht, stride_wd, stride_ht;
    const int8_t *filter_data;
    const int32_t *bias;
    int8_t *out_data;
    int out_wd, out_channels;
    int32_t out_offset;
    const int32_t *out_shift, *out_mult;
    int32_t activation_min, activation_max;
    int pos_begin, pos_end;
    void *scratch;
} conv3x3_staged_mt_job_t;

static void conv3x3_staged_mt_worker(void *p)
{
    const conv3x3_staged_mt_job_t *j = (const conv3x3_staged_mt_job_t *)p;
    esp_nn_3x3_staged_range(j->input, j->input_wd, j->input_ht, j->in_channels,
                            j->input_offset, j->pad_wd, j->pad_ht, j->stride_wd,
                            j->stride_ht, j->filter_data, j->bias, j->out_data,
                            j->out_wd, j->out_channels, j->out_offset,
                            j->out_shift, j->out_mult, j->activation_min,
                            j->activation_max, j->pos_begin, j->pos_end,
                            j->scratch);
}
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */

void esp_nn_conv_s8_3x3_opt(const int8_t *input,
                             const uint16_t input_wd,
                             const uint16_t input_ht,
                             const uint16_t in_channels,
                             const int32_t input_offset,
                             const uint16_t pad_wd,
                             const uint16_t pad_ht,
                             const uint16_t stride_wd,
                             const uint16_t stride_ht,
                             const int8_t *filter_data,
                             const int32_t *bias,
                             int8_t *out_data,
                             const uint16_t out_wd,
                             const uint16_t out_ht,
                             const uint16_t out_channels,
                             const int32_t out_offset,
                             const int32_t *out_shift,
                             const int32_t *out_mult,
                             const int32_t activation_min,
                             const int32_t activation_max,
                             void *scratch)
{
    const int window_len = 9 * in_channels; /* 3×3 window */
    const int window_len_aligned = (window_len + 15) & ~15;

    if (esp_nn_conv_s8_3x3_small_map_can_use(in_channels, out_channels, out_wd, out_ht,
                                stride_wd, stride_ht)) {
        esp_nn_conv_s8_3x3_small_map(input, input_wd, input_ht, in_channels,
                input_offset, pad_wd, pad_ht, stride_wd, stride_ht,
                filter_data, bias, out_data, out_wd, out_ht, out_channels,
                out_offset, out_shift, out_mult, activation_min,
                activation_max, scratch);
        return;
    }

    if (esp_nn_conv_s8_3x3_use_batch(out_wd, out_ht)) {
        const int total_positions = (int)out_wd * out_ht;
        if (window_len_aligned != window_len) {
            /* The asm reads filter rows of window_len_aligned bytes: copy the
             * filter with zero-padded rows. The staged windows' tails then
             * multiply zeros. */
            int8_t *fcopy = (int8_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);
            for (int oc = 0; oc < out_channels; oc++) {
                esp_nn_3x3_copy_filter_row(fcopy + oc * window_len_aligned,
                                           filter_data + oc * window_len,
                                           window_len, window_len_aligned);
            }
            filter_data = fcopy;
            scratch = fcopy + out_channels * window_len_aligned;
        }
#if ESP_NN_DUAL_CORE_SUPPORTED
        /* Split the position range across cores: disjoint output slices, same
         * arithmetic. The worker stages into its own scratch. */
        const int batch = esp_nn_3x3_batch_positions(in_channels, out_channels);
        if (total_positions >= 2 * batch && esp_nn_dual_core_begin()) {
            const int wneed = batch * window_len_aligned + 32
                    + esp_nn_3x3_panel_work_bytes(window_len_aligned, in_channels, out_channels);
            void *wscr = esp_nn_dual_core_scratch(wneed);
            const int split = ((total_positions / 2) / batch) * batch;
            if (wscr != NULL && split > 0 && split < total_positions) {
                conv3x3_staged_mt_job_t job = {
                    .input = input, .input_wd = input_wd, .input_ht = input_ht,
                    .in_channels = in_channels, .input_offset = input_offset,
                    .pad_wd = pad_wd, .pad_ht = pad_ht,
                    .stride_wd = stride_wd, .stride_ht = stride_ht,
                    .filter_data = filter_data, .bias = bias,
                    .out_data = out_data, .out_wd = out_wd,
                    .out_channels = out_channels, .out_offset = out_offset,
                    .out_shift = out_shift, .out_mult = out_mult,
                    .activation_min = activation_min,
                    .activation_max = activation_max,
                    .pos_begin = 0, .pos_end = split, .scratch = wscr,
                };
                if (esp_nn_dual_core_run(conv3x3_staged_mt_worker, &job)) {
                    esp_nn_3x3_staged_range(input, input_wd, input_ht,
                            in_channels, input_offset, pad_wd, pad_ht,
                            stride_wd, stride_ht, filter_data, bias, out_data,
                            out_wd, out_channels, out_offset, out_shift,
                            out_mult, activation_min, activation_max,
                            split, total_positions, scratch);
                    esp_nn_dual_core_wait();
                    esp_nn_dual_core_end();
                    return;
                }
                /* worker declined (nested or same-core call): fall through */
            }
            esp_nn_dual_core_end();
        }
#endif
        esp_nn_3x3_staged_range(input, input_wd, input_ht, in_channels,
                                input_offset, pad_wd, pad_ht, stride_wd,
                                stride_ht, filter_data, bias, out_data, out_wd,
                                out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max,
                                0, total_positions, scratch);
        return;
    }

    /* Scratch layout: [im2col_buf | filter_aligned | corrections] */
    int8_t *im2col_buf = (int8_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);
    int8_t *filter_aligned = im2col_buf + window_len_aligned;
    int32_t *corrections = (int32_t *)(filter_aligned
                                       + out_channels * window_len_aligned);

    /* The inner loop is the plain aligned dot - no unaligned SIMD, no
     * priming, no reads outside either operand. When window_len is a
     * multiple of 16 (in_ch % 16 == 0) and the filter pointer is 16-byte
     * aligned (TFLM arena/flash weights are), the filter is used in place.
     * Otherwise it is copied into zero-padded aligned rows. */
    const int copy_filter = (((uintptr_t)filter_data & 15) != 0)
                            || (window_len_aligned != window_len);
    const int8_t *filter_base = copy_filter ? filter_aligned : filter_data;
    const int8_t *f_ptr = filter_data;
    for (int oc = 0; oc < out_channels; oc++) {
        if (copy_filter) {
            esp_nn_3x3_copy_filter_row(filter_aligned + oc * window_len_aligned,
                                       f_ptr, window_len, window_len_aligned);
        }
        int32_t corr = bias ? bias[oc] : 0;
        if (input_offset != 0) {
            corr += esp_nn_filter_sum_s8_esp32s3(f_ptr, window_len)
                    * input_offset;
        }
        corrections[oc] = corr;
        f_ptr += window_len;
    }

    /* Zero-pad the tail of im2col buffer for aligned SIMD reads */
    memset(im2col_buf + window_len, 0, window_len_aligned - window_len);

    const int in_row_stride = input_wd * in_channels;

    /* Padding value: a padded position must contribute (pad_q + offset) * w
     * = 0, and the corrections term already adds offset * w for EVERY filter
     * position - so the padded cell holds -offset, cancelling it. Covers
     * explicit padding and TFLite's implicit trailing pad through the same
     * bounds checks. */
    const int8_t pad_val = (int8_t)(-input_offset);

    for (int out_y = 0; out_y < out_ht; out_y++) {
        for (int out_x = 0; out_x < out_wd; out_x++) {
            /* Phase 1: build this pixel's window */
            esp_nn_3x3_build_window(im2col_buf, input,
                                    out_y * stride_ht - pad_ht,
                                    out_x * stride_wd - pad_wd,
                                    input_wd, input_ht, in_channels,
                                    in_row_stride, pad_val);

            /* Phase 2: dot against each output channel's aligned filter copy */
            for (int oc = 0; oc < out_channels; oc++) {
                int32_t acc = esp_nn_dot_s8_aligned_esp32s3(
                        im2col_buf, filter_base + oc * window_len_aligned,
                        window_len_aligned);
                acc += corrections[oc];
                acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[oc], out_shift[oc]);
                acc += out_offset;
                acc = max(acc, activation_min);
                acc = min(acc, activation_max);
                *out_data++ = (int8_t)acc;
            }
        }
    }
}
