/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_esp32s3 function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_esp32s3)
 *      - For all other cases:
 *          > Consider `filter_wd * in_ch` as a single row. These many values can
 *              be continuosly loaded from inputs as well.
 *          > multiply accumulate into a single filter output.
 *          > To speed things up further, we pre-calculate
 *              (filter * in_offset + bias term) earlier and add it at the end of filter
 *
 *      About ((filter * in_offset + bias term)) accumulate term:
 *          > The conv operation before requantization is as follows:
 *              for i in filter_size:
 *                  conv_out += (input + input_offset) * filter;
 *               conv_out += bias
 *
 *          > where input_offset is constant term hence, we can see that
 *              this term can be precalculated as:
 *                  for i in filter_size:
 *                      acc_term += input_offset * filter[i];
 *                  acc_term += bias
 *              OR
 *                   for i in filter_size:
 *                      acc_term += filter[i]; // accumulate filter values
 *                  acc_term = acc_term * input_offset + bias
 *
 *
 * In both the above versions we align the filter if needed, pad the input with
 *       -input_offset if needed and extend the channels to make those multiple
 *       of 8/16 as per function needs
 *
 * 3. Im2col version: (for small in_ch where filter_wd * in_ch < 16)
 *      - Inspired by ESP32-P4 im2col approach.
 *      - Instead of padding channels (wastes 81% of SIMD lanes for in_ch=3),
 *        flatten the entire filter window into one contiguous vector:
 *          window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *      - For each output pixel: copy the input window into a scratch buffer,
 *        then use ACCX dot product on the full window. No wasted MACs.
 */

#include <stdio.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_conv_1x1_panel_esp32s3.h"
#include "esp_memory_utils.h"
#include <esp_nn_multicore.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <esp_nn_defs.h>

#include <common_functions.h>

#include "esp_nn_conv_dispatch_esp32s3.h"

/* 3x3 optimized path: im2col per pixel, iterate OC with input in cache */
extern void esp_nn_conv_s8_3x3_opt(const int8_t *input,
    const uint16_t input_wd, const uint16_t input_ht,
    const uint16_t in_channels, const int32_t input_offset,
                             const uint16_t pad_wd,
                             const uint16_t pad_ht,
    const uint16_t stride_wd, const uint16_t stride_ht,
    const int8_t *filter_data, const int32_t *bias,
    int8_t *out_data, const uint16_t out_wd, const uint16_t out_ht,
    const uint16_t out_channels, const int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    const int32_t activation_min, const int32_t activation_max,
    void *scratch);

/* ANSI C reference conv for comparison */
extern void esp_nn_conv_s8_ansi(const data_dims_t *input_dims,
                                const int8_t *input_data,
                                const data_dims_t *filter_dims,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                const data_dims_t *output_dims,
                                int8_t *out_data,
                                const conv_params_t *conv_params,
                                const quant_data_t *quant_data);

/* 1x1 conv: correct SIMD implementation */
extern int esp_nn_conv_s8_1x1_scratch_size(int in_channels);
extern void esp_nn_conv_s8_1x1(const int8_t *input,
                                const uint16_t input_wd,
                                const uint16_t input_ht,
                                const uint16_t in_channels,
                                const int32_t input_offset,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                int8_t *out_data,
                                const uint16_t out_channels,
                                const int32_t out_offset,
                                const int32_t *out_shift,
                                const int32_t *out_mult,
                                const int32_t activation_min,
                                const int32_t activation_max,
                                void *scratch);

/* Debug heap checks - opt-in: a full heap walk per conv call is far too
 * expensive (and too fragile) to ship enabled. */
#ifdef ESP_NN_DEBUG_HEAP_CHECK
#include "esp_heap_caps.h"
#define CONV_HEAP_CHECK(tag) do { \
    if (!heap_caps_check_integrity_all(false)) { \
        printf("CONV HEAP CORRUPT: %s\n", tag); \
    } \
} while(0)
#else
#define CONV_HEAP_CHECK(tag)
#endif

static int16_t *scratch_buffer = NULL;
static uint8_t *preferred_scratch_buffer = NULL;
static size_t preferred_scratch_size = 0;

extern void esp_nn_conv_s8_mult8_1x1_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const int8_t *filter_aligned,
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
                void *buffer /* scratch buffer */);

extern void esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const uint16_t stride_wd,
                const uint16_t stride_ht,
                const int8_t *filter_data,
                const uint16_t filter_wd,
                const uint16_t filter_ht,
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
                                void *scratch_buffer);

/*
 * The asm takes the position count as uint16 width and height. Call it on
 * one row of at most 65528 positions at a time (a multiple of 8), so a
 * count above 65535 is not truncated.
 */
#define ESP_NN_1X1_MAX_ROW_POSITIONS    (65536 - 8)

static void esp_nn_conv_s8_mult8_1x1_rows(
        const int8_t *input, int positions,
        uint16_t in_channels, int32_t input_offset,
        const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, int32_t out_offset,
        const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch)
{
    while (positions > 0) {
        const int n = min(positions, ESP_NN_1X1_MAX_ROW_POSITIONS);
        esp_nn_conv_s8_mult8_1x1_esp32s3(
                input, n, 1, in_channels, input_offset,
                filter_data, bias, out_data, n, 1, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                scratch);
        input += n * in_channels;
        out_data += n * out_channels;
        positions -= n;
    }
}

/*
 * The assembly kernel handles spatial positions in groups of eight, but its
 * scalar remainder path walks the complete filter once per leftover position.
 * For large pointwise filters that makes a 15-position tensor read the weights
 * eight times (one vector group plus seven scalar passes).  Pad the remainder
 * to one vector group instead, so the weights are streamed only once.
 */
static void esp_nn_conv_s8_mult8_1x1_batched_tail(
        const int8_t *input, int spatial_size,
        uint16_t in_channels, int32_t input_offset,
        const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, int32_t out_offset,
        const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch)
{
    const int tail = spatial_size & 7;
    /* The asm needs a 16-byte aligned buffer. The getter reserves the margin. */
    uint8_t *work = (uint8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);

    if (tail < 2) {
        esp_nn_conv_s8_mult8_1x1_rows(
                input, spatial_size, in_channels, input_offset,
                filter_data, bias, out_data, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                work);
        return;
    }

    int8_t *tail_input = (int8_t *)(work + 16 * in_channels);
    int8_t *tail_output = tail_input + 8 * in_channels;
    const int vector_positions = spatial_size - tail;

    esp_nn_conv_s8_mult8_1x1_rows(
            input, vector_positions, in_channels, input_offset,
            filter_data, bias, out_data, out_channels,
            out_offset, out_shift, out_mult, activation_min, activation_max,
            work);

    memcpy(tail_input, input + vector_positions * in_channels,
           tail * in_channels);
    memset(tail_input + tail * in_channels, (int8_t)-input_offset,
           (8 - tail) * in_channels);
    esp_nn_conv_s8_mult8_1x1_esp32s3(
            tail_input, 8, 1, in_channels, input_offset,
            filter_data, bias, tail_output, 8, 1, out_channels,
            out_offset, out_shift, out_mult, activation_min, activation_max,
            work);
    memcpy(out_data + vector_positions * out_channels, tail_output,
           tail * out_channels);
}

/*
 * GEBP-style OC-panel driver for the mult8 1x1 asm. The asm streams the
 * whole filter matrix once per 8-position batch, which thrashes the 64 KB
 * L1 D-cache whenever the filter exceeds it. Output channels are therefore
 * processed in panels sized to half the data cache: a panel stays
 * resident across every position batch, so filter data is fetched from
 * memory once per layer while the batched SIMD kernel does the math.
 * Each panel's output is staged contiguous and scattered into the NHWC
 * layout. Per-output-channel arithmetic is independent and unchanged:
 * results are bit-identical to a single batched call.
 */
void esp_nn_conv_s8_mult8_1x1_oc_panel_ex(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch,
        int work_bytes)
{
    /* Internal RAM is not cached, so panels only pay off for a filter
     * behind the cache (PSRAM or flash). An internal filter runs as one
     * panel when its work area fits: the positions are transposed once,
     * not once per panel. */
    const int one_panel = min(out_channels, 1024);
    int one_panel_bytes = esp_nn_conv_1x1_batched_work_bytes(in_channels, one_panel) + 32;
    if (one_panel != out_channels || out_stride != out_channels) {
        one_panel_bytes += 16 + 8 * one_panel;  /* output staging */
    }
    int oc_tile = one_panel;
    if (!esp_ptr_internal(filter_data) || one_panel_bytes > work_bytes) {
        oc_tile = esp_nn_conv_1x1_panel_tile(in_channels, out_channels);
    }

    uint8_t *work = (uint8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);
    if (oc_tile == out_channels && out_stride == out_channels) {
        /* Whole filter fits one panel: plain batched call, no staging. */
        esp_nn_conv_s8_mult8_1x1_batched_tail(
                input, spatial_size, in_channels, input_offset,
                filter_data, bias, out_data, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max, work);
        return;
    }

    /* Staging area sits after the asm work region (transpose + tails). */
    int8_t *staging = (int8_t *)(work + esp_nn_conv_1x1_batched_work_bytes(in_channels, oc_tile) + 32);
    /* Output staging gets what is left of the work area, at most 8 KB. */
    int staging_bytes = work_bytes - (esp_nn_conv_1x1_batched_work_bytes(in_channels, oc_tile) + 32 + 16);
    if (staging_bytes > 8 * 1024) {
        staging_bytes = 8 * 1024;
    }
    int pos_chunk = staging_bytes / oc_tile;
    pos_chunk &= ~7;
    /* The header's minimum work area leaves room for 32 positions. A smaller
     * area makes pos_chunk 0 and the loop below would never end, so return
     * without output even when asserts are off. */
    assert(pos_chunk >= 8);
    if (pos_chunk < 8) {
        return;
    }

    for (int oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
        const int oc_n = min(oc_tile, out_channels - oc_base);
        for (int pos0 = 0; pos0 < spatial_size; pos0 += pos_chunk) {
            const int pos_n = min(pos_chunk, spatial_size - pos0);
            esp_nn_conv_s8_mult8_1x1_batched_tail(
                    input + pos0 * in_channels, pos_n, in_channels,
                    input_offset, filter_data + oc_base * in_channels,
                    bias ? bias + oc_base : NULL, staging, oc_n,
                    out_offset, out_shift + oc_base, out_mult + oc_base,
                    activation_min, activation_max, work);
            const int8_t *src = staging;
            int8_t *dst = out_data + pos0 * out_stride + oc_base;
            for (int p = 0; p < pos_n; p++) {
                memcpy(dst, src, oc_n);
                src += oc_n;
                dst += out_stride;
            }
        }
    }
}

void esp_nn_conv_s8_mult8_1x1_oc_panel(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch)
{
    esp_nn_conv_s8_mult8_1x1_oc_panel_ex(input, spatial_size, in_channels,
            input_offset, filter_data, bias, out_data, out_channels,
            out_stride, out_offset, out_shift, out_mult, activation_min,
            activation_max, scratch,
            esp_nn_conv_1x1_panel_scratch_size(in_channels));
}

#if ESP_NN_DUAL_CORE_SUPPORTED
typedef struct {
    const int8_t *input;
    int spatial_size;
    int in_channels;
    int32_t input_offset;
    const int8_t *filter_data;
    const int32_t *bias;
    int8_t *out_data;
    int out_channels;
    int out_stride;
    int32_t out_offset;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min;
    int32_t activation_max;
    uint8_t *copy_scratch;
    size_t copy_scratch_size;
} conv_1x1_panel_mt_job_t;

static void conv_1x1_panel_mt_worker(void *p)
{
    const conv_1x1_panel_mt_job_t *j = (const conv_1x1_panel_mt_job_t *)p;
    esp_nn_conv_s8_mult8_1x1_oc_panel(
            j->input, j->spatial_size, j->in_channels, j->input_offset,
            j->filter_data, j->bias, j->out_data, j->out_channels,
            j->out_stride, j->out_offset, j->out_shift, j->out_mult,
            j->activation_min, j->activation_max, j->copy_scratch);
}
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */

/* Use shared dot product from common. See esp_nn_dot_s8_esp32s3.S */

/**
 * Im2col convolution for small in_ch (filter_wd * in_ch < 16).
 *
 * Instead of padding channels to 16 (wasting 81% MACs for in_ch=3),
 * flatten the entire filter window into one contiguous vector.
 *
 * The windows of a chunk of output positions are staged as rows of
 * window_len_aligned bytes. With its windows built, the conv is a 1x1 conv
 * over window_len_aligned channels, so the batched mult8 1x1 asm runs it.
 * That asm does 8 positions per filter pass and requantizes in place, which
 * is much cheaper than one dot and one scalar requant per output.
 * Pad cells hold -input_offset and the filter is zero where no tap is, so
 * neither adds to the sum.
 *
 * Window layout. The tap order inside a window is free, as long as the
 * filter copy uses the same one. Filter row fy (row_bytes < 16) is split:
 * - its first min(row_bytes, slot) bytes at fy * slot,
 * - the rest after all heads, at filter_ht * slot + fy * tail.
 * The slot is chosen so heads can be copied with one vector store:
 * - row_bytes >= 8: slot 8. Heads are 8-byte stores, tails are bytes.
 *   The window stays as short as when packed.
 * - otherwise slot 16 when that costs at most one extra 16-byte block.
 * - otherwise slot = row_bytes: rows packed, copied bytewise.
 *
 * Scratch layout (16-byte aligned):
 * [aligned_filter: out_ch * window_len_aligned]
 * [staging: ESP_NN_IM2COL_POSITIONS * window_len_aligned]
 * [asm work: 24 * window_len_aligned + 8 * out_ch]
 * The sizes are in esp_nn_conv_dispatch_esp32s3.h.
 */
/* Copy 16 (or the low 8) bytes from any address to an aligned one. Reads
 * the two aligned blocks at and after src. */
static inline void esp_nn_copy16_to_aligned(int8_t *dst, const int8_t *src)
{
    asm volatile (
        "ee.ld.128.usar.ip  q0, %[s], 16    \n"
        "ee.ld.128.usar.ip  q1, %[s], 0     \n"
        "ee.src.q           q0, q0, q1      \n"
        "ee.vst.128.ip      q0, %[d], 0     \n"
        : [s] "+r" (src), [d] "+r" (dst)
        :
        : "memory"
    );
}

static inline void esp_nn_copy8_to_aligned(int8_t *dst, const int8_t *src)
{
    asm volatile (
        "ee.ld.128.usar.ip  q0, %[s], 16    \n"
        "ee.ld.128.usar.ip  q1, %[s], 0     \n"
        "ee.src.q           q0, q0, q1      \n"
        "ee.vst.l.64.ip     q0, %[d], 0     \n"
        : [s] "+r" (src), [d] "+r" (dst)
        :
        : "memory"
    );
}

/* Three filter rows of 9 bytes (3x3 filter, 3 channels), the common first
 * layer: heads to dst[0..23], tails to tails[0..2]. One asm block, so the
 * row loop and its pointer updates cost nothing extra. */
static inline void esp_nn_im2col_copy_3x9(int8_t *dst, int8_t *tails,
                                          const int8_t *src, int32_t in_row)
{
    int32_t t;
    const int32_t step = in_row - 16;
    asm volatile (
        "l8ui               %[t], %[s], 8           \n"
        "ee.ld.128.usar.ip  q0, %[s], 16            \n"
        "ee.ld.128.usar.xp  q1, %[s], %[step]       \n"
        "s8i                %[t], %[tl], 0          \n"
        "ee.src.q           q0, q0, q1              \n"
        "l8ui               %[t], %[s], 8           \n"
        "ee.ld.128.usar.ip  q1, %[s], 16            \n"
        "ee.ld.128.usar.xp  q2, %[s], %[step]       \n"
        "s8i                %[t], %[tl], 1          \n"
        "ee.vst.l.64.ip     q0, %[d], 8             \n"
        "ee.src.q           q1, q1, q2              \n"
        "l8ui               %[t], %[s], 8           \n"
        "ee.ld.128.usar.ip  q0, %[s], 16            \n"
        "ee.ld.128.usar.ip  q2, %[s], 0             \n"
        "s8i                %[t], %[tl], 2          \n"
        "ee.vst.l.64.ip     q1, %[d], 8             \n"
        "ee.src.q           q0, q0, q2              \n"
        "ee.vst.l.64.ip     q0, %[d], 0             \n"
        : [s] "+r" (src), [d] "+r" (dst), [t] "=&r" (t)
        : [tl] "r" (tails), [step] "r" (step)
        : "memory"
    );
}

/* Byte copy. The empty asm keeps GCC from turning it into a memcpy call,
 * which costs more than the few bytes copied here. */
static inline void esp_nn_im2col_copy_bytes(int8_t *dst, const int8_t *src,
                                            int32_t len)
{
    for (int32_t i = 0; i < len; i++) {
        dst[i] = src[i];
        asm volatile ("");
    }
}

/* Floor division for a positive divisor (C '/' truncates towards zero). */
static inline int32_t esp_nn_floor_div_s32(int32_t num, int32_t den)
{
    int32_t q = num / den;
    return (num % den != 0 && num < 0) ? q - 1 : q;
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col_s3(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;

    const int32_t row_bytes = filter_wd * in_ch;
    const int32_t slot = esp_nn_im2col_s3_slot(row_bytes, filter_ht);
    const int32_t head = min(row_bytes, slot);
    const int32_t tail = row_bytes - head;
    const int32_t tail_base = filter_ht * slot;
    const int32_t window_len_aligned =
            esp_nn_im2col_s3_window_aligned(row_bytes, filter_ht);
    const int8_t pad_val = (int8_t)(-input_offset);

    int8_t *aligned_filter = (int8_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);
    int8_t *staging = aligned_filter + out_ch * window_len_aligned;
    void *work = staging + ESP_NN_IM2COL_POSITIONS * window_len_aligned;

    /* Filter rows at the staged layout, zero elsewhere. */
    memset(aligned_filter, 0, out_ch * window_len_aligned);
    const int8_t *fptr = filter_data;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        int8_t *dst = aligned_filter + oc * window_len_aligned;
        for (int32_t fy = 0; fy < filter_ht; fy++) {
            memcpy(dst + fy * slot, fptr, head);
            memcpy(dst + tail_base + fy * tail, fptr + head, tail);
            fptr += row_bytes;
        }
    }
    /* Clear the parts of the windows that no copy writes. */
    memset(staging, 0, ESP_NN_IM2COL_POSITIONS * window_len_aligned);

    /* A vector row copy also reads the aligned block after src. It is in
     * the input only while src is below the last block. */
    const int8_t *last_blk = (const int8_t *)(((uintptr_t)(input_data +
            (int32_t)input_wd * input_ht * in_ch - 1)) & ~(uintptr_t)15);

    /* Interior region: all filter taps fall within valid input. */
    int32_t safe_y_start = (pad_ht + stride_ht - 1) / stride_ht;
    /* floor, not C truncation: the numerator is negative when an input side
     * is smaller than the filter, and truncation would mark a row/column as
     * interior whose taps read past the end of the input. */
    int32_t safe_y_end = esp_nn_floor_div_s32(input_ht - filter_ht + pad_ht, stride_ht) + 1;
    int32_t safe_x_start = (pad_wd + stride_wd - 1) / stride_wd;
    int32_t safe_x_end = esp_nn_floor_div_s32(input_wd - filter_wd + pad_wd, stride_wd) + 1;
    /* An interior run must not cross into the next output row. */
    if (safe_y_end > out_ht) safe_y_end = out_ht;
    if (safe_x_end > out_wd) safe_x_end = out_wd;

    const int32_t total = (int32_t)out_wd * out_ht;
    const int32_t in_row = input_wd * in_ch;
    const int32_t x_step = stride_wd * in_ch;
    const int rows_3x9 = (filter_ht == 3 && row_bytes == 9);
    int32_t out_y = 0, out_x = 0;
    for (int32_t pos0 = 0; pos0 < total; pos0 += ESP_NN_IM2COL_POSITIONS) {
        const int32_t n = min(ESP_NN_IM2COL_POSITIONS, total - pos0);
        int8_t *row = staging;
        for (int32_t k = 0; k < n; k++) {
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int32_t base_x = out_x * stride_wd - pad_wd;
            const int is_safe = out_y >= safe_y_start && out_y < safe_y_end &&
                    out_x >= safe_x_start && out_x < safe_x_end;
            /* Only interior pixels read through src; edge pixels build their
             * rows from in-bounds taps below. */
            const int8_t *src = is_safe
                                ? input_data + (base_y * input_wd + base_x) * in_ch
                                : input_data;
            if (is_safe && slot != row_bytes) {
                /* Interior run of this output row inside the chunk: vector
                 * head copies, no bounds tests. It stops before any pixel
                 * whose last row copy would read past the input. */
                int32_t run = min(n - k, safe_x_end - out_x);
                int32_t run_left;
                const int8_t *last_src = src + (filter_ht - 1) * in_row;
                while (run > 0 && last_src + (run - 1) * x_step >= last_blk) {
                    run--;
                }
                if (rows_3x9) {
                    for (int32_t i = 0; i < run; i++) {
                        esp_nn_im2col_copy_3x9(row, row + tail_base, src, in_row);
                        src += x_step;
                        row += window_len_aligned;
                    }
                    run_left = 0;
                } else {
                    run_left = run;
                }
                for (int32_t i = 0; i < run_left; i++) {
                    const int8_t *s = src;
                    int8_t *buf = row;
                    for (int32_t fy = 0; fy < filter_ht; fy++) {
                        if (slot == 16) {
                            esp_nn_copy16_to_aligned(buf, s);
                        } else {
                            esp_nn_copy8_to_aligned(buf, s);
                            esp_nn_im2col_copy_bytes(row + tail_base + fy * tail,
                                                     s + 8, tail);
                        }
                        s += in_row;
                        buf += slot;
                    }
                    src += x_step;
                    row += window_len_aligned;
                }
                if (run > 0) {
                    k += run - 1;
                    out_x += run;
                    if (out_x == out_wd) {
                        out_x = 0;
                        out_y++;
                    }
                    continue;
                }
            }
            for (int32_t fy = 0; fy < filter_ht; fy++) {
                const int8_t *row_src = src + fy * in_row;
                int8_t tmp[16];
                if (!is_safe) {
                    /* Edge pixel: build the row with pad cells. */
                    const int32_t in_y = base_y + fy;
                    for (int32_t fx = 0; fx < filter_wd; fx++) {
                        const int32_t in_x = base_x + fx;
                        if (in_y >= 0 && in_y < input_ht &&
                                in_x >= 0 && in_x < input_wd) {
                            memcpy(tmp + fx * in_ch,
                                   input_data + (in_y * input_wd + in_x) * in_ch,
                                   in_ch);
                        } else {
                            memset(tmp + fx * in_ch, pad_val, in_ch);
                        }
                    }
                    row_src = tmp;
                }
                esp_nn_im2col_copy_bytes(row + fy * slot, row_src, head);
                esp_nn_im2col_copy_bytes(row + tail_base + fy * tail,
                                         row_src + head, tail);
            }
            row += window_len_aligned;
            if (++out_x == out_wd) {
                out_x = 0;
                out_y++;
            }
        }
        esp_nn_conv_s8_mult8_1x1_batched_tail(
                staging, n, (uint16_t)window_len_aligned, input_offset,
                aligned_filter, bias, out_data + pos0 * out_ch, out_ch,
                conv_params->out_offset, quant_data->shift, quant_data->mult,
                conv_params->activation.min, conv_params->activation.max,
                work);
    }
}

void esp_nn_set_conv_scratch_buf_esp32s3(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

/* The preferred buffer stays set. The 1x1 kernel stages its panels there
 * when it fits, so a fully connected buffer in PSRAM does not slow it. */
void esp_nn_conv_scratch_borrow_esp32s3(void *buf, void **scratch)
{
    *scratch = scratch_buffer;
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_restore_esp32s3(void *scratch)
{
    scratch_buffer = (int16_t *) scratch;
}

void esp_nn_set_conv_preferred_scratch_buf_esp32s3(void *buf, size_t size)
{
    preferred_scratch_buffer = (uint8_t *)buf;
    preferred_scratch_size = size;
}

void esp_nn_conv_s8_esp32s3(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* The fast paths fill pad cells with (int8_t)-input_offset.
     * For input_offset -128 that value is 128, which does not fit int8.
     * Use the reference for it. */
    if (input_offset < -127) {
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    /* Grouped conv (filter_ch < input_ch): run each group through the
     * optimized path via repacked slices; reference row-split otherwise. */
    if (channels != filter_dims->channels) {
        data_dims_t in_g = *input_dims;
        data_dims_t out_g = *output_dims;
        const int32_t groups = filter_dims->channels
                ? channels / filter_dims->channels : 0;
        if (groups > 1 && output_dims->channels % groups == 0 &&
                channels % filter_dims->channels == 0) {
            in_g.channels = filter_dims->channels;
            out_g.channels = output_dims->channels / groups;
            const int inner = esp_nn_conv_scratch_size_esp32s3_impl(
                    &in_g, filter_dims, &out_g, conv_params);
            const int total = esp_nn_conv_scratch_size_esp32s3_impl(
                    input_dims, filter_dims, output_dims, conv_params);
            if (esp_nn_conv_s8_grouped_repack(
                    esp_nn_conv_s8_esp32s3, input_dims, input,
                    filter_dims, filter_data, bias, output_dims, out_data,
                    conv_params, quant_data, scratch_buffer, total, inner)) {
                return;
            }
        }
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    int filter_size = filter_wd * filter_ht * channels * out_channels;

    /* 1x1 stride-1 conv */
    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {
        /* The mult8 asm loads the input and stores the output 8 bytes at a
         * time, so both must be 8-byte aligned. */
        if (channels % 8 == 0 &&
                (((uintptr_t)input | (uintptr_t)out_data) & 7) == 0) {
            int spatial_size = input_wd * input_ht;
            int filter_bytes = channels * out_channels;
            if (filter_bytes > 24 * 1024) {
                /* Panel work + staging (24 * in_ch + 16 KB + 64) goes to the
                 * preferred (internal) scratch when it fits, like the batched
                 * path below; otherwise a PSRAM arena would host the very
                 * buffers the L1-resident scheme depends on. */
                void *panel_scratch = scratch_buffer;
                if (preferred_scratch_buffer != NULL &&
                        (((uintptr_t)preferred_scratch_buffer & 15) == 0) &&
                        (size_t)esp_nn_conv_1x1_panel_scratch_size(channels) <= preferred_scratch_size) {
                    panel_scratch = preferred_scratch_buffer;
                }
                /* Filter exceeds the L1 panel budget: the OC-panel driver keeps
                 * each panel of filter rows hot across all position batches. */
                /* Cores split the output channels; outputs are disjoint,
                 * arithmetic unchanged. */
#if ESP_NN_DUAL_CORE_SUPPORTED
                if (out_channels >= 32 && esp_nn_dual_core_begin()) {
                    const int oc0 = ((out_channels / 2) + 7) & ~7;
                    const int wneed = esp_nn_conv_1x1_panel_scratch_size(channels);
                    uint8_t *wscr = (uint8_t *)esp_nn_dual_core_scratch(wneed);
                    if (wscr != NULL && oc0 > 0 && oc0 < out_channels) {
                        conv_1x1_panel_mt_job_t job = {
                            .input = input, .spatial_size = spatial_size,
                            .in_channels = channels, .input_offset = input_offset,
                            .filter_data = filter_data, .bias = bias,
                            .out_data = out_data, .out_channels = oc0,
                            .out_stride = out_channels,
                            .out_offset = out_offset, .out_shift = out_shift,
                            .out_mult = out_mult, .activation_min = activation_min,
                            .activation_max = activation_max,
                            .copy_scratch = wscr, .copy_scratch_size = (size_t)wneed,
                        };
                        if (esp_nn_dual_core_run(conv_1x1_panel_mt_worker, &job)) {
                            esp_nn_conv_s8_mult8_1x1_oc_panel(
                                    input, spatial_size, channels, input_offset,
                                    filter_data + oc0 * channels,
                                    bias ? bias + oc0 : NULL,
                                    out_data + oc0, out_channels - oc0, out_channels,
                                    out_offset, out_shift + oc0, out_mult + oc0,
                                    activation_min, activation_max, panel_scratch);
                            esp_nn_dual_core_wait();
                            esp_nn_dual_core_end();
                            return;
                        }
                        /* worker declined (nested or same-core call): single-core below */
                    }
                    esp_nn_dual_core_end();
                }
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
                esp_nn_conv_s8_mult8_1x1_oc_panel(
                        input, spatial_size, channels, input_offset,
                        filter_data, bias, out_data, out_channels,
                        out_channels, out_offset, out_shift, out_mult,
                        activation_min, activation_max, panel_scratch);
                return;
            }
            void *pointwise_scratch = scratch_buffer;
            int required = esp_nn_conv_1x1_batched_work_bytes(channels, out_channels) + 16;
            if (preferred_scratch_buffer != NULL &&
                    (((uintptr_t)preferred_scratch_buffer & 15) == 0) &&
                    required <= preferred_scratch_size) {
                pointwise_scratch = preferred_scratch_buffer;
            }
            /* Full asm path: mult8 channels, any filter alignment */
            esp_nn_conv_s8_mult8_1x1_batched_tail(
                    input, spatial_size, channels, input_offset,
                    filter_data, bias, out_data, out_channels, out_offset,
                    out_shift, out_mult, activation_min, activation_max,
                    pointwise_scratch);
        } else {
            /* Fallback: handles any alignment + any channel count */
            esp_nn_conv_s8_1x1(input, input_wd, input_ht, channels, input_offset,
                               filter_data, bias, out_data, out_channels, out_offset,
                               out_shift, out_mult, activation_min, activation_max,
                               scratch_buffer);
        }
        return;
    }

    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }

    {
        int32_t filter_row_size = filter_wd * channels;
        int32_t window_len = filter_wd * filter_ht * channels;

        /* 3x3 optimized path: staged windows through the 1x1 asm, an OC-outer
         * kernel for small maps, or im2col per pixel with the input in cache.
         * Each avoids the general asm's per-output-channel input reload. */
        if (esp_nn_conv_s8_3x3_can_use(filter_wd, filter_ht, channels, out_channels)) {
            esp_nn_conv_s8_3x3_opt(input, input_wd, input_ht, channels,
                                    input_offset, pad_wd, pad_ht, stride_wd, stride_ht,
                                    filter_data, bias, out_data,
                                    out_wd, out_ht, out_channels, out_offset,
                                    out_shift, out_mult, activation_min, activation_max,
                                    (void *)scratch_buffer);
            return;
        }


        /* Im2col path: small in_ch where per-row SIMD is wasteful,
         * but entire window is large enough for SIMD dot product.
         * E.g., 3x3 conv with in_ch=3: row=9 (<16), window=27 (>=16). */
        if (filter_row_size < 16 && window_len >= 16) {
            esp_nn_conv_s8_im2col_s3(input_dims, input, filter_dims, filter_data,
                                      bias, output_dims, out_data, conv_params,
                                      quant_data, scratch_buffer);
            return;
        }

        // align the `filter width * channels` to 16 bytes. Do zero padding for the same
        int32_t filter_alignment_padding = 16 - (filter_row_size & 15);
        int8_t *filter_data_aligned = (int8_t *) filter_data;
        int8_t *input_padded = (int8_t *) input;
        /* The asm reads the filter copy and the padded input with
         * ee.vld.128, which drops the low 4 address bits. Align the copies
         * to 16. The getter reserves 16 bytes for this. */
        int8_t *scratch_data = (int8_t *)(((uintptr_t) scratch_buffer + 15) & ~(uintptr_t) 15);
        int new_input_wd = input_wd, new_input_ht = input_ht;
        if (filter_alignment_padding != 16) {
            // pad filter_data
            int32_t new_row_size = filter_wd * channels + filter_alignment_padding;
            filter_data_aligned = scratch_data;
            int8_t *row_ptr = filter_data_aligned;
            const int8_t *filter_data_ptr = filter_data;
            for (int32_t ch_idx = 0; ch_idx < out_channels; ch_idx++) {
                for (int32_t row_idx = 0; row_idx < filter_ht; row_idx++) {
                    memcpy(row_ptr, filter_data_ptr, filter_row_size);
                    memset(row_ptr + filter_row_size, 0, new_row_size - filter_row_size);
                    filter_data_ptr += filter_row_size;
                    row_ptr += new_row_size;
                }
            }
            scratch_data += new_row_size * filter_ht * out_channels;
            filter_row_size = new_row_size;
        } else if ((int) filter_data & 15) {
            filter_data_aligned = scratch_data;
            memcpy(filter_data_aligned, filter_data, filter_size);
            scratch_data += filter_size;
        }
        // pad_wd/pad_ht carry only the leading (top/left) padding as passed by
        // TFLite. Derive the trailing (bottom/right) padding from the output
        // extent: it differs from the leading one whenever the total "SAME"
        // padding is odd, and is also needed when pad_wd/pad_ht are 0.
        int32_t pad_right = max(0, (out_wd - 1) * stride_wd + filter_wd - pad_wd - input_wd);
        int32_t pad_bottom = max(0, (out_ht - 1) * stride_ht + filter_ht - pad_ht - input_ht);

        if (pad_wd != 0 || pad_ht != 0 || pad_right > 0 || pad_bottom > 0) {
            input_padded = (int8_t *) scratch_data;
            esp_nn_aligned_s8_pad_asymmetric(input, input_padded, input_wd, input_ht, channels,
                                             -input_offset, pad_wd, pad_ht,
                                             (uint16_t) pad_right, (uint16_t) pad_bottom);
            new_input_wd = input_wd + pad_wd + pad_right;
            new_input_ht = input_ht + pad_ht + pad_bottom;
            scratch_data += new_input_wd * new_input_ht * channels;
        }


        int filter_total = filter_wd * filter_ht * channels * out_channels;
        if (input_offset != 0 && filter_total > 16384) {
            int32_t *corrections = (int32_t *)scratch_data;
            int32_t filter_ch_size = filter_wd * filter_ht * channels;
            const int8_t *f_src = filter_data; // use ORIGINAL (not aligned) filter for sum
            for (int ch = 0; ch < out_channels; ch++) {
                /* esp-nn#36: this was a scalar reduction the same algorithmic
                 * size as the conv itself, recomputed every call. */
                int32_t filter_sum = esp_nn_filter_sum_s8_esp32s3(f_src, filter_ch_size);
                corrections[ch] = filter_sum * input_offset;
                if (bias) {
                    corrections[ch] += bias[ch];
                }
                f_src += filter_ch_size;
            }
            // Pass input_offset=0 to assembly so it skips its pre-computation.
            // Pass scratch_data as "bias" pointer. The assembly's bias-copy loop
            // will read from scratch and write to scratch (identity, no-op).
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, 0,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                (const int32_t *)scratch_data, out_data, out_wd, out_ht, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                scratch_data);
            CONV_HEAP_CHECK("general: after asm (precomp)");
        } else {
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, input_offset,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                bias, out_data, out_wd, out_ht, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max, scratch_data);
            CONV_HEAP_CHECK("general: after asm (normal)");
        }
    }
}
