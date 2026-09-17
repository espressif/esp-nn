/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * FC multi-path dispatcher for ESP32-S3.
 * - Pre-computes offset corrections per channel in C
 * - Dispatches to s8 MAC assembly (aligned, large row_len) or s16 assembly (fallback)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <common_functions.h>
#include <esp_nn_ansi_headers.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"

/* Original s16 assembly (renamed) */
extern void esp_nn_fc_s16_esp32s3(const int8_t *input_data,
                                   const int32_t input_offset,
                                   const uint16_t row_len,
                                   const int8_t *filter_data,
                                   const int32_t filter_offset,
                                   const int32_t *bias,
                                   int8_t *out_data,
                                   const uint16_t out_channels,
                                   const int32_t out_offset,
                                   const int32_t out_shift,
                                   const int32_t out_mult,
                                   const int32_t activation_min,
                                   const int32_t activation_max);

extern void esp_nn_fc_per_ch_s16_esp32s3(const int8_t *input_data,
                                          const int32_t input_offset,
                                          const uint16_t row_len,
                                          const int8_t *filter_data,
                                          const int32_t filter_offset,
                                          const int32_t *bias,
                                          int8_t *out_data,
                                          const uint16_t out_channels,
                                          const int32_t out_offset,
                                          const int32_t *out_shift,
                                          const int32_t *out_mult,
                                          const int32_t activation_min,
                                          const int32_t activation_max);

/* Shared s8 dot product from common — `a` must be 16-byte aligned, `b` may be
 * unaligned (handled via USAR+QUP). The product is symmetric, so whichever of
 * input/filter happens to be aligned can be passed as `a`.
 *
 * Both operands are read past their logical end, by the usual esp-nn amount:
 * the primed 2x-unrolled loop issues one 128-bit block more than it consumes.
 * Per call, with n = len_div16:
 *   a (aligned):   16 bytes over for even n, 0 for odd n
 *   b (unaligned): 17..31 bytes over for even n, 1..15 for odd n
 * Reads only — nothing is written outside out_data.
 *
 * Swapping the operands therefore moves the larger over-read from the filter
 * onto the input. It does not introduce one: as `a`, a 16-byte aligned input was
 * already over-read by 16 bytes on even n, once per output channel, and that is
 * the case for every model the aligned fast path has ever served. The swap takes
 * the input from <=16 to <=31 bytes and drops the filter from <=31 to <=16.
 * These are aligned loads from mapped SRAM/PSRAM and do not fault; the only
 * theoretical corner is a tensor ending within 32 bytes of the end of a mapped
 * region. */
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a,
                                                const int8_t *b,
                                                int32_t len_div16);

/* The s16 assembly loads the input with 8-byte vector loads and only derives
 * SAR_BYTE from the filter pointer, so it silently requires an 8-byte aligned
 * input. Anything less has to go to the ansi reference. */
#define FC_S16_INPUT_ALIGN  8

/* When the dot path beats the fused s16 assembly. It has ~2x the assembly's
 * throughput but pays a per-channel filter-sum the assembly folds into its MAC
 * for free, plus a scalar tail once more per pass for a row_len that is not a
 * whole number of vectors. Two regimes, both measured on S3 (row_len 16..1024,
 * out_ch 1..256; out_ch never shifts the boundary):
 *   input_offset != 0: correction pass runs, tail paid twice -> 192 + tail*16
 *   input_offset == 0: no correction pass, tail paid once    ->  64 + tail*8
 * The io==0 constants being exactly half-ish of the io!=0 ones matches the
 * model: one filter pass and one tail instead of two of each.
 *
 * These constants are empirical, so they can drift with cache geometry: the
 * two-pass case doubles the traffic once the filter outgrows dcache. Both paths
 * are bit-exact, so a mis-tuned boundary costs a few percent, never
 * correctness. Fusing the sum into the MAC pass would remove the second pass
 * and the boundary with it. */
static inline bool fc_dot_path_wins(uint16_t row_len, int32_t input_offset)
{
    const int tail = row_len & 15;
    if (input_offset == 0) {
        return row_len >= 64 + tail * 8;
    }
    return row_len >= 192 + tail * 16;
}

void esp_nn_fully_connected_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t out_shift,
                                       const int32_t out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max)
{
    /* The s8 fast path needs one of the two operands 16-byte aligned. Filter
     * rows are aligned only if the base is aligned and every row is a whole
     * number of vectors. */
    const bool input_aligned = ((uintptr_t)input_data & 15) == 0;
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);

    if (__builtin_expect(filter_offset != 0 || !fc_dot_path_wins(row_len, input_offset)
        || (!input_aligned && !filter_rows_aligned), 0)) {
        if ((uintptr_t)input_data & (FC_S16_INPUT_ALIGN - 1)) {
            esp_nn_fully_connected_s8_ansi(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult,
                                           activation_min, activation_max);
            return;
        }
        /* Fallback to original s16 assembly — tail call, no extra overhead */
        esp_nn_fc_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                              filter_offset, bias, out_data, out_channels,
                              out_offset, out_shift, out_mult,
                              activation_min, activation_max);
        return;
    }
    {
        int32_t row_len_div16 = row_len >> 4;

        int32_t row_len_rem = row_len & 15;
        int32_t simd_bytes = row_len_div16 << 4;

        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            /* Per-channel correction, inline (no out_channels-sized VLA) */
            int32_t corr = 0;
            if (input_offset != 0) {
                corr = esp_nn_filter_sum_s8_esp32s3(f_ptr, row_len) * input_offset;
            }
            if (bias) {
                corr += bias[ch];
            }
            /* Pass the aligned operand first; the dot product is symmetric. */
            int32_t acc = input_aligned
                ? esp_nn_dot_s8_unaligned_esp32s3(input_data, f_ptr, row_len_div16)
                : esp_nn_dot_s8_unaligned_esp32s3(f_ptr, input_data, row_len_div16);

            /* Scalar remainder for non-multiple-of-16 row_len */
            for (int i = 0; i < row_len_rem; i++) {
                acc += (int32_t)input_data[simd_bytes + i] * (int32_t)f_ptr[simd_bytes + i];
            }

            acc += corr;

            acc = esp_nn_multiply_by_quantized_mult(acc, out_mult, out_shift);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[ch] = (int8_t)acc;
        }
    }
}

void esp_nn_fully_connected_per_ch_s8_esp32s3(const int8_t *input_data,
                                       const int32_t input_offset,
                                       const uint16_t row_len,
                                       const int8_t *filter_data,
                                       const int32_t filter_offset,
                                       const int32_t *bias,
                                       int8_t *out_data,
                                       const uint16_t out_channels,
                                       const int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       const int32_t activation_min,
                                       const int32_t activation_max)
{
    const bool input_aligned = ((uintptr_t)input_data & 15) == 0;
    const bool filter_rows_aligned = (((uintptr_t)filter_data & 15) == 0)
                                     && ((row_len & 15) == 0);

    if (__builtin_expect(filter_offset != 0 || !fc_dot_path_wins(row_len, input_offset)
        || (!input_aligned && !filter_rows_aligned), 0)) {
        if ((uintptr_t)input_data & (FC_S16_INPUT_ALIGN - 1)) {
            esp_nn_fully_connected_per_ch_s8_ansi(input_data, input_offset, row_len,
                                                  filter_data, filter_offset, bias,
                                                  out_data, out_channels, out_offset,
                                                  out_shift, out_mult,
                                                  activation_min, activation_max);
            return;
        }
        esp_nn_fc_per_ch_s16_esp32s3(input_data, input_offset, row_len, filter_data,
                                     filter_offset, bias, out_data, out_channels,
                                     out_offset, out_shift, out_mult,
                                     activation_min, activation_max);
        return;
    }
    {
        int32_t row_len_div16 = row_len >> 4;

        int32_t row_len_rem = row_len & 15;
        int32_t simd_bytes = row_len_div16 << 4;

        for (int ch = 0; ch < out_channels; ch++) {
            const int8_t *f_ptr = filter_data + ch * row_len;
            /* Per-channel correction, inline (no out_channels-sized VLA) */
            int32_t corr = 0;
            if (input_offset != 0) {
                corr = esp_nn_filter_sum_s8_esp32s3(f_ptr, row_len) * input_offset;
            }
            if (bias) {
                corr += bias[ch];
            }
            /* Pass the aligned operand first; the dot product is symmetric. */
            int32_t acc = input_aligned
                ? esp_nn_dot_s8_unaligned_esp32s3(input_data, f_ptr, row_len_div16)
                : esp_nn_dot_s8_unaligned_esp32s3(f_ptr, input_data, row_len_div16);

            for (int i = 0; i < row_len_rem; i++) {
                acc += (int32_t)input_data[simd_bytes + i] * (int32_t)f_ptr[simd_bytes + i];
            }

            acc += corr;

            acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[ch], out_shift[ch]);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[ch] = (int8_t)acc;
        }
    }
}
