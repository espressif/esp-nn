/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * One-pass s8 FC for ESP32-S3: filter_offset == 0, a 16-byte aligned input
 * and row_len >= 16. The caller checks all three. input_offset may be any
 * value in [-256, 254].
 *
 * The dot path walks each filter row twice when input_offset != 0: once for
 * sum(filter) and once for the dot. A filter larger than the data cache then
 * streams from PSRAM twice. Here the offset term rides in the same pass:
 *   sum((in + io) * f) = sum(in * f) + sum(io * f)
 * Both sums go into ACCX, the second as a MAC against a broadcast of io.
 * io = 128 does not fit s8, so it runs as 127 + 1 (one more MAC). In general
 * io = io_hi + io_lo, with io_hi clamped to s8 and io_lo the rest, which is
 * s8 for io in [-256, 254].
 *
 * Filter rows that are all 16-byte aligned (aligned base, row_len % 16 == 0)
 * run a loop that fuses each MAC with the next load. Otherwise the filter
 * goes through USAR+QUP, and the last chunk of each row (1..16 bytes) runs
 * against a zero-padded copy of the input tail and of the io vectors. Filter
 * bytes past the row then meet zero lanes, so the row needs no scalar tail.
 *
 * Requant uses fc_requant() (esp_nn_fc_requant_esp32s3.h). Per row it costs
 * more than the MACs of a short row.
 *
 * Memory reads:
 * - Filter: aligned blocks only, and only blocks that hold live bytes of the
 *   row. Aligned rows also touch the next row's first word. On the last row
 *   that is one aligned word past the filter.
 * - Input: whole blocks inside the input, plus the stack copies. Aligned rows
 *   with an odd block count read one more aligned block past the input.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <common_functions.h>
#include "esp_nn_fc_requant_esp32s3.h"

/* Per-call constants for the last chunk of every row. */
typedef struct {
    int8_t in[16];  /* input tail, zero past row_len */
    int8_t hi[16];  /* io (clamped to s8), zero past row_len */
    int8_t lo[16];  /* io_lo, zero past row_len */
} fc_fused_tail_t;

/* ACCX = sum over the row of f * (in [+ io_hi [+ io_lo]]), for any filter
 * alignment. Chunks 0..n-1 (n = chunks - 1) run from `in`, the last chunk
 * from the tail copy `t`. Filter chunks come from two raw blocks through
 * USAR+QUP. The next raw block is loaded three instructions before the QUP
 * that uses it, so raw block k + 2 is loaded during chunk k. The loop stops
 * one chunk early so that no raw block past the row is loaded: `extra` says
 * whether the last chunk spills into one more block. MACS are the io MACs of
 * a chunk (q4 * q5, q4 * q6), TAIL_MACS those of the last chunk, against the
 * zero-padded io vectors that follow the input tail in `t`.
 * q5 and q6 hold the io_hi and io_lo broadcasts, loaded once per call.
 * Both row macros zero `n` for the ee.srs.accx shift. A C zero operand would
 * hold one more register through the row loop. ee.srs.accx follows the last
 * MAC with no gap: the core interlocks on ACCX like on any operand (TRM 1.7.1
 * "Data Hazard"), so the read waits for that MAC. A gap would only fill the
 * stall. */
#define FC_FUSED_ROW(MACS, TAIL_MACS)                                   \
    asm volatile (                                                      \
        "ee.zero.accx                             \n"                   \
        "ee.ld.128.usar.ip  q0, %[f], 16          \n"                   \
        "beqz               %[n], 3f              \n"                   \
        "ee.vld.128.ip      q1, %[in], 16         \n"                   \
        "ee.ld.128.usar.ip  q2, %[f], 16          \n"                   \
        "addi               %[n], %[n], -1        \n"                   \
        "loopgtz            %[n], 1f              \n"                   \
        "ee.src.q.qup       q4, q0, q2            \n"                   \
        "ee.ld.128.usar.ip  q2, %[f], 16          \n"                   \
        "ee.vmulas.s8.accx.ld.ip q1, %[in], 16, q4, q1 \n"              \
        MACS                                                            \
        "1:                                       \n"                   \
        "ee.src.q.qup       q4, q0, q2            \n"                   \
        "ee.vmulas.s8.accx  q4, q1                \n"                   \
        MACS                                                            \
        "3:                                       \n"                   \
        "ee.vld.128.ip      q1, %[t], 16          \n"                   \
        "beqz               %[x], 2f              \n"                   \
        "ee.ld.128.usar.ip  q2, %[f], 16          \n"                   \
        "2:                                       \n"                   \
        "ee.src.q.qup       q4, q0, q2            \n"                   \
        "ee.vmulas.s8.accx  q4, q1                \n"                   \
        TAIL_MACS                                                       \
        "movi.n             %[n], 0               \n"                   \
        "ee.srs.accx        %[acc], %[n], 0       \n"                   \
        : [acc] "=r" (acc), [f] "+r" (f), [in] "+r" (in), [n] "+r" (n), \
          [t] "+r" (t)                                                  \
        : [x] "r" (extra)                                               \
        : "memory")

/* Same sum for 16-byte aligned rows of nblk = nm1 + 1 whole chunks. Each MAC also
 * loads a block. The filter is in q0/q2, the input in q1/q3. Each pass of the
 * loop takes two blocks, A in q0 and B in q2, loads B and the next A back to
 * back, and then runs the MACs and the input loads. With a filter in PSRAM,
 * A and B are the two halves of a 32-byte cache line when the row starts on a
 * line. A miss returns the first half before the rest of the line arrives,
 * and the cache serves one miss at a time. So the work between the two loads
 * of a line is free, and the work after the second load is not.
 * The loop runs (nblk - 1) / 2 times, then 1 or 2 blocks remain. `f` ends at
 * the next row. Input blocks are loaded one pair ahead: with odd nblk, the
 * row reads one aligned block past the input.
 * The tail touches the next row's first word (l32i at `f`, one aligned word
 * past the filter on the last row). It returns at that word and the line
 * keeps filling, so the requant of this row runs during the fill.
 * q5 and q6 hold the io_hi and io_lo broadcasts, loaded once per call.
 * FB loads B into q2, fused with the io MAC of A when io != 0. I3/I4 load the
 * next pair's input blocks, fused with the MACs of B. HI0/HI2 are io MACs
 * without a load, LO0/LO2 the extra MACs for io == 128. Unused ones are empty. */
#define FC_FUSED_ROW_ALIGNED(FB, I3, I4, HI0, HI2, LO0, LO2)            \
    asm volatile (                                                      \
        "ee.zero.accx                             \n"                   \
        "ee.vld.128.ip      q0, %[f], 16          \n"                   \
        "ee.vld.128.ip      q1, %[in], 16         \n"                   \
        "ee.vld.128.ip      q3, %[in], 16         \n"                   \
        "srai               %[n], %[nm1], 1       \n"                   \
        "loopgtz            %[n], 1f              \n"                   \
        LO0 FB                                                          \
        "ee.vmulas.s8.accx.ld.ip q0, %[f], 16, q0, q1 \n"              \
        I3 I4 LO2                                                       \
        "1:                                       \n"                   \
        "bbsi               %[nm1], 0, 2f         \n"                   \
        "l32i               %[n], %[f], 0         \n"                   \
        "ee.vmulas.s8.accx  q0, q1                \n"                   \
        HI0 LO0                                                         \
        "j                  3f                    \n"                   \
        "2:                                       \n"                   \
        LO0 FB                                                          \
        "l32i               %[n], %[f], 0         \n"                   \
        "ee.vmulas.s8.accx  q0, q1                \n"                   \
        "ee.vmulas.s8.accx  q2, q3                \n"                   \
        HI2 LO2                                                         \
        "3:                                       \n"                   \
        "movi.n             %[n], 0               \n"                   \
        "ee.srs.accx        %[acc], %[n], 0       \n"                   \
        : [acc] "=r" (acc), [f] "+r" (f), [in] "+r" (in), [n] "=&r" (n) \
        : [nm1] "r" (chunks - 1)                                        \
        : "memory")

#define FC_AL_LDF(q)        "ee.vld.128.ip      " #q ", %[f], 16 \n"
#define FC_AL_MAC_LDF(q, x) "ee.vmulas.s8.accx.ld.ip " #q ", %[f], 16, " #x ", q5 \n"
#define FC_AL_MAC(x, y)     "ee.vmulas.s8.accx  " #x ", " #y " \n"
#define FC_AL_LDI(q)        "ee.vld.128.ip      " #q ", %[in], 16 \n"
#define FC_AL_MAC_LDI(q, x, y) "ee.vmulas.s8.accx.ld.ip " #q ", %[in], 16, " #x ", " #y " \n"
#define FC_MAC_HI   "ee.vmulas.s8.accx  q4, q5                \n"
#define FC_MAC_LO   "ee.vmulas.s8.accx  q4, q6                \n"
#define FC_TAIL_HI  "ee.vld.128.ip      q3, %[t], 16          \n" \
                    "ee.vmulas.s8.accx  q4, q3                \n"
#define FC_TAIL_LO  "ee.vld.128.ip      q7, %[t], 16          \n" \
                    "ee.vmulas.s8.accx  q4, q7                \n"

/* One aligned row into acc. */
#define FC_ROW_AL(...) do {                                             \
        int32_t n;                                                      \
        FC_FUSED_ROW_ALIGNED(__VA_ARGS__);                              \
    } while (0)

/* One row of any alignment into acc. A row needs one block more than its
 * chunks when its misalignment pushes its end into another block. */
#define FC_ROW_UN(...) do {                                             \
        int32_t n = chunks - 1;                                         \
        const int8_t *t = tail.in;                                      \
        const int32_t extra = ((uintptr_t)f & 15) + row_len > (chunks << 4); \
        FC_FUSED_ROW(__VA_ARGS__);                                      \
    } while (0)

/* The row loop of one variant: ROW, then bias, requant, offset and clamp.
 * Each variant gets its own loop, so the variant is picked once per call and
 * not once per row. NEXT is the next row's filter: an aligned row leaves `f`
 * there, an unaligned one does not. */
#define FC_EACH_ROW(ROW, NEXT, MULT, SHIFT)                             \
    for (int ch = 0; ch < out_channels; ch++) {                         \
        const int8_t *f = f_row;                                        \
        const int8_t *in = input_data;                                  \
        int32_t acc;                                                    \
        ROW;                                                            \
        f_row = NEXT;                                                   \
        acc += bias_p[ch * bias_step];                                  \
        acc = fc_requant(acc, MULT, SHIFT) + out_offset;                \
        acc = max(acc, activation_min);                                 \
        out_data[ch] = (int8_t)min(acc, activation_max);                \
    }

/* The three aligned variants, by input offset. */
#define FC_ALIGNED_ROWS(MULT, SHIFT) do {                               \
        if (io_lo) {                                                    \
            FC_EACH_ROW(FC_ROW_AL(FC_AL_MAC_LDF(q2, q0),                \
                                  FC_AL_MAC_LDI(q1, q2, q5), FC_AL_MAC_LDI(q3, q2, q3), \
                                  FC_AL_MAC(q0, q5), FC_AL_MAC(q2, q5), \
                                  FC_AL_MAC(q0, q6), FC_AL_MAC(q2, q6)), \
                        f, MULT, SHIFT);                                \
        } else if (io_hi) {                                             \
            FC_EACH_ROW(FC_ROW_AL(FC_AL_MAC_LDF(q2, q0),                \
                                  FC_AL_MAC_LDI(q1, q2, q5), FC_AL_MAC_LDI(q3, q2, q3), \
                                  FC_AL_MAC(q0, q5), FC_AL_MAC(q2, q5), "", ""), \
                        f, MULT, SHIFT);                                \
        } else {                                                        \
            FC_EACH_ROW(FC_ROW_AL(FC_AL_LDF(q2), FC_AL_MAC_LDI(q1, q2, q3), FC_AL_LDI(q3), \
                                  "", "", "", ""),                      \
                        f, MULT, SHIFT);                                \
        }                                                               \
    } while (0)

void esp_nn_fc_fused_s8_esp32s3(const int8_t *input_data,
                                const int32_t input_offset,
                                const uint16_t row_len,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                int8_t *out_data,
                                const uint16_t out_channels,
                                const int32_t out_offset,
                                const int32_t *out_shift,
                                const int32_t *out_mult,
                                const int32_t quant_step,
                                const int32_t activation_min,
                                const int32_t activation_max)
{
    const int32_t chunks = (row_len + 15) >> 4;
    const int32_t body = (chunks - 1) << 4;
    const int32_t rem = row_len - body; /* 1..16 */
    /* 127 + 1 for io == 128, the only API value outside s8. */
    const int8_t io_hi = (int8_t)max(-128, min(input_offset, 127));
    const int8_t io_lo = (int8_t)(input_offset - io_hi);
    /* io_hi and io_lo, for the two broadcasts. */
    const int8_t bcast[2] = { io_hi, io_lo };
    /* NULL bias reads one zero with step 0: no branch per row. */
    static const int32_t no_bias = 0;
    const int32_t *bias_p = bias ? bias : &no_bias;
    const int32_t bias_step = bias ? 1 : 0;
    const bool rows_aligned = (((uintptr_t)filter_data | row_len) & 15) == 0;

    fc_fused_tail_t tail __attribute__((aligned(16)));
    if (!rows_aligned) {
        memset(&tail, 0, sizeof(tail));
        memcpy(tail.in, input_data + body, rem);
        memset(tail.hi, io_hi, rem);
        memset(tail.lo, io_lo, rem);
    }

    /* q5 = io_hi and q6 = io_lo in every lane, for all rows. The row asm blocks
     * read them and nothing else between them touches q registers: the
     * compiler never allocates them, and the loops make no calls. */
    const int8_t *f_row = filter_data;
    const int8_t *bc = bcast;
    asm volatile (
        "ee.vldbc.8.ip      q5, %[bc], 1          \n"
        "ee.vldbc.8         q6, %[bc]             \n"
        : [bc] "+r" (bc) : : "memory");

    if (rows_aligned) {
        if (quant_step == 0) {
            /* Per-tensor: mult and shift are loop invariants, so the
             * compiler hoists the shift maths out of the row loop. */
            const int32_t m = out_mult[0];
            const int32_t sh = out_shift[0];
            FC_ALIGNED_ROWS(m, sh);
        } else {
            FC_ALIGNED_ROWS(out_mult[ch], out_shift[ch]);
        }
        return;
    }

    if (io_lo) {
        FC_EACH_ROW(FC_ROW_UN(FC_MAC_HI FC_MAC_LO, FC_TAIL_HI FC_TAIL_LO), f_row + row_len,
                    out_mult[ch * quant_step], out_shift[ch * quant_step]);
    } else if (io_hi) {
        FC_EACH_ROW(FC_ROW_UN(FC_MAC_HI, FC_TAIL_HI), f_row + row_len,
                    out_mult[ch * quant_step], out_shift[ch * quant_step]);
    } else {
        FC_EACH_ROW(FC_ROW_UN("", ""), f_row + row_len,
                    out_mult[ch * quant_step], out_shift[ch * quant_step]);
    }
}
