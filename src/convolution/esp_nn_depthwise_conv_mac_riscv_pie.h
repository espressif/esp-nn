/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-P4 / S31 PIE helpers of the ch_mult == 1 depthwise kernel in
 * esp_nn_depthwise_conv_riscv_pie.c. Each is out of line, with its
 * arguments in a0-a7, so the asm runs as one leaf routine.
 * Each asm block starts 4-byte aligned, and its gas macros are purged at
 * its end, so a cloned copy of a function assembles too.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* Position state of esp_nn_dw_win_riscv_pie. The asm reads it by offset. */
typedef struct {
    const int8_t *ip;   /* 0: input of the next position */
    int32_t oxl;        /* 4: positions left in its output row, >= 1 */
    int32_t row_len;    /* 8: positions per output row */
    int32_t step;       /* 12: input step between positions */
    int32_t wrap;       /* 16: extra input step to the next row */
    int32_t np;         /* 20: positions of this call */
    const int8_t *fp;   /* 24: filter of the block */
    const int32_t *comb; /* 28: QACC start of the block */
    int32_t *acc;       /* 32: 16 words per position */
    int32_t ch;         /* 36: input and filter tap step */
    int32_t rows;       /* 40 */
    int32_t cols;       /* 44 */
    int32_t ip_row;     /* 48: input step from the end of a row */
} dw_win_t;

_Static_assert(offsetof(dw_win_t, ip_row) == 48, "dw_win_t layout");

/*
 * QACC += in[tap] * filter[tap] (16 int8 lanes) over rows x cols taps.
 * Pointers step by *_tap after each tap and by *_row after each row.
 * A zero tap step on a 16-byte pad vector feeds a constant input.
 * QACC is not zeroed. rows or cols of 0 do nothing. Clobbers q0-q5.
 *
 * Taps run in groups of 3: the loads go out before the MACs, and the first
 * MAC carries the third input load, so the load latency is hidden instead
 * of stalling every MAC.
 */
static __attribute__((noinline))
void esp_nn_dw_mac_rows_riscv_pie(const int8_t *ip, const int8_t *fp,
                                  int32_t ip_tap, int32_t fp_tap,
                                  int32_t rows, int32_t cols,
                                  int32_t ip_row, int32_t fp_row)
{
    register const int8_t *a0 asm("a0") = ip;
    register const int8_t *a1 asm("a1") = fp;
    register int32_t a2 asm("a2") = ip_tap;
    register int32_t a3 asm("a3") = fp_tap;
    register int32_t a4 asm("a4") = rows;
    register int32_t a5 asm("a5") = cols;
    register int32_t a6 asm("a6") = ip_row;
    register int32_t a7 asm("a7") = fp_row;
    asm volatile (
        ".balign 4                              \n\t"
        "beqz    a4, 9f                         \n\t"
        "beqz    a5, 9f                         \n\t"
        "mv      x30, a0                        \n\t"
        "mv      x31, a1                        \n\t"
        "mv      x28, a2                        \n\t"
        "mv      x29, a3                        \n\t"
        "li      t1, 3                          \n\t"
        "1:                                     \n\t"
        "mv      t0, a5                         \n\t"
        "blt     t0, t1, 3f                     \n\t"
        "2:                                     \n\t"
        "esp.vld.128.xp  q0, x30, x28           \n\t"
        "esp.vld.128.xp  q1, x31, x29           \n\t"
        "esp.vld.128.xp  q2, x30, x28           \n\t"
        "esp.vld.128.xp  q3, x31, x29           \n\t"
        "esp.vld.128.xp  q5, x31, x29           \n\t"
        "esp.vmulas.s8.qacc.ld.xp q4, x30, x28, q0, q1 \n\t"
        "esp.vmulas.s8.qacc q2, q3              \n\t"
        "esp.vmulas.s8.qacc q4, q5              \n\t"
        "addi    t0, t0, -3                     \n\t"
        "bge     t0, t1, 2b                     \n\t"
        "3:                                     \n\t"
        "beqz    t0, 5f                         \n\t"
        "esp.vld.128.xp  q0, x30, x28           \n\t"
        "esp.vld.128.xp  q1, x31, x29           \n\t"
        "addi    t0, t0, -1                     \n\t"
        "beqz    t0, 4f                         \n\t"
        "esp.vld.128.xp  q2, x30, x28           \n\t"
        "esp.vld.128.xp  q3, x31, x29           \n\t"
        "esp.vmulas.s8.qacc q2, q3              \n\t"
        "4:                                     \n\t"
        "esp.vmulas.s8.qacc q0, q1              \n\t"
        "5:                                     \n\t"
        "add     x30, x30, a6                   \n\t"
        "add     x31, x31, a7                   \n\t"
        "addi    a4, a4, -1                     \n\t"
        "bnez    a4, 1b                         \n\t"
        "9:                                     \n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5),
          "+r"(a6), "+r"(a7)
        :
        : "t0", "t1", "x28", "x29", "x30", "x31", "memory"
    );
}

/*
 * Full-window MAC of w->np interior positions for one block of 16
 * channels. For each position: QACC = comb, then QACC += the rows x cols
 * window at ip (input taps step by ch, rows by w->ip_row; the filter is
 * contiguous with step ch), then QACC is stored to w->acc + 16 * p.
 * Positions run along an output row: ip steps by w->step, and after
 * w->row_len positions it also steps by w->wrap to the next row. w->oxl
 * is the number of positions left in the current row (>= 1). The asm
 * generates the positions, so the caller needs no per-position work.
 * w->ip and w->oxl are written back for the next call. Clobbers q0-q5.
 *
 * 3x3 and 5x5 windows are fully unrolled as a software pipeline: tap t
 * runs one MAC that also loads the input of tap t + 2, and one load of
 * the filter of tap t + 2. That is two instructions per tap, and every
 * load is used at least 2 instructions later. Taps rotate over q0-q5.
 */
#define DW_WIN_MACROS \
    /* QACC = the 16 int32 lanes of comb at a3. */ \
    ".macro DW_COMB_LOAD                    \n\t" \
    "mv      x30, a3                        \n\t" \
    "esp.ld.qacc.l.l.128.ip x30, 16         \n\t" \
    "esp.ld.qacc.l.h.128.ip x30, 16         \n\t" \
    "esp.ld.qacc.h.l.128.ip x30, 16         \n\t" \
    "esp.ld.qacc.h.h.128.ip x30, 0          \n\t" \
    ".endm                                  \n\t" \
    /* Position start of the unrolled windows: QACC = comb, then the \
     * input and filter of taps 0 and 1 into q0-q3. a0 = ip. */ \
    ".macro DW_WIN_BEGIN                    \n\t" \
    "DW_COMB_LOAD                           \n\t" \
    "mv      x30, a0                        \n\t" \
    "mv      x31, a2                        \n\t" \
    "esp.vld.128.xp  q0, x30, x28           \n\t" \
    "esp.vld.128.xp  q1, x31, x28           \n\t" \
    "esp.vld.128.xp  q2, x30, x28           \n\t" \
    "esp.vld.128.xp  q3, x31, x28           \n\t" \
    ".endm                                  \n\t" \
    /* One tap: QACC += qa * qb, load the input of a later tap to qn \
     * (input pointer steps by inc) and its filter to qf. */ \
    ".macro DW_TAP qn, inc, qa, qb, qf      \n\t" \
    "esp.vmulas.s8.qacc.ld.xp \\qn, x30, \\inc, \\qa, \\qb \n\t" \
    "esp.vld.128.xp  \\qf, x31, x28         \n\t" \
    ".endm                                  \n\t" \
    /* Position state of the unrolled windows to registers: t0 = struct, \
     * a0 = ip, t2 = oxl, a5 = step, a6 = row_len, a7 = wrap. */ \
    ".macro DW_STATE_LD                     \n\t" \
    "mv      t0, a0                         \n\t" \
    "lw      a0, 0(t0)                      \n\t" \
    "lw      t2, 4(t0)                      \n\t" \
    "lw      a6, 8(t0)                      \n\t" \
    "lw      a5, 12(t0)                     \n\t" \
    "lw      a7, 16(t0)                     \n\t" \
    ".endm                                  \n\t" \
    /* Store QACC to acc and step acc. */ \
    ".macro DW_ACC_ST                       \n\t" \
    "mv      x30, a4                        \n\t" \
    "esp.st.qacc.l.l.128.ip x30, 16         \n\t" \
    "esp.st.qacc.l.h.128.ip x30, 16         \n\t" \
    "esp.st.qacc.h.l.128.ip x30, 16         \n\t" \
    "esp.st.qacc.h.h.128.ip x30, 16         \n\t" \
    "mv      a4, x30                        \n\t" \
    ".endm                                  \n\t" \
    /* Position end of the unrolled windows: next position in registers, \
     * loop, or write ip and oxl back to the struct at t0 and finish. */ \
    ".macro DW_WIN_END loop                 \n\t" \
    "DW_ACC_ST                              \n\t" \
    "add     a0, a0, a5                     \n\t" \
    "addi    t2, t2, -1                     \n\t" \
    "bnez    t2, 91f                        \n\t" \
    "mv      t2, a6                         \n\t" \
    "add     a0, a0, a7                     \n\t" \
    "91:                                    \n\t" \
    "addi    a1, a1, -1                     \n\t" \
    "bnez    a1, \\loop                     \n\t" \
    "sw      a0, 0(t0)                      \n\t" \
    "sw      t2, 4(t0)                      \n\t" \
    "j       99f                            \n\t" \
    ".endm                                  \n\t"

#define DW_WIN_PURGE \
    ".purgem DW_COMB_LOAD                   \n\t" \
    ".purgem DW_WIN_BEGIN                   \n\t" \
    ".purgem DW_TAP                         \n\t" \
    ".purgem DW_STATE_LD                    \n\t" \
    ".purgem DW_ACC_ST                      \n\t" \
    ".purgem DW_WIN_END                     \n\t"

static __attribute__((noinline)) void esp_nn_dw_win_riscv_pie(dw_win_t *w)
{
    register dw_win_t *a0 asm("a0") = w;
    asm volatile (
        DW_WIN_MACROS
        ".balign 4                              \n\t"
        "lw      a1, 20(a0)                     \n\t"
        "lw      a2, 24(a0)                     \n\t"
        "lw      a3, 28(a0)                     \n\t"
        "lw      a4, 32(a0)                     \n\t"
        "lw      x28, 36(a0)                    \n\t"
        "lw      a5, 40(a0)                     \n\t"
        "lw      a6, 44(a0)                     \n\t"
        "lw      a7, 48(a0)                     \n\t"
        "li      t1, 3                          \n\t"
        /* x29: input step from the last tap of a row to the next row */
        "add     x29, a7, x28                   \n\t"
        "bne     a5, a6, 1f                     \n\t"
        "beq     a6, t1, 10f                    \n\t"
        "li      t0, 5                          \n\t"
        "beq     a6, t0, 11f                    \n\t"
        /* Any other window: rows, cols and the position state stay in
         * the struct at a0, as the loops use all free registers. */
        "1:                                     \n\t"
        "DW_COMB_LOAD                           \n\t"
        "lw      x30, 0(a0)                     \n\t"
        "mv      x31, a2                        \n\t"
        "mv      t2, a5                         \n\t"
        "2:                                     \n\t"
        "mv      t0, a6                         \n\t"
        "blt     t0, t1, 4f                     \n\t"
        "3:                                     \n\t"
        "esp.vld.128.xp  q0, x30, x28           \n\t"
        "esp.vld.128.xp  q1, x31, x28           \n\t"
        "esp.vld.128.xp  q2, x30, x28           \n\t"
        "esp.vld.128.xp  q3, x31, x28           \n\t"
        "esp.vld.128.xp  q5, x31, x28           \n\t"
        "esp.vmulas.s8.qacc.ld.xp q4, x30, x28, q0, q1 \n\t"
        "esp.vmulas.s8.qacc q2, q3              \n\t"
        "esp.vmulas.s8.qacc q4, q5              \n\t"
        "addi    t0, t0, -3                     \n\t"
        "bge     t0, t1, 3b                     \n\t"
        "4:                                     \n\t"
        "beqz    t0, 6f                         \n\t"
        "esp.vld.128.xp  q0, x30, x28           \n\t"
        "esp.vld.128.xp  q1, x31, x28           \n\t"
        "addi    t0, t0, -1                     \n\t"
        "beqz    t0, 5f                         \n\t"
        "esp.vld.128.xp  q2, x30, x28           \n\t"
        "esp.vld.128.xp  q3, x31, x28           \n\t"
        "esp.vmulas.s8.qacc q2, q3              \n\t"
        "5:                                     \n\t"
        "esp.vmulas.s8.qacc q0, q1              \n\t"
        "6:                                     \n\t"
        "add     x30, x30, a7                   \n\t"
        "addi    t2, t2, -1                     \n\t"
        "bnez    t2, 2b                         \n\t"
        "DW_ACC_ST                              \n\t"
        "lw      x30, 0(a0)                     \n\t"
        "lw      x31, 12(a0)                    \n\t"
        "lw      t0, 4(a0)                      \n\t"
        "add     x30, x30, x31                  \n\t"
        "addi    t0, t0, -1                     \n\t"
        "bnez    t0, 7f                         \n\t"
        "lw      t0, 8(a0)                      \n\t"
        "lw      x31, 16(a0)                    \n\t"
        "add     x30, x30, x31                  \n\t"
        "7:                                     \n\t"
        "sw      x30, 0(a0)                     \n\t"
        "sw      t0, 4(a0)                      \n\t"
        "addi    a1, a1, -1                     \n\t"
        "bnez    a1, 1b                         \n\t"
        "j       99f                            \n\t"
        /* 3x3 and 5x5: the position state moves to registers. */
        "10:                                    \n\t"
        "DW_STATE_LD                            \n\t"
        "j       20f                            \n\t"
        "11:                                    \n\t"
        "DW_STATE_LD                            \n\t"
        "j       30f                            \n\t"
        "20:                                    \n\t"
        "DW_WIN_BEGIN                           \n\t"
        "DW_TAP  q4, x29, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x29, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x29, q0, q1, q5            \n\t"
        "esp.vmulas.s8.qacc q2, q3              \n\t"
        "esp.vmulas.s8.qacc q4, q5              \n\t"
        "DW_WIN_END 20b                         \n\t"
        "30:                                    \n\t"
        "DW_WIN_BEGIN                           \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x29, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x29, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x29, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x29, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x28, q2, q3, q1            \n\t"
        "DW_TAP  q2, x28, q4, q5, q3            \n\t"
        "DW_TAP  q4, x28, q0, q1, q5            \n\t"
        "DW_TAP  q0, x29, q2, q3, q1            \n\t"
        "esp.vmulas.s8.qacc q4, q5              \n\t"
        "esp.vmulas.s8.qacc q0, q1              \n\t"
        "DW_WIN_END 30b                         \n\t"
        "99:                                    \n\t"
        DW_WIN_PURGE
        : "+r"(a0)
        :
        : "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2",
          "x28", "x29", "x30", "x31", "memory"
    );
}

/*
 * Writes one row of a padded 16-channel input slab: nl pad vectors, nin
 * vectors of 16 bytes loaded from src (stepping by step), then nr pad
 * vectors. dst and pad are 16-byte aligned.
 */
static __attribute__((noinline))
void esp_nn_dw_slab_row_riscv_pie(int8_t *dst, const int8_t *src, int32_t step,
                                  int32_t nl, int32_t nin, int32_t nr,
                                  const int8_t *pad)
{
    register int8_t *a0 asm("a0") = dst;
    register const int8_t *a1 asm("a1") = src;
    register int32_t a2 asm("a2") = step;
    register int32_t a3 asm("a3") = nl;
    register int32_t a4 asm("a4") = nin;
    register int32_t a5 asm("a5") = nr;
    register const int8_t *a6 asm("a6") = pad;
    asm volatile (
        ".balign 4                              \n\t"
        "mv      x30, a6                        \n\t"
        "esp.vld.128.ip  q0, x30, 0             \n\t"
        "mv      x30, a0                        \n\t"
        "mv      x31, a1                        \n\t"
        "mv      x28, a2                        \n\t"
        "beqz    a3, 2f                         \n\t"
        "1:                                     \n\t"
        "esp.vst.128.ip  q0, x30, 16            \n\t"
        "addi    a3, a3, -1                     \n\t"
        "bnez    a3, 1b                         \n\t"
        "2:                                     \n\t"
        /* two loads in flight before their stores hide the load latency */
        "li      t0, 2                          \n\t"
        "blt     a4, t0, 31f                    \n\t"
        "3:                                     \n\t"
        "esp.vld.128.xp  q1, x31, x28           \n\t"
        "esp.vld.128.xp  q2, x31, x28           \n\t"
        "addi    a4, a4, -2                     \n\t"
        "esp.vst.128.ip  q1, x30, 16            \n\t"
        "esp.vst.128.ip  q2, x30, 16            \n\t"
        "bge     a4, t0, 3b                     \n\t"
        "31:                                    \n\t"
        "beqz    a4, 4f                         \n\t"
        "esp.vld.128.xp  q1, x31, x28           \n\t"
        "esp.vst.128.ip  q1, x30, 16            \n\t"
        "4:                                     \n\t"
        "beqz    a5, 6f                         \n\t"
        "5:                                     \n\t"
        "esp.vst.128.ip  q0, x30, 16            \n\t"
        "addi    a5, a5, -1                     \n\t"
        "bnez    a5, 5b                         \n\t"
        "6:                                     \n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5),
          "+r"(a6)
        :
        : "t0", "x28", "x30", "x31", "memory"
    );
}

/*
 * Requantizes, clamps and stores n >= 1 positions of a 16-channel block.
 * acc holds v = acc + comb, 16 words per position, |v| < 2^26.
 * Position p goes to opp[p] + ci, or to op + p * ostride when opp is NULL.
 * halves = 2 stores 16 channels, 1 stores the low 8.
 *
 * tbl is 16-byte aligned. Per half it holds 7 vectors of 8 int16 lanes:
 * - ml, mh: mult = mh * 2^16 + ml, with ml signed.
 * - ra, rb: ra * rb = (1 + 2 * out_offset) << rs.
 * - sh = rs + 1.
 * - lo, hi: the activation range.
 * With v = vh * 2^16 + vl (vl signed) and H = floor(v * mult / 2^30):
 *   H = 4 vh mh + floor((vh ml + vl mh + floor(vl ml / 2^16)) / 2^14),
 * every product a 16 x 16 bit one, summed in the 64-bit QACC lanes; srcmb
 * floors in place. The result is (H + 1 - 2 (v < 0) + ra rb) >> (rs + 1),
 * which is requant(v) + out_offset, saturated to int16, then clamped.
 *
 * VRQ_HALF requantizes 8 lanes, one half of a position. x30 = acc,
 * x31 = table half, x26 = tmp, q6 = 16-bit ones, q7 = zero. The 8 bytes go
 * to x26. The first half writes at the 16-byte aligned acc row and the
 * second half 8 bytes later. The 16 bytes are then read back from
 * the row start with one esp.vld.128 and stored to the output.
 * Clobbers q0-q5 and QACC.
 */
static __attribute__((noinline))
void esp_nn_dw_vrq_riscv_pie(const int32_t *acc, int32_t n, const int16_t *tbl,
                             int8_t *op, int32_t ostride, int8_t *const *opp,
                             int32_t ci, int32_t halves)
{
    register const int32_t *a0 asm("a0") = acc;
    register int32_t a1 asm("a1") = n;
    register const int16_t *a2 asm("a2") = tbl;
    register int8_t *a3 asm("a3") = op;
    register int32_t a4 asm("a4") = ostride;
    register int8_t *const *a5 asm("a5") = opp;
    register int32_t a6 asm("a6") = ci;
    register int32_t a7 asm("a7") = halves;
    asm volatile (
        ".macro VRQ_HALF step                   \n\t"
        "esp.vld.128.ip  q0, x30, 16            \n\t"
        "esp.vld.128.ip  q1, x30, 16            \n\t"
        "esp.vld.128.ip  q4, x31, 16            \n\t"
        "esp.vunzip.16   q0, q1                 \n\t"
        "esp.vld.128.ip  q5, x31, 16            \n\t"
        "esp.vcmp.lt.s16 q2, q0, q7             \n\t"
        "esp.zero.qacc                          \n\t"
        "esp.vsub.s16    q3, q1, q2             \n\t"
        "esp.vmulas.s16.qacc q0, q4             \n\t"
        "esp.vcmp.lt.s16 q2, q1, q7             \n\t"
        "esp.srcmb.s16.qacc q1, x29, 1          \n\t"
        "esp.vmulas.s16.qacc q3, q4             \n\t"
        "esp.vmulas.s16.qacc.ld.ip q4, x31, 16, q0, q5 \n\t"
        "esp.vadd.s16    q3, q3, q3             \n\t"
        "esp.orq         q2, q2, q6             \n\t"
        "esp.vadd.s16    q3, q3, q3             \n\t"
        "esp.srcmb.s16.qacc q1, x28, 1          \n\t"
        "esp.vmulas.s16.qacc.ld.ip q0, x31, 16, q3, q5 \n\t"
        "esp.vmulas.s16.qacc.ld.ip q5, x31, 16, q2, q6 \n\t"
        "esp.vmulas.s16.qacc.ld.ip q2, x31, 16, q4, q0 \n\t"
        "esp.vld.128.ip  q3, x31, 16            \n\t"
        "esp.srcmb.s16.q.qacc q1, q5, 1         \n\t"
        "esp.vmax.s16    q1, q1, q2             \n\t"
        "esp.vmin.s16    q1, q1, q3             \n\t"
        "esp.vunzip.8    q1, q0                 \n\t"
        "esp.vst.l.64.ip q1, x26, \\step        \n\t"
        ".endm                                  \n\t"
        ".balign 4                              \n\t"
        "mv      x30, a0                        \n\t"
        "li      x28, 14                        \n\t"
        "li      x29, 16                        \n\t"
        /* q7 = 0, q6 = 16-bit ones */
        "esp.zero.q      q7                     \n\t"
        "esp.vcmp.eq.s16 q6, q7, q7             \n\t"
        "esp.vsub.s16    q6, q7, q6             \n\t"
        "1:                                     \n\t"
        "beqz    a5, 2f                         \n\t"
        "lw      x27, 0(a5)                     \n\t"
        "addi    a5, a5, 4                      \n\t"
        "add     x27, x27, a6                   \n\t"
        "j       3f                             \n\t"
        "2:                                     \n\t"
        "mv      x27, a3                        \n\t"
        "add     a3, a3, a4                     \n\t"
        "3:                                     \n\t"
        /* The int8 results go to the start of this position's acc row,
         * which is consumed by then, and from there to the output. */
        "mv      x26, x30                       \n\t"
        "mv      x31, a2                        \n\t"
        "li      t0, 1                          \n\t"
        "beq     a7, t0, 4f                     \n\t"
        "VRQ_HALF 8                             \n\t"
        "VRQ_HALF -8                            \n\t"
        "esp.vld.128.ip  q1, x26, 0             \n\t"
        "esp.vst.128.ip  q1, x27, 0             \n\t"
        "j       5f                             \n\t"
        "4:                                     \n\t"
        /* low 8 channels only: byte stores, as the output may be unaligned */
        "VRQ_HALF 0                             \n\t"
        "addi    x30, x30, 32                   \n\t"
        "lw      t1, 0(x26)                     \n\t"
        "lw      t2, 4(x26)                     \n\t"
        "sb      t1, 0(x27)                     \n\t"
        "srli    t1, t1, 8                      \n\t"
        "sb      t1, 1(x27)                     \n\t"
        "srli    t1, t1, 8                      \n\t"
        "sb      t1, 2(x27)                     \n\t"
        "srli    t1, t1, 8                      \n\t"
        "sb      t1, 3(x27)                     \n\t"
        "sb      t2, 4(x27)                     \n\t"
        "srli    t2, t2, 8                      \n\t"
        "sb      t2, 5(x27)                     \n\t"
        "srli    t2, t2, 8                      \n\t"
        "sb      t2, 6(x27)                     \n\t"
        "srli    t2, t2, 8                      \n\t"
        "sb      t2, 7(x27)                     \n\t"
        "5:                                     \n\t"
        "addi    a1, a1, -1                     \n\t"
        "bnez    a1, 1b                         \n\t"
        ".purgem VRQ_HALF                       \n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5),
          "+r"(a6), "+r"(a7)
        :
        : "t0", "t1", "t2", "x26", "x27", "x28", "x29", "x30", "x31",
          "memory"
    );
}
