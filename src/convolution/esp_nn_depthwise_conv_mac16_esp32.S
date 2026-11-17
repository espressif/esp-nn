// Copyright 2026 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * ESP32 (MAC16) helpers for 3x3 depthwise conv.
 *
 * Pure .S: the dot and widen routines use all 14 address registers, and
 * some also need stack slots. Inline asm would leave the compiler too few
 * registers.
 *
 * The band layout is one s16 plane per channel, column-major:
 * sample (row r, column x) is at plane[x * DW_ROWS + r].
 * DW_ROWS is fixed, so every tap offset is an immediate. It comes from
 * esp_nn_conv_dispatch_esp32.h, which the C side uses too.
 */

#include "esp_nn_conv_dispatch_esp32.h"

#define COL_BYTES   (DW_ROWS * 2)

/*
 * void esp_nn_dw_widen_t_esp32(int16_t *dst,       a2 (plane 0, first column)
 *                              const int8_t *src,  a3 (first pixel, first channel)
 *                              int32_t n_x,        a4 (>= 0)
 *                              int32_t n_c,        a5 (>= 1)
 *                              const int32_t *p)   a6
 *
 * p[] = { src pixel step (bytes), plane step (bytes), offset }
 * For each pixel x and channel c: dst[c plane][x column] = src[x][c] + offset.
 * Four channels per step, each byte loaded 4 instructions before its first use.
 * No stack slots are used.
 */
    .text
/*
 * void esp_nn_dw_zero_esp32(int16_t *dst,  a2 (word aligned)
 *                           int32_t n_cols, a3 (columns of COL_BYTES)
 *                           int32_t half)   a4 (1: only rows 0 .. DW_ROWS / 2 - 1)
 *
 * Zero the planes of a chunk. Word stores in a zero-overhead loop: much
 * faster than the libc memset on these sizes. Bands of DW_ROWS / 2 rows or
 * fewer only zero the top half of each column.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_dw_zero_esp32
    .type   esp_nn_dw_zero_esp32,@function

esp_nn_dw_zero_esp32:
    entry   a1, 32
    movi    a5, 0
    bnez    a4, 2f
    loopnez a3, 1f
    s32i    a5, a2, 0
    s32i    a5, a2, 4
    s32i    a5, a2, 8
    s32i    a5, a2, 12
    s32i    a5, a2, 16
    s32i    a5, a2, 20
    s32i    a5, a2, 24
    s32i    a5, a2, 28
    addi    a2, a2, COL_BYTES
1:
    retw.n
2:
    loopnez a3, 3f
    s32i    a5, a2, 0
    s32i    a5, a2, 4
    s32i    a5, a2, 8
    s32i    a5, a2, 12
    addi    a2, a2, COL_BYTES
3:
    retw.n

    .size   esp_nn_dw_zero_esp32, . - esp_nn_dw_zero_esp32

    .align  4
    .global esp_nn_dw_widen_t_esp32
    .type   esp_nn_dw_widen_t_esp32,@function

esp_nn_dw_widen_t_esp32:
    entry   a1, 32

    l32i    a8, a6, 0           // src pixel step
    l32i    a9, a6, 4           // plane step
    l32i    a10, a6, 8          // offset
    bnei    a5, 1, 5f
    bnei    a8, 1, 8f
    /* One channel and src step 1: 4 pixels per pass, immediate offsets. */
    srli    a6, a4, 2
    loopnez a6, 7f
    l8ui    a12, a3, 0
    l8ui    a13, a3, 1
    l8ui    a14, a3, 2
    l8ui    a15, a3, 3
    sext    a12, a12, 7
    sext    a13, a13, 7
    sext    a14, a14, 7
    sext    a15, a15, 7
    add     a12, a12, a10
    add     a13, a13, a10
    add     a14, a14, a10
    add     a15, a15, a10
    s16i    a12, a2, 0
    s16i    a13, a2, COL_BYTES
    s16i    a14, a2, 2 * COL_BYTES
    s16i    a15, a2, 3 * COL_BYTES
    addi    a3, a3, 4
    addi    a2, a2, 4 * COL_BYTES
7:
    extui   a4, a4, 0, 2
8:  // one channel, any src step
    loopnez a4, 6f
    l8ui    a13, a3, 0
    add     a3, a3, a8
    sext    a13, a13, 7
    add     a13, a13, a10
    s16i    a13, a2, 0
    addi    a2, a2, COL_BYTES
6:
    retw.n
5:
    srli    a6, a5, 2           // channel groups of 4
    extui   a5, a5, 0, 2        // channels left over
    beqz    a4, 4f
1:  // pixel loop
    mov     a11, a3
    mov     a12, a2
    loopnez a6, 2f
    l8ui    a13, a11, 0
    l8ui    a14, a11, 1
    l8ui    a15, a11, 2
    l8ui    a7, a11, 3
    sext    a13, a13, 7
    sext    a14, a14, 7
    sext    a15, a15, 7
    sext    a7, a7, 7
    add     a13, a13, a10
    add     a14, a14, a10
    add     a15, a15, a10
    add     a7, a7, a10
    s16i    a13, a12, 0
    add     a12, a12, a9
    s16i    a14, a12, 0
    add     a12, a12, a9
    s16i    a15, a12, 0
    add     a12, a12, a9
    s16i    a7, a12, 0
    add     a12, a12, a9
    addi    a11, a11, 4
2:
    loopnez a5, 3f
    l8ui    a13, a11, 0
    addi    a11, a11, 1
    sext    a13, a13, 7
    add     a13, a13, a10
    s16i    a13, a12, 0
    add     a12, a12, a9
3:
    add     a3, a3, a8
    addi    a2, a2, COL_BYTES
    addi    a4, a4, -1
    bnez    a4, 1b
4:
    retw.n

    .size   esp_nn_dw_widen_t_esp32, . - esp_nn_dw_widen_t_esp32

/*
 * esp_nn_dw_widen_t_esp32 for offset 128, a word aligned source, pixel step
 * and channel count a multiple of 4. s + 128 is the byte s ^ 0x80 read as
 * unsigned, so one l32i and xor serve 4 channels. Two pixels per step, the
 * second at column offset COL_BYTES.
 *
 * void esp_nn_dw_widen_u8_esp32(int16_t *dst,       a2
 *                               const int8_t *src,  a3
 *                               int32_t n_x,        a4 (>= 0)
 *                               int32_t n_c,        a5 (multiple of 4, >= 4)
 *                               const int32_t *p)   a6 (as above, offset 128)
 *
 * Stack slots (entry a1, 64): 0 n_x, 4 channel groups left.
 * a4 and a5 are temps in the pixel loop.
 */
    .align  4
    .global esp_nn_dw_widen_u8_esp32
    .type   esp_nn_dw_widen_u8_esp32,@function

esp_nn_dw_widen_u8_esp32:
    entry   a1, 64

    l32i    a7, a6, 0           // src pixel step
    l32i    a8, a6, 4           // plane step
    movi    a9, 0x80808080
    srli    a5, a5, 2           // channel groups of 4
    s32i    a4, a1, 0
    beqz    a4, 9f
1:  // channel group loop
    mov     a6, a3
    mov     a10, a2             // 4 planes
    add     a11, a10, a8
    add     a12, a11, a8
    add     a13, a12, a8
    s32i    a5, a1, 4
    srli    a5, a4, 1
    loopnez a5, 2f
    l32i    a14, a6, 0
    add     a6, a6, a7
    l32i    a15, a6, 0
    add     a6, a6, a7
    xor     a14, a14, a9
    xor     a15, a15, a9
    extui   a4, a14, 0, 8
    extui   a5, a15, 0, 8
    s16i    a4, a10, 0
    s16i    a5, a10, COL_BYTES
    extui   a4, a14, 8, 8
    extui   a5, a15, 8, 8
    s16i    a4, a11, 0
    s16i    a5, a11, COL_BYTES
    extui   a4, a14, 16, 8
    extui   a5, a15, 16, 8
    s16i    a4, a12, 0
    s16i    a5, a12, COL_BYTES
    srli    a4, a14, 24
    srli    a5, a15, 24
    s16i    a4, a13, 0
    s16i    a5, a13, COL_BYTES
    addi    a10, a10, 2 * COL_BYTES
    addi    a11, a11, 2 * COL_BYTES
    addi    a12, a12, 2 * COL_BYTES
    addi    a13, a13, 2 * COL_BYTES
2:
    l32i    a4, a1, 0
    bbci    a4, 0, 3f           // odd n_x: the last pixel
    l32i    a14, a6, 0
    xor     a14, a14, a9
    extui   a5, a14, 0, 8
    s16i    a5, a10, 0
    extui   a5, a14, 8, 8
    s16i    a5, a11, 0
    extui   a5, a14, 16, 8
    s16i    a5, a12, 0
    srli    a5, a14, 24
    s16i    a5, a13, 0
3:
    l32i    a5, a1, 4
    addi    a3, a3, 4
    addx4   a2, a8, a2          // the next 4 planes
    addi    a5, a5, -1
    bnez    a5, 1b
9:
    retw.n

    .size   esp_nn_dw_widen_u8_esp32, . - esp_nn_dw_widen_u8_esp32

/*
 * 3x3 dot products of one channel over a band of output rows.
 *
 * void esp_nn_dw3x3_dot_esp32(const int16_t *plane,  a2 (top-left tap of output 0)
 *                             int32_t *sums,         a3
 *                             int32_t out_wd,        a4 (>= 1)
 *                             int32_t n_rows,        a5 (>= 1)
 *                             const int32_t *fp)     a6
 *
 * fp[] = { [f0 f1], [f2 f3], [f4 f5], [f6 f7] (s16 pairs, low half first),
 *          f8, bias, column step (bytes), row step (bytes) }
 * Tap t = ky * 3 + kx is at plane + ky * 2 + kx * COL_BYTES.
 * f0..f7 live in m0..m3 and f8 in a6. Each l16si is used 3 instructions later.
 * sums[row * out_wd + x] = bias + dot, low 32 bits of ACC.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_dw3x3_dot_esp32
    .type   esp_nn_dw3x3_dot_esp32,@function

esp_nn_dw3x3_dot_esp32:
    entry   a1, 32

    addi    a11, a6, -4
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a7, a6, 20          // bias
    l32i    a8, a6, 24          // column step
    l32i    a9, a6, 28          // row step
    l32i    a6, a6, 16          // f8
1:  // row loop
    mov     a10, a2
    loop    a4, 2f
    wsr.acclo a7                // ACCLO = bias. ACCHI does not affect ACCLO.
    l16si   a11, a10, 0                     // t0
    l16si   a12, a10, COL_BYTES             // t1
    l16si   a13, a10, 2 * COL_BYTES         // t2
    mula.da.ll m0, a11
    l16si   a14, a10, 2                     // t3
    mula.da.hl m0, a12
    l16si   a15, a10, 2 + COL_BYTES         // t4
    mula.da.ll m1, a13
    l16si   a11, a10, 2 + 2 * COL_BYTES     // t5
    mula.da.hl m1, a14
    l16si   a12, a10, 4                     // t6
    mula.ad.ll a15, m2
    l16si   a13, a10, 4 + COL_BYTES         // t7
    mula.ad.lh a11, m2
    l16si   a14, a10, 4 + 2 * COL_BYTES     // t8
    mula.ad.ll a12, m3
    add     a10, a10, a8
    mula.ad.lh a13, m3
    mula.aa.ll a14, a6
    rsr.acclo a15
    s32i    a15, a3, 0
    addi    a3, a3, 4
2:
    add     a2, a2, a9
    addi    a5, a5, -1
    bnez    a5, 1b
    retw.n

    .size   esp_nn_dw3x3_dot_esp32, . - esp_nn_dw3x3_dot_esp32

/*
 * One column step of the stride 1 row pair loop (13: in
 * esp_nn_dw3x3_dot_rq_esp32): outputs (r, x) and (r + 1, x).
 * A, B, C are the register pairs of columns x, x + 1, x + 2. Word 0 holds
 * rows (r, r + 1), word 1 rows (r + 2, r + 3), so the even tap packing
 * serves both rows, with other half selects for row r + 1.
 * C is loaded here. off is its column offset from a10.
 * prev = 1: finish the row r + 1 output of the previous step (in a2).
 * a3 stores row r, then moves on. a9 moves on, then the next step stores
 * row r + 1 at it.
 */
    .macro  DW_S1_STEP ca0, ca1, cb0, cb1, cc0, cc1, off, prev
    l32i    \cc0, a10, \off
    l32i    \cc1, a10, \off + 4
    mula.da.ll m0, \ca0                          // row r
    .if \prev
    mulsh   a2, a2, a4
    .endif
    mula.da.hh m0, \ca0
    mula.ad.ll \ca1, m3
    .if \prev
    add     a2, a2, a5
    .endif
    mula.da.ll m1, \cb0
    .if \prev
    sra     a2, a2
    .endif
    mula.da.hh m1, \cb0
    .if \prev
    clamps  a2, a2, 7
    .endif
    mula.ad.lh \cb1, m3
    .if \prev
    s8i     a2, a9, 0
    .endif
    mula.ad.ll \cc0, m2
    mula.ad.hh \cc0, m2
    mula.aa.ll a6, \cc1
    rsr.acclo a2
    wsr.acclo a7
    mula.da.lh m0, \ca0                          // row r + 1
    mulsh   a2, a2, a4                          // 3 after the ACC read
    mula.da.hl m0, \ca1
    mula.ad.hl \ca1, m3
    add     a2, a2, a5
    mula.da.lh m1, \cb0
    sra     a2, a2
    mula.da.hl m1, \cb1
    clamps  a2, a2, 7
    mula.ad.hh \cb1, m3
    s8i     a2, a3, 0
    l32i    a2, a1, 20                          // out step
    mula.ad.hl \cc0, m2
    mula.ad.lh \cc1, m2
    add     a3, a3, a2
    add     a9, a9, a2
    mula.aa.lh a6, \cc1
    rsr.acclo a2
    wsr.acclo a7
    .endm

/* Finish the row r + 1 output of the last DW_S1_STEP. */
    .macro  DW_S1_LAST
    mulsh   a2, a2, a4
    nop                         // mulsh result 3 later
    add     a2, a2, a5
    sra     a2, a2
    clamps  a2, a2, 7
    s8i     a2, a9, 0
    .endm

/*
 * esp_nn_dw3x3_dot_esp32 with the requant fused in, for taps and bias scaled by
 * 256. Same math as esp_nn_conv_rq256s_oc_esp32: ACC holds 256 * x,
 * and out = (mulsh(A, mult) + K - 128 * (A < 0)) >> (n + 7).
 * Right shift n in [1, 15], activation range [-128, 127] and |x| < 2^23.
 * Results go straight to s8 output.
 *
 * Rows r and r + 1 of a column are adjacent samples, so one l32i loads two
 * taps: rows (r, r+1) when the top row r is even (word aligned), and rows
 * (r+1, r+2) when it is odd. That is 6 loads per output instead of 9. Each
 * parity has its own filter packing and its own loop.
 *
 * void esp_nn_dw3x3_dot_rq_esp32(const int16_t *plane,  a2 (2-byte aligned)
 *                                int8_t *out,           a3
 *                                int32_t out_wd,        a4 (>= 1)
 *                                int32_t n_rows,        a5 (>= 1)
 *                                const int32_t *fp)     a6
 *
 * Taps f0..f8 row-major, times 256. fp[] =
 *  - 0..3: even rows [f0 f3], [f1 f4], [f2 f5], [f6 f7]; 4: f8;
 *  - 5: 256 * bias, 6: column step, 7: row step (bytes);
 *  - 8: mult, 9: n + 7, 10: K = 64 + 2^(n+6) + out_offset * 2^(n+7),
 *    11: out step (bytes), 12: K - 128;
 *  - 13..16: odd rows [f3 f6], [f4 f7], [f5 f8], [f0 f1]; 17: f2;
 *  - 18: out_wd / 3 for stride 1 (both ways), else 0; 19: out_wd % 3;
 *    20: (out_wd - 1) * out step.
 *
 * With out_offset -128 the caller passes fp[12] = K. The sign term is not
 * needed then, see esp_nn_conv_rq256s_oc_esp32. Those loops use a2
 * to load the next pixel's first word while mulsh runs. That load reads the
 * column after the last pixel, which the plane has (see dw_layout).
 * With fp[18] != 0 they run two stride 1 rows at a time with DW_S1_STEP.
 * The top row of a pair is always even then.
 *
 * Stack slots (entry a1, 96): 0 out_wd, 4 rows left, 8 row step,
 * 12 plane row, 16 fp, 20 out step, 24 and 28 for stride 1 (see 13:).
 * 32..44 are free.
 * a2 holds K - 128 after the setup.
 */
    .align  4
    .global esp_nn_dw3x3_dot_rq_esp32
    .type   esp_nn_dw3x3_dot_rq_esp32,@function

esp_nn_dw3x3_dot_rq_esp32:
    entry   a1, 96

    s32i    a4, a1, 0           // out_wd
    s32i    a5, a1, 4           // rows left
    s32i    a2, a1, 12          // plane row
    s32i    a6, a1, 16          // fp
    l32i    a9, a6, 28          // row step
    l32i    a7, a6, 20          // 256 * bias
    l32i    a8, a6, 24          // column step
    s32i    a9, a1, 8
    l32i    a4, a6, 32          // mult
    l32i    a9, a6, 36          // n + 7
    l32i    a5, a6, 40          // K
    l32i    a2, a6, 48          // K - 128
    ssr     a9
    l32i    a9, a6, 44          // out step
    s32i    a9, a1, 20
    beq     a2, a5, 6f          // out_offset -128
1:  // row loop
    l32i    a10, a1, 12
    l32i    a12, a1, 16
    l32i    a13, a1, 0
    wsr.acclo a7
    bbsi    a10, 1, 5f          // odd top row
    addi    a11, a12, -4
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a6, a12, 16         // f8
    loop    a13, 2f
    l32i    a11, a10, 0                     // rows (r, r+1), column 0
    l32i    a12, a10, 0 + COL_BYTES
    l32i    a13, a10, 0 + 2 * COL_BYTES
    mula.da.ll m0, a11
    l16si   a14, a10, 4                     // row r+2, column 0
    mula.da.hh m0, a11
    l16si   a15, a10, 4 + COL_BYTES
    mula.da.ll m1, a12
    l16si   a11, a10, 4 + 2 * COL_BYTES
    mula.da.hh m1, a12
    mula.ad.ll a13, m2
    mula.ad.hh a13, m2
    mula.ad.ll a14, m3
    mula.ad.lh a15, m3
    mula.aa.ll a6, a11
    rsr.acclo a15               // A
    add     a10, a10, a8
    wsr.acclo a7                // bias for the next pixel
    mulsh   a11, a15, a4        // H, 3 after the ACC read: closer stalls
    mov     a12, a5
    movltz  a12, a2, a15        // K - 128 * (A < 0)
    add     a11, a11, a12
    sra     a11, a11            // >> (n + 7)
    clamps  a11, a11, 7
    s8i     a11, a3, 0
    add     a3, a3, a9
2:
    l32i    a11, a1, 8
    l32i    a10, a1, 12
    l32i    a12, a1, 4
    add     a10, a10, a11
    addi    a12, a12, -1
    s32i    a10, a1, 12
    s32i    a12, a1, 4
    bnez    a12, 1b
    retw.n
5:  // odd top row: the pairs are rows (r+1, r+2), the base is row r - 1
    addi    a10, a10, -2
    addi    a11, a12, 48
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a6, a12, 68         // f2
    loop    a13, 4f
    l32i    a11, a10, 4                     // rows (r+1, r+2), column 0
    l32i    a12, a10, 4 + COL_BYTES
    l32i    a13, a10, 4 + 2 * COL_BYTES
    mula.da.ll m0, a11
    l16si   a14, a10, 2                     // row r, column 0
    mula.da.hh m0, a11
    l16si   a15, a10, 2 + COL_BYTES
    mula.da.ll m1, a12
    l16si   a11, a10, 2 + 2 * COL_BYTES
    mula.da.hh m1, a12
    mula.ad.ll a13, m2
    mula.ad.hh a13, m2
    mula.ad.ll a14, m3
    mula.ad.lh a15, m3
    mula.aa.ll a6, a11
    rsr.acclo a15               // A
    add     a10, a10, a8
    wsr.acclo a7                // bias for the next pixel
    mulsh   a11, a15, a4        // H, 3 after the ACC read: closer stalls
    mov     a12, a5
    movltz  a12, a2, a15        // K - 128 * (A < 0)
    add     a11, a11, a12
    sra     a11, a11            // >> (n + 7)
    clamps  a11, a11, 7
    s8i     a11, a3, 0
    add     a3, a3, a9
4:
    l32i    a11, a1, 8
    l32i    a10, a1, 12
    l32i    a12, a1, 4
    add     a10, a10, a11
    addi    a12, a12, -1
    s32i    a10, a1, 12
    s32i    a12, a1, 4
    bnez    a12, 1b
    retw.n

6:  // out_offset -128: no sign term
    sub     a3, a3, a9          // the store pointer moves before each store
    l32i    a2, a6, 72          // stride 1: column groups
    bnez    a2, 13f
7:  // row loop
    l32i    a10, a1, 12
    l32i    a12, a1, 16
    l32i    a13, a1, 0
    wsr.acclo a7
    bbsi    a10, 1, 8f          // odd top row
    addi    a11, a12, -4
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a6, a12, 16         // f8
    l32i    a2, a10, 0
    loop    a13, 9f
    l32i    a12, a10, 0 + COL_BYTES
    l32i    a13, a10, 0 + 2 * COL_BYTES
    mula.da.ll m0, a2
    l16si   a14, a10, 4
    mula.da.hh m0, a2
    l16si   a15, a10, 4 + COL_BYTES
    mula.da.ll m1, a12
    l16si   a11, a10, 4 + 2 * COL_BYTES
    mula.da.hh m1, a12
    mula.ad.ll a13, m2
    mula.ad.hh a13, m2
    mula.ad.ll a14, m3
    mula.ad.lh a15, m3
    mula.aa.ll a6, a11
    rsr.acclo a15               // A
    add     a10, a10, a8
    wsr.acclo a7
    mulsh   a11, a15, a4        // 3 after the ACC read
    add     a3, a3, a9
    l32i    a2, a10, 0                 // next pixel, rows (r, r+1) or (r+1, r+2)
    add     a11, a11, a5        // 3 after the mulsh
    sra     a11, a11
    clamps  a11, a11, 7
    s8i     a11, a3, 0
9:
    l32i    a11, a1, 8
    l32i    a10, a1, 12
    l32i    a12, a1, 4
    add     a10, a10, a11
    addi    a12, a12, -1
    s32i    a10, a1, 12
    s32i    a12, a1, 4
    bnez    a12, 7b
    retw.n
8:  // odd top row
    addi    a10, a10, -2
    addi    a11, a12, 48
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a6, a12, 68         // f2
    l32i    a2, a10, 4
    loop    a13, 10f
    l32i    a12, a10, 4 + COL_BYTES
    l32i    a13, a10, 4 + 2 * COL_BYTES
    mula.da.ll m0, a2
    l16si   a14, a10, 2
    mula.da.hh m0, a2
    l16si   a15, a10, 2 + COL_BYTES
    mula.da.ll m1, a12
    l16si   a11, a10, 2 + 2 * COL_BYTES
    mula.da.hh m1, a12
    mula.ad.ll a13, m2
    mula.ad.hh a13, m2
    mula.ad.ll a14, m3
    mula.ad.lh a15, m3
    mula.aa.ll a6, a11
    rsr.acclo a15               // A
    add     a10, a10, a8
    wsr.acclo a7
    mulsh   a11, a15, a4        // 3 after the ACC read
    add     a3, a3, a9
    l32i    a2, a10, 4                 // next pixel, rows (r, r+1) or (r+1, r+2)
    add     a11, a11, a5        // 3 after the mulsh
    sra     a11, a11
    clamps  a11, a11, 7
    s8i     a11, a3, 0
10:
    l32i    a11, a1, 8
    l32i    a10, a1, 12
    l32i    a12, a1, 4
    add     a10, a10, a11
    addi    a12, a12, -1
    s32i    a10, a1, 12
    s32i    a12, a1, 4
    bnez    a12, 7b
    retw.n

13: // stride 1: two rows per pass, the top row is always even
    l32i    a13, a6, 80
    s32i    a2, a1, 28          // column groups
    s32i    a13, a1, 24         // (out_wd - 1) * out step
    addi    a11, a6, -4
    ldinc   m0, a11
    ldinc   m1, a11
    ldinc   m2, a11
    ldinc   m3, a11
    l32i    a6, a6, 16          // f8
    wsr.acclo a7
    l32i    a10, a1, 12
17: // row pair loop
    l32i    a13, a1, 4          // rows left
    beqi    a13, 1, 7b          // the last row alone
    l32i    a13, a1, 24
    add     a3, a3, a9          // a3: row r, store then move on
    add     a9, a3, a13         // a9: row r + 1, one step behind
    /*
     * Output x uses columns x, x + 1, x + 2. Three steps per pass rotate
     * the column sets (a11, a12), (a13, a14), (a15, a8), so each step
     * loads one new column. No load goes past column out_wd + 1.
     */
    l32i    a11, a10, 0
    l32i    a12, a10, 4
    l32i    a13, a10, COL_BYTES
    l32i    a14, a10, COL_BYTES + 4
    l32i    a2, a1, 28
    loop    a2, 14f
    DW_S1_STEP a11, a12, a13, a14, a15, a8, 2 * COL_BYTES, 0
    DW_S1_STEP a13, a14, a15, a8, a11, a12, 3 * COL_BYTES, 1
    DW_S1_STEP a15, a8, a11, a12, a13, a14, 4 * COL_BYTES, 1
    addi    a10, a10, 3 * COL_BYTES
    DW_S1_LAST
14:
    l32i    a2, a1, 16
    l32i    a2, a2, 76          // out_wd % 3
    loopnez a2, 15f
    DW_S1_STEP a11, a12, a13, a14, a15, a8, 2 * COL_BYTES, 0
    DW_S1_LAST
    mov     a11, a13
    mov     a12, a14
    mov     a13, a15
    mov     a14, a8
    addi    a10, a10, COL_BYTES
15:
    mov     a3, a9              // last output of row r + 1
    movi    a8, COL_BYTES
    l32i    a9, a1, 20
    l32i    a10, a1, 12
    l32i    a12, a1, 4
    addi    a10, a10, 4         // two rows down
    addi    a12, a12, -2
    s32i    a10, a1, 12
    s32i    a12, a1, 4
    bnez    a12, 17b
    retw.n

    .size   esp_nn_dw3x3_dot_rq_esp32, . - esp_nn_dw3x3_dot_rq_esp32
