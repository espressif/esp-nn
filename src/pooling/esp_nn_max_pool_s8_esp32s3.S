/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Max pool leaf for ESP32-S3, called from esp_nn_max_pool_s8_esp32s3.c.
 *
 * void esp_nn_max_pool_s8_esp32s3_run(s3mp_args_t *a, const int8_t *in,
 *                                     int8_t *out, int n, int fw, int fh)
 *
 * Writes n output pixels. Each one is the max over a fw x fh window that
 * starts at `in`, clamped to the activation range. `in` moves by a->in_step
 * per pixel. The caller clips the window, so there are no bounds checks.
 *
 * Channels go in chunks: 32 lanes (two accumulators, so the loads of one
 * hide the load-use stall of the other), then 16, then 8, when
 * channels % 8 == 0. Otherwise pairs of 4-lane ee.vldbc.32 chunks, then one.
 * These loads, and the aligned 128-bit ones, read window bytes only. The
 * funnel-shift loads of .Lk2c4x16 read whole aligned blocks (see there).
 *
 * A full 2x2 window takes a fast path with no window loops: four loads and
 * three vmax per chunk. 16 lanes when channels % 16 == 0, 4 lanes when
 * channels % 8 == 4. A full 3x3 window does the same with nine loads per
 * chunk, 16 or 4 lanes. With channels % 8 == 4 and channels >= 16, both take
 * 16 lanes from two aligned blocks (see .Lk2c4x16).
 *
 * Other windows use 128-bit loads when channels % 16 == 0 and `in` is
 * 16-byte aligned, so every pixel is. Chunks are 64, then 32, then 16 lanes.
 * Else the 64/32-bit chunk loops.
 *
 * esp_nn_max_pool_s8_esp32s3_rows() takes the same arguments and runs
 * esp_nn_max_pool_s8_esp32s3_run() on a->rows rows. `in` moves by
 * a->in_row_step per row and `out` by a->out_row_step. One band makes
 * several calls with the same a->rows, so each call counts a copy down.
 *
 * The s3mp_args_t field offsets are in esp_nn_max_pool_s8_esp32s3.h.
 *
 * Pure .S rather than inline asm: _rows makes a windowed call8, and the
 * leaf uses all of a2-a15 and the zero-overhead loop registers.
 */

#include "esp_nn_max_pool_s8_esp32s3.h"

    .text
    .align  4
    .global esp_nn_max_pool_s8_esp32s3_run
    .type   esp_nn_max_pool_s8_esp32s3_run, @function
esp_nn_max_pool_s8_esp32s3_run:
    entry   a1, 32
    l32i    a8, a2, S3MP_CHANNELS
    mull    a9, a6, a8
    l32i    a15, a2, S3MP_ROW_BYTES
    sub     a9, a15, a9             // row_skip = row_bytes - fw * ch
    addi    a10, a8, -8
    addi    a15, a2, S3MP_ACT_MIN
    ee.vldbc.8.ip q6, a15, 1        // act_min in every lane
    ee.vldbc.8 q7, a15              // act_max in every lane
    bne     a6, a7, .Lgen
    bnei    a6, 2, .L3chk
    l32i    a14, a2, S3MP_N16_FULL
    bnez    a14, .Lk2c16
    l32i    a14, a2, S3MP_N4
    bnez    a14, .Lk2c4x16
.L3chk:                             // a 2x2 window gets here only for .Lgen
    bnei    a6, 3, .Lgen
    l32i    a14, a2, S3MP_N4
    srli    a15, a14, 2
    bnez    a15, .Lk3c4x16              // 16-lane chunks fit
    bnez    a14, .Lk3c4
    l32i    a14, a2, S3MP_N16_FULL
    bnez    a14, .Lk3c16

.Lgen:
    l32i    a14, a2, S3MP_N16_FULL
    extui   a15, a3, 0, 4
    bnez    a14, .Lpixa
.Lpix:
    mov     a11, a3                 // chunk start in the first window pixel
    l32i    a14, a2, S3MP_N32
    beqz    a14, .Lc16s
    addi    a15, a8, -24
.Lc32:
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6              // starting at act_min applies the lower clamp
    ee.orq  q3, q6, q6
.Lr32:
    loopnez a6, .Lr32e
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.ip q1, a12, 8
    ee.vld.l.64.ip q2, a12, 8
    ee.vld.h.64.xp q2, a12, a15
    ee.vmax.s8 q0, q0, q1
    ee.vmax.s8 q3, q3, q2
.Lr32e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lr32
    ee.vmin.s8 q0, q0, q7
    ee.vmin.s8 q3, q3, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
    ee.vst.l.64.ip q3, a4, 8
    ee.vst.h.64.ip q3, a4, 8
    addi    a11, a11, 32
    addi    a14, a14, -1
    bnez    a14, .Lc32

.Lc16s:
    l32i    a14, a2, S3MP_N16
    beqz    a14, .Lc8
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
.Lr16:
    loopnez a6, .Lr16e
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a10
    ee.vmax.s8 q0, q0, q1
.Lr16e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lr16
    ee.vmin.s8 q0, q0, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
    addi    a11, a11, 16

.Lc8:
    l32i    a14, a2, S3MP_N8
    beqz    a14, .Lc4
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
.Lr8:
    loopnez a6, .Lr8e
    ee.vld.l.64.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q1
.Lr8e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lr8
    ee.vmin.s8 q0, q0, q7
    ee.vst.l.64.ip q0, a4, 8
    addi    a11, a11, 8

.Lc4:
    l32i    a14, a2, S3MP_N4
    beqz    a14, .Lnext
    addi    a15, a8, -4
.Lc4p:                              // two 4-lane chunks per pass, as in .Lc32
    bltui   a14, 2, .Lc4l
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
    ee.orq  q3, q6, q6
.Lr4p:
    loopnez a6, .Lr4pe
    ee.vldbc.32.ip q1, a12, 4
    ee.vldbc.32.xp q2, a12, a15
    ee.vmax.s8 q0, q0, q1
    ee.vmax.s8 q3, q3, q2
.Lr4pe:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lr4p
    ee.vmin.s8 q0, q0, q7
    ee.vmin.s8 q3, q3, q7
    ee.movi.32.a q0, a12, 0
    s32i    a12, a4, 0
    ee.movi.32.a q3, a12, 0
    s32i    a12, a4, 4
    addi    a4, a4, 8
    addi    a11, a11, 8
    addi    a14, a14, -2
    j       .Lc4p

.Lc4l:                              // n4 is odd, so one chunk is left
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
.Lr4:
    loopnez a6, .Lr4e
    ee.vldbc.32.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q1
.Lr4e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lr4
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a15, 0
    s32i    a15, a4, 0
    addi    a4, a4, 4
    addi    a11, a11, 4

.Lnext:
    l32i    a15, a2, S3MP_IN_STEP
    add     a3, a3, a15
    addi    a5, a5, -1
    bnez    a5, .Lpix
    retw

.Lk2c16:                            // 2x2 window, a14 = channels / 16
    l32i    a9, a2, S3MP_ROW_BYTES
    l32i    a11, a2, S3MP_IN_STEP
    or      a15, a3, a4
    extui   a15, a15, 0, 4
    beqz    a15, .Lk2c16_q              // every pixel is 16-byte aligned
    xor     a15, a3, a4
    extui   a15, a15, 0, 4
    beqz    a15, .Lk2c16_off8              // in and out both at 8 mod 16
    movi    a15, 8
    sub     a15, a15, a8            // 8 - ch: from the right pixel back to the next chunk
.Lk2c16_pix:
    mov     a12, a3                 // top row
    add     a13, a3, a9             // bottom row
    loopnez a14, .Lk2c16_end
    ee.vld.l.64.ip q0, a12, 8
    ee.vld.h.64.xp q0, a12, a10
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a15
    ee.vld.l.64.ip q2, a13, 8
    ee.vld.h.64.xp q2, a13, a10
    ee.vmax.s8 q0, q0, q1
    ee.vld.l.64.ip q3, a13, 8
    ee.vld.h.64.xp q3, a13, a15
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q6
    ee.vmax.s8 q0, q0, q3
    ee.vmin.s8 q0, q0, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
.Lk2c16_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk2c16_pix
    retw

.Lk2c16_q:                          // .Lk2c16 with 128-bit loads, one pointer per window pixel
    mov     a12, a3                 // top left
    add     a13, a3, a8             // top right
    add     a6, a3, a9              // bottom left
    add     a7, a6, a8              // bottom right
    loopnez a14, .Lk2c16_q_end
    ee.vld.128.ip q0, a12, 16
    ee.vld.128.ip q1, a13, 16
    ee.vmax.s8.ld.incp q2, a6, q0, q0, q6
    ee.vmax.s8.ld.incp q3, a7, q0, q0, q1
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q3
    ee.vmin.s8 q0, q0, q7
    ee.vst.128.ip q0, a4, 16
.Lk2c16_q_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk2c16_q
    retw

.Lk2c4:                             // 2x2 window, a14 = channels / 4
    l32i    a9, a2, S3MP_ROW_BYTES
    l32i    a11, a2, S3MP_IN_STEP
    movi    a6, 4
    sub     a6, a6, a8              // 4 - ch: from the right pixel back to the next chunk
.Lk2c4_pix:
    mov     a12, a3                 // top row
    add     a13, a3, a9             // bottom row
    loopnez a14, .Lk2c4_end
    ee.vldbc.32.xp q0, a12, a8
    ee.vldbc.32.xp q1, a12, a6
    ee.vldbc.32.xp q2, a13, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q3, a13, a6
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q6
    ee.vmax.s8 q0, q0, q3
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a7, 0
    s32i    a7, a4, 0
    addi    a4, a4, 4
.Lk2c4_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk2c4_pix
    retw

.Lk3c4:                             // 3x3 window, a14 = channels / 4
    l32i    a11, a2, S3MP_IN_STEP
    l32i    a10, a2, S3MP_ROW_BYTES
    slli    a9, a8, 1
    sub     a9, a10, a9             // row_bytes - 2 ch: right pixel to the next row
    add     a10, a10, a8
    slli    a10, a10, 1
    addi    a6, a10, -4
    neg     a6, a6                  // 4 - 2 (row_bytes + ch): bottom right back to the next chunk
.Lk3c4_pix:
    mov     a12, a3
    loopnez a14, .Lk3c4_end
    ee.vldbc.32.xp q1, a12, a8
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q6, q1
    ee.vldbc.32.xp q1, a12, a9
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a9
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a6
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q1
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a7, 0
    s32i    a7, a4, 0
    addi    a4, a4, 4
.Lk3c4_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk3c4_pix
    retw
.Lk3c16:                            // 3x3 window, a14 = channels / 16
    l32i    a11, a2, S3MP_IN_STEP
    or      a15, a3, a4
    extui   a15, a15, 0, 4
    beqz    a15, .Lk3c16_q             // every pixel is 16-byte aligned
    l32i    a6, a2, S3MP_ROW_BYTES
    slli    a9, a8, 1
    sub     a9, a6, a9
    addi    a9, a9, -8              // row_bytes - 2 ch - 8: right pixel to the next row
    add     a6, a6, a8
    slli    a6, a6, 1
    neg     a6, a6
    addi    a6, a6, 8               // 8 - 2 (row_bytes + ch): bottom right back to the next chunk
.Lk3c16_pix:
    mov     a12, a3
    loopnez a14, .Lk3c16_end
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a10
    ee.vld.l.64.ip q2, a12, 8
    ee.vld.h.64.xp q2, a12, a10
    ee.vmax.s8 q0, q6, q1
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a9
    ee.vmax.s8 q0, q0, q2
    ee.vld.l.64.ip q2, a12, 8
    ee.vld.h.64.xp q2, a12, a10
    ee.vmax.s8 q0, q0, q1
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a10
    ee.vmax.s8 q0, q0, q2
    ee.vld.l.64.ip q2, a12, 8
    ee.vld.h.64.xp q2, a12, a9
    ee.vmax.s8 q0, q0, q1
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a10
    ee.vmax.s8 q0, q0, q2
    ee.vld.l.64.ip q2, a12, 8
    ee.vld.h.64.xp q2, a12, a10
    ee.vmax.s8 q0, q0, q1
    ee.vld.l.64.ip q1, a12, 8
    ee.vld.h.64.xp q1, a12, a6
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q1
    ee.vmin.s8 q0, q0, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
.Lk3c16_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk3c16_pix
    retw
.Lk3c16_q:                          // .Lk3c16 with 128-bit loads
    l32i    a6, a2, S3MP_ROW_BYTES
    slli    a9, a8, 1
    sub     a9, a6, a9              // row_bytes - 2 ch: right pixel to the next row
    add     a6, a6, a8
    slli    a6, a6, 1
    neg     a6, a6
    addi    a6, a6, 16              // 16 - 2 (row_bytes + ch): bottom right back to the next chunk
.Lk3c16_q_pix:
    mov     a12, a3
    loopnez a14, .Lk3c16_q_end
    ee.vld.128.xp q1, a12, a8
    ee.vld.128.xp q2, a12, a8
    ee.vmax.s8 q0, q6, q1
    ee.vld.128.xp q1, a12, a9
    ee.vmax.s8 q0, q0, q2
    ee.vld.128.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vld.128.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vld.128.xp q2, a12, a9
    ee.vmax.s8 q0, q0, q1
    ee.vld.128.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vld.128.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vld.128.xp q1, a12, a6
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q1
    ee.vmin.s8 q0, q0, q7
    ee.vst.128.ip q0, a4, 16
.Lk3c16_q_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk3c16_q_pix
    retw
.Lpixa:                             // the chunk loops with 128-bit loads
    bnez    a15, .Lpix              // `in` is not 16-byte aligned
    addi    a15, a8, -16
    addi    a10, a8, -48
.Lpixa1:
    mov     a11, a3
    l32i    a14, a2, S3MP_N32
.La64:                              // 64 lanes, four accumulators
    bltui   a14, 2, .La32
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
    ee.orq  q1, q6, q6
    ee.orq  q2, q6, q6
    ee.orq  q3, q6, q6
    ee.orq  q4, q6, q6
    ee.orq  q5, q6, q6
.Lra64:                             // q1/q2 carry the last two loads of the previous pixel
    loopnez a6, .Lra64e
    // The vmax reads q1 before the load writes it: qy is read in stage 1
    // and qu written in stage 2 (S3 TRM, Table 1.7-2)
    ee.vmax.s8.ld.incp q1, a12, q4, q4, q1
    ee.vmax.s8.ld.incp q2, a12, q5, q5, q2
    ee.vmax.s8.ld.incp q1, a12, q0, q0, q1
    ee.vmax.s8 q3, q3, q2
    ee.vld.128.xp q2, a12, a10
.Lra64e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lra64
    ee.vmax.s8 q4, q4, q1
    ee.vmax.s8 q5, q5, q2
    ee.vmin.s8 q0, q0, q7
    ee.vmin.s8 q3, q3, q7
    ee.vmin.s8 q4, q4, q7
    ee.vmin.s8 q5, q5, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
    ee.vst.l.64.ip q3, a4, 8
    ee.vst.h.64.ip q3, a4, 8
    ee.vst.l.64.ip q4, a4, 8
    ee.vst.h.64.ip q4, a4, 8
    ee.vst.l.64.ip q5, a4, 8
    ee.vst.h.64.ip q5, a4, 8
    addi    a11, a11, 64
    addi    a14, a14, -2
    j       .La64
.La32:                              // one 32-lane chunk left at most
    beqz    a14, .La16
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
    ee.orq  q1, q6, q6
    ee.orq  q2, q6, q6
    ee.orq  q3, q6, q6
.Lra32:                             // each vmax takes the load of the previous pixel
    loopnez a6, .Lra32e
    ee.vmax.s8.ld.incp q1, a12, q0, q0, q1
    ee.vmax.s8 q3, q3, q2
    ee.vld.128.xp q2, a12, a15
.Lra32e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lra32
    ee.vmax.s8 q0, q0, q1
    ee.vmax.s8 q3, q3, q2
    ee.vmin.s8 q0, q0, q7
    ee.vmin.s8 q3, q3, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
    ee.vst.l.64.ip q3, a4, 8
    ee.vst.h.64.ip q3, a4, 8
    addi    a11, a11, 32
.La16:
    l32i    a14, a2, S3MP_N16
    beqz    a14, .Lanext
    mov     a12, a11
    mov     a13, a7
    ee.orq  q0, q6, q6
.Lra16:
    loopnez a6, .Lra16e
    ee.vld.128.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q1
.Lra16e:
    add     a12, a12, a9
    addi    a13, a13, -1
    bnez    a13, .Lra16
    ee.vmin.s8 q0, q0, q7
    ee.vst.l.64.ip q0, a4, 8
    ee.vst.h.64.ip q0, a4, 8
.Lanext:
    l32i    a14, a2, S3MP_IN_STEP
    add     a3, a3, a14
    addi    a5, a5, -1
    bnez    a5, .Lpixa1
    retw
/*
 * .Lk2c16_q for in and out at 8 mod 16. The first and last 8 bytes of each pixel
 * go in one register, the aligned blocks between them take 128-bit loads.
 */
.Lk2c16_off8:
    movi    a15, 16
    sub     a15, a15, a8            // 16 - ch: from the tail back to the first block
    addi    a14, a14, -1            // aligned blocks per pixel
.Lk2c16_off8_pix:
    mov     a12, a3                 // top left
    add     a13, a3, a8             // top right
    add     a6, a3, a9              // bottom left
    add     a7, a6, a8              // bottom right
    ee.vld.l.64.xp q0, a12, a10
    ee.vld.h.64.xp q0, a12, a15
    ee.vld.l.64.xp q1, a13, a10
    ee.vld.h.64.xp q1, a13, a15
    ee.vld.l.64.xp q2, a6, a10
    ee.vld.h.64.xp q2, a6, a15
    ee.vmax.s8 q5, q0, q1
    ee.vld.l.64.xp q3, a7, a10
    ee.vld.h.64.xp q3, a7, a15
    ee.vmax.s8 q5, q5, q2
    ee.vmax.s8 q5, q5, q6
    ee.vmax.s8 q5, q5, q3
    ee.vmin.s8 q5, q5, q7
    ee.vst.l.64.ip q5, a4, 8
    loopnez a14, .Lk2c16_off8_end
    ee.vld.128.ip q0, a12, 16
    ee.vld.128.ip q1, a13, 16
    ee.vmax.s8.ld.incp q2, a6, q0, q0, q6
    ee.vmax.s8.ld.incp q3, a7, q0, q0, q1
    ee.vmax.s8 q0, q0, q2
    ee.vmax.s8 q0, q0, q3
    ee.vmin.s8 q0, q0, q7
    ee.vst.128.ip q0, a4, 16
.Lk2c16_off8_end:
    ee.vst.h.64.ip q5, a4, 8
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk2c16_off8_pix
    retw
/*
 * .Lk2c4 and .Lk3c4 with 16 lanes, for channels % 8 == 4 and channels >= 16.
 * Each window pixel is loaded from the two aligned blocks holding its bytes
 * p and p + 15, then funnel-shifted. Both blocks hold window bytes, so no
 * load touches a 16-byte block outside the input. A block can still reach
 * up to 15 bytes past either end of the input. `out` can be at 4 mod 8, so the stores
 * are 32-bit. The 4 or 12 lanes left take 4-lane chunks. The channels < 16
 * check for 2x2 windows sits here, at the entry of this path, so the more
 * common paths dispatch with one branch.
 */
.Lk2c4x16:
    bltui   a14, 4, .Lk2c4
    l32i    a9, a2, S3MP_ROW_BYTES
    l32i    a11, a2, S3MP_IN_STEP
    srli    a15, a8, 4              // 16-lane chunks
    extui   a10, a14, 0, 2          // 4-lane chunks left
    movi    a2, 15
    addi    a6, a8, -15             // to the right pixel's last block
    movi    a7, 1
    sub     a7, a7, a8              // 1 - ch: from the right pixel to the next chunk
.Lk2c4x16_pix:
    mov     a12, a3                 // top row
    add     a13, a3, a9             // bottom row
    loopnez a15, .Lk2c4x16_end
    ee.ld.128.usar.xp q0, a12, a2
    ee.vld.128.xp q1, a12, a6
    ee.src.q q0, q0, q1
    ee.ld.128.usar.xp q2, a12, a2
    ee.vld.128.xp q3, a12, a7
    ee.vmax.s8 q0, q0, q6
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q1, a13, a2
    ee.vld.128.xp q3, a13, a6
    ee.vmax.s8 q0, q0, q2
    ee.src.q q1, q1, q3
    ee.ld.128.usar.xp q2, a13, a2
    ee.vld.128.xp q3, a13, a7
    ee.vmax.s8 q0, q0, q1
    ee.src.q q2, q2, q3
    ee.vmax.s8 q0, q0, q2
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a14, 0
    s32i    a14, a4, 0
    ee.movi.32.a q0, a14, 1
    s32i    a14, a4, 4
    ee.movi.32.a q0, a14, 2
    s32i    a14, a4, 8
    ee.movi.32.a q0, a14, 3
    s32i    a14, a4, 12
    addi    a4, a4, 16
.Lk2c4x16_end:
    loopnez a10, .Lk2c4x16_tail_end            // as .Lk2c4, the steps end 3 bytes short
    ee.vldbc.32.xp q0, a12, a8
    ee.vldbc.32.xp q1, a12, a7
    ee.vldbc.32.xp q2, a13, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q3, a13, a7
    ee.vmax.s8 q0, q0, q2
    addi    a12, a12, 3
    ee.vmax.s8 q0, q0, q6
    addi    a13, a13, 3
    ee.vmax.s8 q0, q0, q3
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a14, 0
    s32i    a14, a4, 0
    addi    a4, a4, 4
.Lk2c4x16_tail_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk2c4x16_pix
    retw

.Lk3c4x16:
    l32i    a11, a2, S3MP_IN_STEP
    l32i    a9, a2, S3MP_ROW_BYTES
    srli    a15, a8, 4
    extui   a10, a14, 0, 2
    slli    a13, a8, 1
    sub     a13, a9, a13            // row_bytes - 2 ch: right pixel to the next row
    addi    a7, a13, -15
    add     a9, a9, a8
    slli    a9, a9, 1
    movi    a2, 1
    sub     a9, a2, a9              // 1 - 2 (row_bytes + ch): bottom right to the next chunk
    movi    a2, 15
    addi    a6, a8, -15
.Lk3c4x16_pix:
    mov     a12, a3
    loopnez a15, .Lk3c4x16_end
    ee.ld.128.usar.xp q1, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.src.q q1, q1, q3
    ee.ld.128.usar.xp q2, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.vmax.s8 q0, q6, q1
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q1, a12, a2
    ee.vld.128.xp q3, a12, a7
    ee.vmax.s8 q0, q0, q2
    ee.src.q q1, q1, q3
    ee.ld.128.usar.xp q2, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.vmax.s8 q0, q0, q1
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q1, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.vmax.s8 q0, q0, q2
    ee.src.q q1, q1, q3
    ee.ld.128.usar.xp q2, a12, a2
    ee.vld.128.xp q3, a12, a7
    ee.vmax.s8 q0, q0, q1
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q1, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.vmax.s8 q0, q0, q2
    ee.src.q q1, q1, q3
    ee.ld.128.usar.xp q2, a12, a2
    ee.vld.128.xp q3, a12, a6
    ee.vmax.s8 q0, q0, q1
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q1, a12, a2
    ee.vld.128.xp q3, a12, a9
    ee.vmax.s8 q0, q0, q2
    ee.src.q q1, q1, q3
    ee.vmax.s8 q0, q0, q1
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a14, 0
    s32i    a14, a4, 0
    ee.movi.32.a q0, a14, 1
    s32i    a14, a4, 4
    ee.movi.32.a q0, a14, 2
    s32i    a14, a4, 8
    ee.movi.32.a q0, a14, 3
    s32i    a14, a4, 12
    addi    a4, a4, 16
.Lk3c4x16_end:
    loopnez a10, .Lk3c4x16_tail_end            // as .Lk3c4, the last step ends 3 bytes short
    ee.vldbc.32.xp q1, a12, a8
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q6, q1
    ee.vldbc.32.xp q1, a12, a13
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a13
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a8
    ee.vmax.s8 q0, q0, q2
    ee.vldbc.32.xp q2, a12, a8
    ee.vmax.s8 q0, q0, q1
    ee.vldbc.32.xp q1, a12, a9
    ee.vmax.s8 q0, q0, q2
    addi    a12, a12, 3
    ee.vmax.s8 q0, q0, q1
    ee.vmin.s8 q0, q0, q7
    ee.movi.32.a q0, a14, 0
    s32i    a14, a4, 0
    addi    a4, a4, 4
.Lk3c4x16_tail_end:
    add     a3, a3, a11
    addi    a5, a5, -1
    bnez    a5, .Lk3c4x16_pix
    retw
    .size   esp_nn_max_pool_s8_esp32s3_run, . - esp_nn_max_pool_s8_esp32s3_run

    .align  4
    .global esp_nn_max_pool_s8_esp32s3_rows
    .type   esp_nn_max_pool_s8_esp32s3_rows, @function
esp_nn_max_pool_s8_esp32s3_rows:
    entry   a1, 32
    l32i    a8, a2, S3MP_ROWS
    s32i    a8, a2, S3MP_ROWS_LEFT          // call8 clobbers a8..a15, so count in memory
.Lrows:
    mov     a10, a2
    mov     a11, a3
    mov     a12, a4
    mov     a13, a5
    mov     a14, a6
    mov     a15, a7
    call8   esp_nn_max_pool_s8_esp32s3_run
    l32i    a8, a2, S3MP_IN_ROW_STEP
    add     a3, a3, a8
    l32i    a8, a2, S3MP_OUT_ROW_STEP
    add     a4, a4, a8
    l32i    a8, a2, S3MP_ROWS_LEFT
    addi    a8, a8, -1
    s32i    a8, a2, S3MP_ROWS_LEFT
    bnez    a8, .Lrows
    retw
    .size   esp_nn_max_pool_s8_esp32s3_rows, . - esp_nn_max_pool_s8_esp32s3_rows

    .macro pool_sliding_store value
    ee.vmax.s8    q0,\value,q6
    ee.vmin.s8    q0,q0,q7
    ee.vst.128.xp q0,a3,a5
    .endm

    // q2..q5 hold running maxima over the last one through four columns.
    // q1 holds the new column's vertical maximum. q6/q7 are activation bounds.
    .macro pool_sliding_row pad, history
    movi.n  a8,-\pad
1:
    mov.n   a9,a2
    ee.vld.128.xp q1,a9,a6
    loopgtz a7,2f
    ee.vld.128.xp q0,a9,a6
    ee.vmax.s8 q1,q1,q0
2:
    add.n   a2,a2,a5
    bltz    a8,3f
    ee.vmax.s8 q0,\history,q1
    pool_sliding_store q0
3:
    .if \pad == 2
    ee.vmax.s8 q5,q4,q1
    ee.vmax.s8 q4,q3,q1
    .endif
    ee.vmax.s8 q3,q2,q1
    mv.qr   q2,q1
    addi.n  a8,a8,1
    addi.n  a4,a4,-1
    bnez    a4,1b

    // The last \pad windows have no new input columns on the right.
    pool_sliding_store \history
    .if \pad == 2
    pool_sliding_store q4
    .endif
    retw.n
    .endm

    .align 4
    .global esp_nn_max_pool_s8_esp32s3_sliding_row_asm
    .hidden esp_nn_max_pool_s8_esp32s3_sliding_row_asm
    .type   esp_nn_max_pool_s8_esp32s3_sliding_row_asm, @function
esp_nn_max_pool_s8_esp32s3_sliding_row_asm:
    // One aligned 16-channel group across a row. The caller clips vertical
    // bounds, so a7 is the positive count of input rows in every column.
    entry   a1,64
    addi.n  a7,a7,-1
    movi    a8,-128
    s32i.n  a8,a1,0
    ee.vldbc.8 q2,a1
    mv.qr   q3,q2
    addi    a9,a1,68
    addi    a10,a1,72
    ee.vldbc.8 q6,a9
    ee.vldbc.8 q7,a10
    l32i.n  a8,a1,64
    beqi    a8,1,.Lpool_sliding_3x3
    mv.qr   q4,q2
    mv.qr   q5,q2
    pool_sliding_row 2,q5
.Lpool_sliding_3x3:
    pool_sliding_row 1,q3
    .size   esp_nn_max_pool_s8_esp32s3_sliding_row_asm, . - esp_nn_max_pool_s8_esp32s3_sliding_row_asm

    // Full INT8 range: q6/q7 are free for independent loads. The caller
    // clips the vertical bounds, so specialize for 2..5 valid input rows.
    .macro pool_sliding_vertical_full count
    mov.n   a9,a2
    ee.vld.128.xp q1,a9,a6
    ee.vld.128.xp q0,a9,a6
    .if \count >= 3
    ee.vld.128.xp q6,a9,a6
    .endif
    ee.vmax.s8 q1,q1,q0
    .if \count >= 4
    ee.vld.128.xp q7,a9,a6
    .endif
    .if \count == 5
    ee.vld.128.xp q0,a9,a6
    .endif
    add.n   a2,a2,a5
    .if \count >= 3
    ee.vmax.s8 q1,q1,q6
    .endif
    .if \count == 5
    ee.vmax.s8 q7,q7,q0
    .endif
    .if \count >= 4
    ee.vmax.s8 q1,q1,q7
    .endif
    .endm

    // Same running maxima as pool_sliding_row, without output clamps.
    .macro pool_sliding_row_full pad, history, count
    movi.n  a8,-\pad
1:
    pool_sliding_vertical_full \count
    bltz    a8,2f
    ee.vmax.s8 q0,\history,q1
    ee.vst.128.xp q0,a3,a5
2:
    .if \pad == 2
    ee.vmax.s8 q5,q4,q1
    ee.vmax.s8 q4,q3,q1
    .endif
    ee.vmax.s8 q3,q2,q1
    mv.qr   q2,q1
    addi.n  a8,a8,1
    addi.n  a4,a4,-1
    bnez    a4,1b
    ee.vst.128.xp \history,a3,a5
    .if \pad == 2
    ee.vst.128.xp q4,a3,a5
    .endif
    retw.n
    .endm

    .align 4
    .global esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm
    .hidden esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm
    .type   esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm, @function
esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm:
    // Same call ABI as the clamped leaf. The activation arguments are unused.
    // Keep the sentinel below the register-window spill area (frame - 48).
    entry   a1,64
    movi    a8,-128
    s32i.n  a8,a1,0
    ee.vldbc.8 q2,a1
    mv.qr   q3,q2
    l32i.n  a8,a1,64
    beqi    a8,1,.Lpool_sliding_full_3x3
    mv.qr   q4,q2
    mv.qr   q5,q2
    beqi    a7,5,.Lpool_sliding_full_5rows
    beqi    a7,4,.Lpool_sliding_full_4rows
    pool_sliding_row_full 2,q5,3
.Lpool_sliding_full_4rows:
    pool_sliding_row_full 2,q5,4
.Lpool_sliding_full_5rows:
    pool_sliding_row_full 2,q5,5
.Lpool_sliding_full_3x3:
    beqi    a7,3,.Lpool_sliding_full_3rows
    pool_sliding_row_full 1,q3,2
.Lpool_sliding_full_3rows:
    pool_sliding_row_full 1,q3,3
    .size   esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm, . - esp_nn_max_pool_s8_esp32s3_sliding_row_full_asm
