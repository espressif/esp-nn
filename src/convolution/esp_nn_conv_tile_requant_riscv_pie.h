/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <common_functions.h>

/*
 * Transposes 8 rows of 16 bytes in q0-q7: byte pairs (rows 2k, 2k + 1),
 * then quads, then all 8 rows. Columns 0-1, 2-3, ... 14-15 end up in
 * q0 q4 q2 q6 q1 q5 q3 q7, 8 bytes per column.
 */
#define ESP_NN_PIE_ZIP8X16 \
        "esp.vzip.8  q0, q1                  \n\t" \
        "esp.vzip.8  q2, q3                  \n\t" \
        "esp.vzip.8  q4, q5                  \n\t" \
        "esp.vzip.8  q6, q7                  \n\t" \
        "esp.vzip.16 q0, q2                  \n\t" \
        "esp.vzip.16 q1, q3                  \n\t" \
        "esp.vzip.16 q4, q6                  \n\t" \
        "esp.vzip.16 q5, q7                  \n\t" \
        "esp.vzip.32 q0, q4                  \n\t" \
        "esp.vzip.32 q2, q6                  \n\t" \
        "esp.vzip.32 q1, q5                  \n\t" \
        "esp.vzip.32 q3, q7                  \n\t"

#define TILE_RQ_LOAD(r0, r1, r2, r3, l, LS) \
        "lw     %[" #r0 "], (" #l "+0)*" #LS "(%[src]) \n\t" \
        "lw     %[" #r1 "], (" #l "+1)*" #LS "(%[src]) \n\t" \
        "lw     %[" #r2 "], (" #l "+2)*" #LS "(%[src]) \n\t" \
        "lw     %[" #r3 "], (" #l "+3)*" #LS "(%[src]) \n\t"
#define TILE_RQ_MUL(r0, r1, r2, r3) \
        "esp.addx2 %[" #r0 "], %[c2], %[" #r0 "] \n\t" \
        "esp.addx2 %[" #r1 "], %[c2], %[" #r1 "] \n\t" \
        "esp.addx2 %[" #r2 "], %[c2], %[" #r2 "] \n\t" \
        "esp.addx2 %[" #r3 "], %[c2], %[" #r3 "] \n\t" \
        "mulhsu %[" #r0 "], %[" #r0 "], %[m2] \n\t" \
        "mulhsu %[" #r1 "], %[" #r1 "], %[m2] \n\t" \
        "mulhsu %[" #r2 "], %[" #r2 "], %[m2] \n\t" \
        "mulhsu %[" #r3 "], %[" #r3 "], %[m2] \n\t"
#define TILE_RQ_STORE(r0, r1, r2, r3, l) \
        "sw     %[" #r0 "], (" #l "+0)*4(%[res]) \n\t" \
        "sw     %[" #r1 "], (" #l "+1)*4(%[res]) \n\t" \
        "sw     %[" #r2 "], (" #l "+2)*4(%[res]) \n\t" \
        "sw     %[" #r3 "], (" #l "+3)*4(%[res]) \n\t"
/* Scalar part of the tile requant: res[l] = mulhsu(2 comb + 2 src[l], 2 mult)
 * for 16 lanes, lane l at byte offset l * LS of src. Two groups of four
 * lanes are in flight, to cover the mulhsu latency. */
#define TILE_RQ_SCALAR(src_, res_, c2_, m2_, LS) do { \
    int32_t a0, a1, a2, a3, b0, b1, b2, b3; \
    asm volatile ( \
        TILE_RQ_LOAD(a0, a1, a2, a3, 0, LS) \
        TILE_RQ_MUL(a0, a1, a2, a3) \
        TILE_RQ_LOAD(b0, b1, b2, b3, 4, LS) \
        TILE_RQ_STORE(a0, a1, a2, a3, 0) \
        TILE_RQ_MUL(b0, b1, b2, b3) \
        TILE_RQ_LOAD(a0, a1, a2, a3, 8, LS) \
        TILE_RQ_STORE(b0, b1, b2, b3, 4) \
        TILE_RQ_MUL(a0, a1, a2, a3) \
        TILE_RQ_LOAD(b0, b1, b2, b3, 12, LS) \
        TILE_RQ_STORE(a0, a1, a2, a3, 8) \
        TILE_RQ_MUL(b0, b1, b2, b3) \
        TILE_RQ_STORE(b0, b1, b2, b3, 12) \
        : [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3), \
          [b0] "=&r"(b0), [b1] "=&r"(b1), [b2] "=&r"(b2), [b3] "=&r"(b3) \
        : [src] "r"(src_), [res] "r"(res_), [c2] "r"(c2_), [m2] "r"(m2_) \
        : "memory"); \
    } while (0)

/* PIE part: a = (a + 2 * (a < 0 ? -1 : 0) + rnd) >> e1, clamped, for two
 * quarters at once so that no PIE result is used by the next instruction.
 * q4 = 0, q5 = rnd. */
#define TILE_RQ_VEC2(qa, qb) \
        "esp.vld.128.ip  " #qa ", x30, 16     \n\t" \
        "esp.vld.128.ip  " #qb ", x30, 16     \n\t" \
        "esp.vcmp.lt.s32 q6, " #qa ", q4       \n\t" \
        "esp.vcmp.lt.s32 q7, " #qb ", q4       \n\t" \
        "esp.vadd.s32    " #qa ", " #qa ", q6    \n\t" \
        "esp.vadd.s32    " #qb ", " #qb ", q7    \n\t" \
        "esp.vadd.s32    " #qa ", " #qa ", q6    \n\t" \
        "esp.vadd.s32    " #qb ", " #qb ", q7    \n\t" \
        "esp.vadd.s32    " #qa ", " #qa ", q5    \n\t" \
        "esp.vadd.s32    " #qb ", " #qb ", q5    \n\t" \
        "esp.vsr.s32     " #qa ", " #qa "        \n\t" \
        "esp.vsr.s32     " #qb ", " #qb "        \n\t" \
        "esp.vsat.s32    " #qa ", " #qa ", x28, x29 \n\t" \
        "esp.vsat.s32    " #qb ", " #qb ", x28, x29 \n\t"
static inline __attribute__((always_inline))
void esp_nn_conv_requant_tile_vec(const int32_t *res, const int32_t *rnd,
                                  int32_t e1, int32_t act_min,
                                  int32_t act_max, int8_t *row)
{
    asm volatile (
        "mv     x30, %[res]                  \n\t"
        "mv     x28, %[lo]                   \n\t"
        "mv     x29, %[hi]                   \n\t"
        "mv     x31, %[e1]                   \n\t"
        "esp.movx.w.sar x31                  \n\t"
        "mv     x31, %[rnd]                  \n\t"
        "esp.vldbc.32.ip q5, x31, 0          \n\t"
        "esp.zero.q      q4                  \n\t"
        TILE_RQ_VEC2(q0, q1)
        TILE_RQ_VEC2(q2, q3)
        /* Low byte of each 32-bit lane, lanes in order. */
        "esp.vunzip.16   q0, q1              \n\t"
        "esp.vunzip.16   q2, q3              \n\t"
        "esp.vunzip.8    q0, q2              \n\t"
        "mv     x31, %[row]                  \n\t"
        "esp.vst.128.ip  q0, x31, 0          \n\t"
        :
        : [res] "r"(res), [e1] "r"(e1), [rnd] "r"(rnd),
          [lo] "r"(act_min), [hi] "r"(act_max), [row] "r"(row)
        : "x28", "x29", "x30", "x31", "memory"
    );
}

/*
 * Requantize the 16 QACC lanes (16 pixels of one output channel) into one
 * 16-byte tile row. Same formula as esp_nn_conv_requant_lanes_fast, with
 * prm = {2 * comb, 2 * mult, rnd} and e1 = 1 - shift.
 *
 * PIE has no 32-bit multiply, so only `lw, addx2, mulhsu, sw` stay scalar.
 * The sign fix, rounding, shift, clamp and the pack to bytes run on four
 * lanes per PIE instruction. The core issues one instruction per cycle:
 * about 100 instructions per row against about 170 for the scalar requant
 * with strided byte stores.
 *
 * `res` is a 16-byte aligned 64-byte buffer. `row` is 16-byte aligned.
 */
static inline __attribute__((always_inline))
void esp_nn_conv_requant_tile_row(int32_t *res, const int32_t *prm,
                                  int32_t e1, int32_t act_min,
                                  int32_t act_max, int8_t *row)
{
    ESP_NN_QACC_EXTRACT_S32(res);
    TILE_RQ_SCALAR(res, res, prm[0], prm[1], 4);
    esp_nn_conv_requant_tile_vec(res, prm + 2, e1, act_min, act_max, row);
}

/*
 * Transposes an 8 x 16 byte tile (row j = output channel j, byte p =
 * pixel p) and stores pixel p's 8 bytes at out + p * stride.
 */
static inline __attribute__((always_inline))
void esp_nn_conv_tile8x16_store(const int8_t *tile, int8_t *out,
                                int32_t stride)
{
    asm volatile (
        "mv     x30, %[t]                    \n\t"
        "esp.vld.128.ip  q0, x30, 16         \n\t"
        "esp.vld.128.ip  q1, x30, 16         \n\t"
        "esp.vld.128.ip  q2, x30, 16         \n\t"
        "esp.vld.128.ip  q3, x30, 16         \n\t"
        "esp.vld.128.ip  q4, x30, 16         \n\t"
        "esp.vld.128.ip  q5, x30, 16         \n\t"
        "esp.vld.128.ip  q6, x30, 16         \n\t"
        "esp.vld.128.ip  q7, x30, 16         \n\t"
        "mv     x31, %[o]                    \n\t"
        "mv     x29, %[s]                    \n\t"
        ESP_NN_PIE_ZIP8X16
        /* Pixels 0-1, 2-3, ... 14-15 are in q0 q4 q2 q6 q1 q5 q3 q7. */
        "esp.vst.l.64.xp q0, x31, x29        \n\t"
        "esp.vst.h.64.xp q0, x31, x29        \n\t"
        "esp.vst.l.64.xp q4, x31, x29        \n\t"
        "esp.vst.h.64.xp q4, x31, x29        \n\t"
        "esp.vst.l.64.xp q2, x31, x29        \n\t"
        "esp.vst.h.64.xp q2, x31, x29        \n\t"
        "esp.vst.l.64.xp q6, x31, x29        \n\t"
        "esp.vst.h.64.xp q6, x31, x29        \n\t"
        "esp.vst.l.64.xp q1, x31, x29        \n\t"
        "esp.vst.h.64.xp q1, x31, x29        \n\t"
        "esp.vst.l.64.xp q5, x31, x29        \n\t"
        "esp.vst.h.64.xp q5, x31, x29        \n\t"
        "esp.vst.l.64.xp q3, x31, x29        \n\t"
        "esp.vst.h.64.xp q3, x31, x29        \n\t"
        "esp.vst.l.64.xp q7, x31, x29        \n\t"
        "esp.vst.h.64.xp q7, x31, x29        \n\t"
        :
        : [t] "r"(tile), [o] "r"(out), [s] "r"(stride)
        : "x29", "x30", "x31", "memory"
    );
}
