/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* int8 depthwise for ch_mult 1 and KxK filters of up to 31 taps.
 *
 * The int16 path converts every input row once per output row it feeds and
 * runs 8 lanes per MAC. This one keeps the data in int8 and runs 16 lanes
 * per ee.vmulas.s8.qacc. The input offset is folded into the bias, so a
 * padded tap holds -input_offset and contributes nothing. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <common_functions.h>
#include <esp_nn_multicore.h>
#include <sdkconfig.h>
#include "../common/esp_nn_qacc_esp32s3.h"
#include "esp_nn_conv_dispatch_esp32s3.h"

/* The requant block (ESP_NN_DW_S8_KXK_REQ_BLOCK), as the row kernel reads it
 * at fixed offsets:
 *   0 (1 << rs) - 1  64 ((1 << rs) - 1) >> 1  128 right_shift  192 mult
 *   256 left_shift  320 bias with the offset folded in, times 2 when
 *   req_ctl bit 8 is clear (the fast requant takes 2x) */
#define KXK_REQ_BLOCK       ESP_NN_DW_S8_KXK_REQ_BLOCK
/* Index of the bias row in a requant block: 5 rows of 16 int32 come first */
#define KXK_REQ_BIAS        (5 * 16)

/* ------------------------------------------------------------ Row kernel --
 *
 * One output row of n_groups 16-channel groups, from a padded int8 strip.
 * Input and filter taps are both cs bytes apart, so one stride register
 * walks both. Input rows come from a table of kh row pointers, so the
 * caller can keep them in a ring and copy each input row once.
 *
 * The kernel is one asm statement. Its state lives in kxk_row_t, which a4
 * points to for the whole statement. The asm uses every other address
 * register, and never touches a0 or sp. Offsets into kxk_row_t are below,
 * checked against the struct. */

#define R_DUMP_A        0       /* QACC dump of a pixel, 52 bytes */
#define R_STAGE_END     52      /* end of the used stages */
#define R_FILTER        56      /* the group's filter */
#define R_OUT           60      /* the next output pixel of the group */
#define R_STAGE         64      /* staged sums: group g of pixel p at 32 g + 16 p */
#define R_G_OFF         192     /* byte offset of the group in a row */
#define R_G_OUT         196     /* the group's first output pixel */
#define R_G_REQ         200     /* the group's requant block */
#define R_G_LEFT        204     /* groups left, this one included */
#define R_DUMP_B        208     /* QACC dump of the second pixel of a pair */
#define R_SAVE_A2       260
#define R_SAVE_A5       264
#define R_CS            268
#define R_HSTEP         272     /* 32 when the pair requant runs two halves */
#define R_HBACK         276     /* 0 */
#define R_ROWS          280
#define R_KW            284
#define R_KH            288
#define R_X_STEP        292
#define R_N_OUT         296
#define R_OUT_STEP      300
#define R_HSTEP2        304     /* R_HSTEP + 32: 0 */
#define R_HBACK2        308     /* R_HBACK + 32: 32 */
#define R_OUT_OFFSET    312
#define R_CLAMP_MIN     316
#define R_CLAMP_MAX     320
#define R_STORE         324
#define R_REQ_CTL       328
#define R_VARIANT       332     /* 5 or 3 for a 5x5 or 3x3 filter, else 0 */

typedef struct {
    int32_t dump_a[13];
    int32_t *stage_end;
    const int8_t *filter;
    int8_t *out;
    int32_t stage[32];
    int32_t g_off;
    int8_t *g_out;
    const int32_t *g_req;
    int32_t g_left;
    int32_t dump_b[13];
    int32_t save_a2, save_a5;
    int32_t cs;
    int32_t hstep, hback;
    const int8_t *const *rows;
    int32_t kw, kh;
    int32_t x_step;
    int32_t n_out;
    int32_t out_step;
    int32_t hstep2, hback2;
    int32_t out_offset;
    int32_t clamp_min;  /* activation_min - out_offset */
    int32_t clamp_max;  /* activation_max - out_offset */
    /* 0 one 16-byte store (out 16-aligned), 1 two 8-byte stores, 2 the low
     * 8 bytes only (an 8-channel tail group) */
    int32_t store;
    /* bits 0-2 the lane groups of four to requant, 4, or 2 for an
     * 8-channel group (lanes 8-15 unused). Bit 8 selects the general
     * requant: some channel has a left shift, or some |x| may reach 2^30. */
    int32_t req_ctl;
    int32_t variant;
} __attribute__((aligned(16))) kxk_row_t;

#define KXK_CHECK(f, o) _Static_assert(offsetof(kxk_row_t, f) == (o), #f)
KXK_CHECK(dump_a, R_DUMP_A);
KXK_CHECK(stage_end, R_STAGE_END);
KXK_CHECK(filter, R_FILTER);
KXK_CHECK(out, R_OUT);
KXK_CHECK(stage, R_STAGE);
KXK_CHECK(g_off, R_G_OFF);
KXK_CHECK(g_out, R_G_OUT);
KXK_CHECK(g_req, R_G_REQ);
KXK_CHECK(g_left, R_G_LEFT);
KXK_CHECK(dump_b, R_DUMP_B);
KXK_CHECK(save_a2, R_SAVE_A2);
KXK_CHECK(save_a5, R_SAVE_A5);
KXK_CHECK(cs, R_CS);
KXK_CHECK(hstep, R_HSTEP);
KXK_CHECK(hback, R_HBACK);
KXK_CHECK(rows, R_ROWS);
KXK_CHECK(kw, R_KW);
KXK_CHECK(kh, R_KH);
KXK_CHECK(x_step, R_X_STEP);
KXK_CHECK(n_out, R_N_OUT);
KXK_CHECK(out_step, R_OUT_STEP);
KXK_CHECK(hstep2, R_HSTEP2);
KXK_CHECK(hback2, R_HBACK2);
KXK_CHECK(out_offset, R_OUT_OFFSET);
KXK_CHECK(clamp_min, R_CLAMP_MIN);
KXK_CHECK(clamp_max, R_CLAMP_MAX);
KXK_CHECK(store, R_STORE);
KXK_CHECK(req_ctl, R_REQ_CTL);
KXK_CHECK(variant, R_VARIANT);
_Static_assert(R_HSTEP2 == R_HSTEP + 32 && R_HBACK2 == R_HBACK + 32,
               "the pair requant reads these 32 bytes on in its second half");

/* a4-relative load or store of a kxk_row_t field */
#define LD(r, f)    "l32i   " r ", a4, " ESP_NN_QACC_S(f) "\n"
#define ST(r, f)    "s32i   " r ", a4, " ESP_NN_QACC_S(f) "\n"

/* The requant follows esp_nn_multiply_by_quantized_mult_ver1_esp32s3 step
 * for step, so it is bit-exact for any mult and shift. */

/* The general requant, for left shifts or a large bias. Lanes k and k+1
 * of a group of four, interleaved to hide the multiply latency:
 * x = staged sum << ls, the doubling high multiply with the 1 << 30 nudge
 * (a15), then r into q0 and r >> rs into q7. a6 points at the staged sums
 * and a7 at the requant block. */
#define KXK_REQ_PAIR(k)                                 \
    "l32i   a8, a6, 4 * " #k "\n"                       \
    "l32i   a12, a6, 4 * " #k " + 4\n"                  \
    "l32i   a10, a7, 192 + 4 * " #k "\n"                \
    "l32i   a13, a7, 196 + 4 * " #k "\n"                \
    "l32i   a9, a7, 256 + 4 * " #k "\n"                 \
    "l32i   a3, a7, 260 + 4 * " #k "\n"                 \
    "ssl    a9\n"                                       \
    "sll    a8, a8\n"                                   \
    "ssl    a3\n"                                       \
    "sll    a12, a12\n"                                 \
    "mull   a11, a10, a8\n"                             \
    "mulsh  a10, a10, a8\n"                             \
    "mull   a14, a13, a12\n"                            \
    "mulsh  a13, a13, a12\n"                            \
    "add.n  a11, a11, a15\n"                            \
    "saltu  a9, a11, a15\n"                             \
    "add.n  a14, a14, a15\n"                            \
    "saltu  a3, a14, a15\n"                             \
    "add.n  a10, a10, a9\n"                             \
    "add.n  a13, a13, a3\n"                             \
    "ssai   31\n"                                       \
    "src    a8, a10, a11\n"                             \
    "src    a12, a13, a14\n"                            \
    "l32i   a9, a7, 128 + 4 * " #k "\n"                 \
    "l32i   a3, a7, 132 + 4 * " #k "\n"                 \
    "ee.movi.32.q   q0, a8, " #k "\n"                   \
    "ee.movi.32.q   q0, a12, " #k " + 1\n"              \
    "ssr    a9\n"                                       \
    "sra    a8, a8\n"                                   \
    "ssr    a3\n"                                       \
    "sra    a12, a12\n"                                 \
    "ee.movi.32.q   q7, a8, " #k "\n"                   \
    "ee.movi.32.q   q7, a12, " #k " + 1\n"

/* Lanes k and k+1 when every |x| < 2^30 and no channel shifts left. The
 * stage holds 2x, and (x * m + 2^30) >> 31 is the high word of 2x * m plus
 * the top bit of its low word. No carry or funnel shift is needed. */
#define KXK_REQ_PAIR_2X(k)                              \
    "l32i   a8, a6, 4 * " #k "\n"                       \
    "l32i   a12, a6, 4 * " #k " + 4\n"                  \
    "l32i   a10, a7, 192 + 4 * " #k "\n"                \
    "l32i   a13, a7, 196 + 4 * " #k "\n"                \
    "mull   a11, a10, a8\n"                             \
    "mulsh  a10, a10, a8\n"                             \
    "mull   a14, a13, a12\n"                            \
    "mulsh  a13, a13, a12\n"                            \
    "l32i   a9, a7, 128 + 4 * " #k "\n"                 \
    "l32i   a3, a7, 132 + 4 * " #k "\n"                 \
    "srli   a11, a11, 31\n"                             \
    "srli   a14, a14, 31\n"                             \
    "add.n  a8, a10, a11\n"                             \
    "add.n  a12, a13, a14\n"                            \
    "ee.movi.32.q   q0, a8, " #k "\n"                   \
    "ee.movi.32.q   q0, a12, " #k " + 1\n"              \
    "ssr    a9\n"                                       \
    "sra    a8, a8\n"                                   \
    "ssr    a3\n"                                       \
    "sra    a12, a12\n"                                 \
    "ee.movi.32.q   q7, a8, " #k "\n"                   \
    "ee.movi.32.q   q7, a12, " #k " + 1\n"

/* One group of four lanes, then the rounding step of RoundingDivideByPOT in
 * vector form. The result replaces the staged sums. q4 holds zero, q5 is
 * a temp. */
#define KXK_REQ_GROUP(PAIR)                             \
    PAIR(0)                                             \
    PAIR(2)                                             \
    "ld.qr  q5, a7, 64\n"                               \
    "ee.vld.128.ip  q2, a7, 16\n"                       \
    "ee.vcmp.lt.s32 q3, q7, q4\n"                       \
    "ee.andq    q1, q0, q2\n"                           \
    "ee.vsubs.s32   q2, q5, q3\n"                       \
    "ee.vcmp.gt.s32 q1, q1, q2\n"                       \
    "ee.vsubs.s32   q1, q7, q1\n"                       \
    "ee.vst.128.ip  q1, a6, 32\n"

/* Lane j (0-7) of the two QACC dumps of a pair, pixel A at a4 + R_DUMP_A
 * and pixel B at a4 + R_DUMP_B, into a8 and a12. Lane j holds bits 20 j to
 * 20 j + 19 of the five words. Each word is loaded once per half: A's words
 * alternate in a2 and a5, B's in a13 and a3. One ssai serves both pixels. */
#define KXK_LD_AB(a, b, w)                                      \
    "l32i   " a ", a4, " ESP_NN_QACC_S(R_DUMP_A) " + 4 * " #w "\n"      \
    "l32i   " b ", a4, " ESP_NN_QACC_S(R_DUMP_B) " + 4 * " #w "\n"
#define KXK_SRC_AB(sh, ahi, alo, bhi, blo)                      \
    "ssai   " #sh "\n"                                          \
    "src    a8, " ahi ", " alo "\n"                             \
    "src    a12, " bhi ", " blo "\n"                            \
    "sext   a8, a8, 19\n"                                       \
    "sext   a12, a12, 19\n"
#define KXK_MID_AB(sh, a, b)                                    \
    "slli   a8, " a ", " #sh "\n"                               \
    "slli   a12, " b ", " #sh "\n"                              \
    "srai   a8, a8, 12\n"                                       \
    "srai   a12, a12, 12\n"

#define KXK_LANE_2PX_0                                          \
    KXK_LD_AB("a2", "a13", 0)                                   \
    "sext   a8, a2, 19\n"                                       \
    "sext   a12, a13, 19\n"
#define KXK_LANE_2PX_1                                          \
    KXK_LD_AB("a5", "a3", 1)                                    \
    KXK_SRC_AB(20, "a5", "a2", "a3", "a13")
#define KXK_LANE_2PX_2  KXK_MID_AB(4, "a5", "a3")
#define KXK_LANE_2PX_3                                          \
    KXK_LD_AB("a2", "a13", 2)                                   \
    KXK_SRC_AB(28, "a2", "a5", "a13", "a3")
#define KXK_LANE_2PX_4                                          \
    KXK_LD_AB("a5", "a3", 3)                                    \
    KXK_SRC_AB(16, "a5", "a2", "a3", "a13")
#define KXK_LANE_2PX_5  KXK_MID_AB(8, "a5", "a3")
#define KXK_LANE_2PX_6                                          \
    KXK_LD_AB("a2", "a13", 4)                                   \
    KXK_SRC_AB(24, "a2", "a5", "a13", "a3")
#define KXK_LANE_2PX_7                                          \
    "srai   a8, a2, 12\n"                                       \
    "srai   a12, a13, 12\n"

/* The fast requant of lane k of a group for two pixels, straight from
 * their QACC dumps: lane j of the half. The bias row holds 2 * bias, so
 * one addx2 gives the 2x of KXK_REQ_PAIR_2X. r lands in q0 and q1,
 * r >> rs in q7 and q6. a7 is 16 bytes past the group's block. */
#define KXK_DLANE_2PX(k, j)                             \
    KXK_LANE_2PX_##j                                    \
    "l32i   a15, a7, 304 + 4 * " #k "\n"                \
    "l32i   a10, a7, 176 + 4 * " #k "\n"                \
    "addx2  a8, a8, a15\n"                              \
    "addx2  a12, a12, a15\n"                            \
    "mull   a11, a10, a8\n"                             \
    "mulsh  a8, a10, a8\n"                              \
    "mull   a14, a10, a12\n"                            \
    "mulsh  a12, a10, a12\n"                            \
    "l32i   a9, a7, 112 + 4 * " #k "\n"                 \
    "srli   a11, a11, 31\n"                             \
    "srli   a14, a14, 31\n"                             \
    "add.n  a8, a8, a11\n"                              \
    "add.n  a12, a12, a14\n"                            \
    "ee.movi.32.q   q0, a8, " #k "\n"                   \
    "ee.movi.32.q   q1, a12, " #k "\n"                  \
    "ssr    a9\n"                                       \
    "sra    a8, a8\n"                                   \
    "sra    a12, a12\n"                                 \
    "ee.movi.32.q   q7, a8, " #k "\n"                   \
    "ee.movi.32.q   q6, a12, " #k "\n"

/* Four lanes of the half dumps for two pixels: one group, then the
 * rounding step for both. q4 holds zero. */
#define KXK_DGROUP_2PX(LANES)                           \
    "ld.qr  q5, a7, 64\n"                               \
    "ee.vld.128.ip  q2, a7, 16\n"                       \
    LANES                                               \
    "ee.vcmp.lt.s32 q3, q7, q4\n"                       \
    "ee.andq    q0, q0, q2\n"                           \
    "ee.vsubs.s32   q3, q5, q3\n"                       \
    "ee.vcmp.gt.s32 q0, q0, q3\n"                       \
    "ee.vcmp.lt.s32 q3, q6, q4\n"                       \
    "ee.vsubs.s32   q7, q7, q0\n"                       \
    "ee.andq    q1, q1, q2\n"                           \
    "ee.vsubs.s32   q3, q5, q3\n"                       \
    "ee.vst.128.ip  q7, a6, 16\n"                       \
    "ee.vcmp.gt.s32 q1, q1, q3\n"                       \
    "ee.vsubs.s32   q6, q6, q1\n"                       \
    "ee.vst.128.ip  q6, a6, 16\n"

#define KXK_DLANES_LO                                   \
    KXK_DLANE_2PX(0, 0) KXK_DLANE_2PX(1, 1)             \
    KXK_DLANE_2PX(2, 2) KXK_DLANE_2PX(3, 3)
#define KXK_DLANES_HI                                   \
    KXK_DLANE_2PX(0, 4) KXK_DLANE_2PX(1, 5)             \
    KXK_DLANE_2PX(2, 6) KXK_DLANE_2PX(3, 7)

/* MAC bodies: QACC += one pixel of the group. On entry a2 is the pixel's
 * byte offset in a row, a7 cs, a9 the filter and a12 the row table. They
 * may use a8-a10 and a12-a15, and must leave a2, a3, a5, a6, a7 and a11.
 *
 * Generic rows: the last tap of each row is outside the loop, so nothing
 * past the filter or the strip is loaded. */
#define KXK_MAC_ANY                                     \
    LD("a12", R_ROWS)                                   \
    LD("a13", R_KW)                                     \
    LD("a14", R_KH)                                     \
    "addi   a15, a13, -1\n"                             \
    ".Lkxk_row%=:\n"                                    \
    "l32i   a8, a12, 0\n"                               \
    "addi   a12, a12, 4\n"                              \
    "add.n  a10, a8, a2\n"                              \
    "ee.vld.128.xp  q0, a10, a7\n"                      \
    "ee.vld.128.xp  q1, a9, a7\n"                       \
    "loopgtz    a15, .Lkxk_row_end%=\n"                 \
    "ee.vmulas.s8.qacc.ld.xp    q0, a10, a7, q0, q1\n"  \
    "ee.vld.128.xp  q1, a9, a7\n"                       \
    ".Lkxk_row_end%=:\n"                                \
    "ee.vmulas.s8.qacc  q0, q1\n"                       \
    "addi.n a14, a14, -1\n"                             \
    "bnez   a14, .Lkxk_row%=\n"

/* 5x5 and 3x3 rows: three input registers rotate, so no load feeds the
 * next instruction. The last two MACs of a row load the next row's first
 * two filter taps, so every MAC carries a load. The last row is peeled and
 * loads nothing past the filter. Fully unrolled, so each row pointer is
 * one load at a fixed offset. */
#define KXK_ROW5(off)                                   \
    "l32i   a8, a12, " #off "\n"                        \
    "ee.vld.128.xp  q3, a9, a7\n"                       \
    "add.n  a10, a8, a2\n"                              \
    "ee.vld.128.xp  q4, a9, a7\n"                       \
    "ee.vld.128.xp  q0, a10, a7\n"                      \
    "ee.vld.128.xp  q6, a10, a7\n"                      \
    "ee.vld.128.xp  q5, a9, a7\n"                       \
    "ee.vmulas.s8.qacc.ld.xp    q7, a10, a7, q0, q1\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q0, a10, a7, q6, q2\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q6, a10, a7, q7, q3\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q1, a9, a7, q0, q4\n"   \
    "ee.vmulas.s8.qacc.ld.xp    q2, a9, a7, q6, q5\n"

#define KXK_MAC_55                                      \
    "ee.vld.128.xp  q1, a9, a7\n"                       \
    "ee.vld.128.xp  q2, a9, a7\n"                       \
    KXK_ROW5(0)                                         \
    KXK_ROW5(4)                                         \
    KXK_ROW5(8)                                         \
    KXK_ROW5(12)                                        \
    "l32i   a8, a12, 16\n"                              \
    "ee.vld.128.xp  q3, a9, a7\n"                       \
    "add.n  a10, a8, a2\n"                              \
    "ee.vld.128.xp  q4, a9, a7\n"                       \
    "ee.vld.128.xp  q0, a10, a7\n"                      \
    "ee.vld.128.xp  q6, a10, a7\n"                      \
    "ee.vld.128.xp  q5, a9, a7\n"                       \
    "ee.vmulas.s8.qacc.ld.xp    q7, a10, a7, q0, q1\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q0, a10, a7, q6, q2\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q6, a10, a7, q7, q3\n"  \
    "ee.vmulas.s8.qacc  q0, q4\n"                       \
    "ee.vmulas.s8.qacc  q6, q5\n"

#define KXK_ROW3(off)                                   \
    "l32i   a8, a12, " #off "\n"                        \
    "ee.vld.128.xp  q3, a9, a7\n"                       \
    "add.n  a10, a8, a2\n"                              \
    "ee.vld.128.xp  q0, a10, a7\n"                      \
    "ee.vld.128.xp  q6, a10, a7\n"                      \
    "ee.vmulas.s8.qacc.ld.xp    q7, a10, a7, q0, q1\n"  \
    "ee.vmulas.s8.qacc.ld.xp    q1, a9, a7, q6, q2\n"   \
    "ee.vmulas.s8.qacc.ld.xp    q2, a9, a7, q7, q3\n"

#define KXK_MAC_33                                      \
    "ee.vld.128.xp  q1, a9, a7\n"                       \
    "ee.vld.128.xp  q2, a9, a7\n"                       \
    KXK_ROW3(0)                                         \
    KXK_ROW3(4)                                         \
    "l32i   a8, a12, 8\n"                               \
    "ee.vld.128.xp  q3, a9, a7\n"                       \
    "add.n  a10, a8, a2\n"                              \
    "ee.vld.128.xp  q0, a10, a7\n"                      \
    "ee.vld.128.xp  q6, a10, a7\n"                      \
    "ee.vmulas.s8.qacc.ld.xp    q7, a10, a7, q0, q1\n"  \
    "ee.vmulas.s8.qacc  q6, q2\n"                       \
    "ee.vmulas.s8.qacc  q7, q3\n"

/* After a pixel's MACs. A first pixel of a fast pair dumps QACC and goes
 * on to the second at next, with a7 (cs) and a12 (rows) kept. The general
 * requant and a lone last pixel unpack and stage instead. Each MAC body has
 * its own copy, so the second pixel of a pair takes no dispatch. */
#define KXK_MAC_DONE(next)                                                  \
    "addi   a8, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    "bbsi   a11, 8, .Lkxk_unpack%=\n"                                       \
    "bne    a6, a8, .Lkxk_dump_b%=\n"                                       \
    "beqi   a5, 1, .Lkxk_unpack%=\n"                                        \
    ESP_NN_QACC_DUMP("a4", R_DUMP_A, "a10")                                 \
    "addi   a6, a6, 16\n"                                                   \
    "add.n  a2, a2, a3\n"                                                   \
    "addi.n a5, a5, -1\n"                                                   \
    "j      " next "\n"

/* The whole row.
 *
 * Two pixels per pass share one requant pass. The fast pair requant reads
 * the lanes from two QACC dumps. The general requant and a lone last pixel
 * unpack and stage, and requant one pixel at a time.
 *
 * The MAC body is picked per pair from R_VARIANT: 5x5, 3x3, else generic.
 * One asm body with three MAC bodies is about 2.3 KB. One function per MAC
 * body, picked from C, would repeat the requant and cost 3.6 KB more.
 * The row stays one asm statement: split into smaller statements, the C
 * code between them would reload the state for every pixel.
 *
 * a2: the pixel's byte offset in a row  a3: x_step during the MACs
 * a4: kxk_row_t  a5: pixels left  a6: the stage  a7: cs, then req
 * a11: req_ctl during the MACs. The rest are temps. */
#define KXK_ROW_ASM                                                         \
    LD("a5", R_N_OUT)                                                       \
    LD("a2", R_G_OFF)                                                       \
    ".Lkxk_pair%=:\n"                                                       \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    LD("a3", R_X_STEP)                                                      \
    LD("a11", R_REQ_CTL)                                                    \
    ".Lkxk_px%=:\n"                                                         \
    LD("a7", R_CS)                                                          \
    LD("a12", R_ROWS)                                                       \
    LD("a8", R_VARIANT)                                                     \
    "bnei   a8, 5, .Lkxk_px_n5%=\n"                                         \
    ".Lkxk_px_b55%=:\n"                                                     \
    "ee.zero.qacc\n"                                                        \
    LD("a9", R_FILTER)                                                      \
    KXK_MAC_55                                                              \
    KXK_MAC_DONE(".Lkxk_px_b55%=")                                          \
    ".Lkxk_px_n5%=:\n"                                                      \
    "bnei   a8, 3, .Lkxk_px_b_any%=\n"                                      \
    ".Lkxk_px_b33%=:\n"                                                     \
    "ee.zero.qacc\n"                                                        \
    LD("a9", R_FILTER)                                                      \
    KXK_MAC_33                                                              \
    KXK_MAC_DONE(".Lkxk_px_b33%=")                                          \
    ".Lkxk_px_b_any%=:\n"                                                   \
    "ee.zero.qacc\n"                                                        \
    LD("a9", R_FILTER)                                                      \
    KXK_MAC_ANY                                                             \
    KXK_MAC_DONE(".Lkxk_px_b_any%=")                                        \
                                                                            \
    /* An 8-channel tail group (store 2) needs lanes 0-7 only, so it */     \
    /* skips the QACC_H unpack and half the requant. */                     \
    ".Lkxk_unpack%=:\n"                                                     \
    ESP_NN_QACC_DUMP("a4", R_DUMP_A, "a10")                                 \
    LD("a9", R_REQ_CTL)                                                     \
    LD("a7", R_G_REQ)                                                       \
    ESP_NN_QACC_UNPACK_HALF("a4", 0, "q0", "q3",                            \
                            "a8", "a12", "a13", "a14", "a15")               \
    "bbci   a9, 2, .Lkxk_unpacked%=\n"                                      \
    ESP_NN_QACC_UNPACK_HALF("a4", 32, "q1", "q2",                           \
                            "a8", "a12", "a13", "a14", "a15")               \
    ".Lkxk_unpacked%=:\n"                                                   \
    /* add the folded bias and stage the four groups at a6. The fast */     \
    /* requant takes 2x, and its bias row holds 2 * bias. */                \
    "addmi  a8, a7, 256\n"                                                  \
    "ld.qr  q4, a8, 64\n"                                                   \
    "ld.qr  q5, a8, 80\n"                                                   \
    "ld.qr  q6, a8, 96\n"                                                   \
    "ld.qr  q7, a8, 112\n"                                                  \
    "bbsi   a9, 8, .Lkxk_bias%=\n"                                          \
    "ee.vadds.s32   q0, q0, q0\n"                                           \
    "ee.vadds.s32   q3, q3, q3\n"                                           \
    "ee.vadds.s32   q1, q1, q1\n"                                           \
    "ee.vadds.s32   q2, q2, q2\n"                                           \
    ".Lkxk_bias%=:\n"                                                       \
    "ee.vadds.s32   q0, q0, q4\n"                                           \
    "ee.vadds.s32   q3, q3, q5\n"                                           \
    "ee.vadds.s32   q1, q1, q6\n"                                           \
    "ee.vadds.s32   q2, q2, q7\n"                                           \
    "st.qr  q0, a6, 0\n"                                                    \
    "st.qr  q3, a6, 32\n"                                                   \
    "st.qr  q1, a6, 64\n"                                                   \
    "st.qr  q2, a6, 96\n"                                                   \
    LD("a8", R_X_STEP)                                                      \
    "addi   a6, a6, 16\n"                                                   \
    "add.n  a2, a2, a8\n"                                                   \
    "addi.n a5, a5, -1\n"                                                   \
    "addi   a8, a4, " ESP_NN_QACC_S(R_STAGE) " + 16\n"                              \
    "beqz   a5, .Lkxk_req%=\n"                                              \
    "beq    a6, a8, .Lkxk_px%=\n"                                           \
    ".Lkxk_req%=:\n"                                                        \
    /* On the fast path only a lone last pixel gets here. It takes a */     \
    /* one-pixel loop: the two-pixel pass costs it much more than half. */  \
    ST("a6", R_STAGE_END)                                                   \
    "extui  a3, a9, 0, 3\n"     /* loop count only, a3 is a temp inside */  \
    "ee.zero.q  q4\n"                                                       \
    "bbsi   a9, 8, .Lkxk_req_shift%=\n"                                     \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    "loop   a3, .Lkxk_req_done%=\n"                                         \
    KXK_REQ_GROUP(KXK_REQ_PAIR_2X)                                          \
    ".Lkxk_req_done%=:\n"                                                   \
    LD("a11", R_STAGE_END)                                                  \
    ".Lkxk_req_done_a11%=:\n"                                               \
    /* Clamp to the range shifted by out_offset, then add out_offset to */  \
    /* 16-bit lanes: half the adds, and the same result as add then clamp */\
    "addi   a8, a4, " ESP_NN_QACC_S(R_OUT_OFFSET) "\n"                              \
    "ee.vldbc.16.ip q1, a8, 4\n"                                            \
    "ee.vldbc.32.ip q2, a8, 4\n"                                            \
    "ee.vldbc.32    q3, a8\n"                                               \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    LD("a10", R_OUT)                                                        \
    LD("a8", R_OUT_STEP)                                                    \
    LD("a9", R_STORE)                                                       \
    ".Lkxk_st%=:\n"                                                         \
    "ld.qr  q5, a6, 32\n"                                                   \
    "ld.qr  q6, a6, 64\n"                                                   \
    "ld.qr  q0, a6, 96\n"                                                   \
    "ee.vld.128.ip  q4, a6, 16\n"                                           \
    "ee.vmin.s32    q5, q5, q3\n"                                           \
    "ee.vmin.s32    q6, q6, q3\n"                                           \
    "ee.vmin.s32    q0, q0, q3\n"                                           \
    "ee.vmin.s32    q4, q4, q3\n"                                           \
    "ee.vmax.s32    q4, q4, q2\n"                                           \
    "ee.vmax.s32    q5, q5, q2\n"                                           \
    "ee.vmax.s32    q6, q6, q2\n"                                           \
    "ee.vmax.s32    q0, q0, q2\n"                                           \
    "ee.vunzip.16   q4, q5\n"                                               \
    "ee.vunzip.16   q6, q0\n"                                               \
    "ee.vadds.s16   q4, q4, q1\n"                                           \
    "ee.vadds.s16   q6, q6, q1\n"                                           \
    "ee.vunzip.8    q4, q6\n"                                               \
    "bnez   a9, .Lkxk_st64%=\n"                                             \
    "ee.vst.128.xp  q4, a10, a8\n"                                          \
    ".Lkxk_stored%=:\n"                                                     \
    "bne    a6, a11, .Lkxk_st%=\n"                                          \
    ST("a10", R_OUT)                                                        \
    "bnez   a5, .Lkxk_pair%=\n"                                             \
                                                                            \
    /* The next group: 16 bytes on in the input, filter and out, and one */ \
    /* requant block on in req. */                                          \
    LD("a8", R_G_LEFT)                                                      \
    LD("a2", R_G_OFF)                                                       \
    LD("a6", R_G_OUT)                                                       \
    "addi.n a8, a8, -1\n"                                                   \
    "beqz   a8, .Lkxk_done%=\n"                                             \
    LD("a7", R_G_REQ)                                                       \
    LD("a3", R_FILTER)                                                      \
    "addi   a2, a2, 16\n"                                                   \
    "addi   a6, a6, 16\n"                                                   \
    "addi   a7, a7, " ESP_NN_QACC_S(KXK_REQ_BLOCK) " * 4\n"                         \
    "addi   a3, a3, 16\n"                                                   \
    LD("a5", R_N_OUT)                                                       \
    ST("a8", R_G_LEFT)                                                      \
    ST("a2", R_G_OFF)                                                       \
    ST("a6", R_G_OUT)                                                       \
    ST("a6", R_OUT)                                                         \
    ST("a7", R_G_REQ)                                                       \
    ST("a3", R_FILTER)                                                      \
    "j      .Lkxk_pair%=\n"                                                 \
                                                                            \
    /* Out of line, so the 16-byte store path takes no jump. */             \
    /* channels % 16 == 8 leaves every other pixel 8-aligned only. */       \
    ".Lkxk_st64%=:\n"                                                       \
    "bbsi   a9, 1, .Lkxk_st_tail%=\n"                                       \
    "ee.vst.l.64.ip q4, a10, 8\n"                                           \
    "ee.vst.h.64.ip q4, a10, -8\n"                                          \
    "add.n  a10, a10, a8\n"                                                 \
    "j      .Lkxk_stored%=\n"                                               \
    ".Lkxk_st_tail%=:\n"                                                    \
    "ee.vst.l.64.ip q4, a10, 0\n"                                           \
    "add.n  a10, a10, a8\n"                                                 \
    "j      .Lkxk_stored%=\n"                                               \
                                                                            \
    /* The fast pair requant from the two QACC dumps. QACC_L gives */       \
    /* groups 0 and 1, QACC_H groups 2 and 3. An 8-channel group runs */    \
    /* QACC_L only. The body is over 256 bytes, too big for a */            \
    /* zero-overhead loop. a2, a3, a5 and a13 hold dump words, so a2 and */ \
    /* a5 are saved. a4 itself walks to the second half: R_HSTEP and */     \
    /* R_HBACK, read 32 bytes on, end the loop and step back. */            \
    ".Lkxk_dump_b%=:\n"                                                     \
    ESP_NN_QACC_DUMP("a4", R_DUMP_B, "a10")                                 \
    "add.n  a2, a2, a3\n"                                                   \
    "addi.n a5, a5, -1\n"                                                   \
    ST("a2", R_SAVE_A2)                                                     \
    ST("a5", R_SAVE_A5)                                                     \
    "ee.zero.q  q4\n"                                                       \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    LD("a7", R_G_REQ)                                                       \
    ".Lkxk_dhalf%=:\n"                                                      \
    KXK_DGROUP_2PX(KXK_DLANES_LO)                                           \
    KXK_DGROUP_2PX(KXK_DLANES_HI)                                           \
    LD("a8", R_HSTEP)                                                       \
    "add.n  a4, a4, a8\n"                                                   \
    "bnez   a8, .Lkxk_dhalf%=\n"                                            \
    LD("a8", R_HBACK)                                                       \
    "sub    a4, a4, a8\n"                                                   \
    LD("a2", R_SAVE_A2)                                                     \
    LD("a5", R_SAVE_A5)                                                     \
    "addi   a11, a4, " ESP_NN_QACC_S(R_STAGE) " + 32\n" /* both stages used */      \
    "j      .Lkxk_req_done_a11%=\n"                                         \
                                                                            \
    /* The general requant, for left shifts or a large bias. One pixel */   \
    /* at a time, both stages. A lone pixel's second stage is stale and */  \
    /* unused. */                                                           \
    ".Lkxk_req_shift%=:\n"                                                  \
    "movi.n a15, 1\n"                                                       \
    "slli   a15, a15, 30\n"                                                 \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) "\n"                                   \
    "loop   a3, .Lkxk_req_shift_end%=\n"                                    \
    KXK_REQ_GROUP(KXK_REQ_PAIR)                                             \
    ".Lkxk_req_shift_end%=:\n"                                              \
    LD("a9", R_REQ_CTL)         /* a3 and a9 are temps above */             \
    LD("a7", R_G_REQ)                                                       \
    "addi   a6, a4, " ESP_NN_QACC_S(R_STAGE) " + 16\n"                              \
    "extui  a3, a9, 0, 3\n"                                                 \
    "loop   a3, .Lkxk_req_shift_end2%=\n"                                   \
    KXK_REQ_GROUP(KXK_REQ_PAIR)                                             \
    ".Lkxk_req_shift_end2%=:\n"                                             \
    "j      .Lkxk_req_done%=\n"                                             \
    ".Lkxk_done%=:\n"

#define KXK_ROW_CLOBBERS                                                    \
    "a2", "a3", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13",   \
    "a14", "a15", "memory"

/* noinline keeps the register pressure of the asm out of the caller's
 * loops. */
static void __attribute__((noinline)) kxk_row(kxk_row_t *r)
{
    register kxk_row_t *a4 asm("a4") = r;
    asm volatile(KXK_ROW_ASM : "+r"(a4) : : KXK_ROW_CLOBBERS);
}

/* ------------------------------------------------------- Small helpers -- */

/* Copies n_px >= 1 pixels of n bytes (n % 8 == 0) from an 8-aligned src
 * to a 16-aligned dst, dst_step bytes apart. An 8-byte tail is stored as
 * 16 bytes. Its upper 8 are padding lanes, which have zero weights. */
static inline void kxk_copy(int8_t *dst, const int8_t *src, int n_px, int n, int dst_step)
{
    const int32_t n32 = n >> 5;
    const int32_t skip = dst_step - (((n + 8) >> 4) << 4);
    asm volatile(
        "1:\n"
        "loopgtz        %[n32], 2f\n"
        "ee.vld.l.64.ip q0, %[src], 8\n"
        "ee.vld.h.64.ip q0, %[src], 8\n"
        "ee.vld.l.64.ip q1, %[src], 8\n"
        "ee.vld.h.64.ip q1, %[src], 8\n"
        "ee.vst.128.ip  q0, %[dst], 16\n"
        "ee.vst.128.ip  q1, %[dst], 16\n"
        "2:\n"
        "bbci           %[n], 4, 3f\n"
        "ee.vld.l.64.ip q0, %[src], 8\n"
        "ee.vld.h.64.ip q0, %[src], 8\n"
        "ee.vst.128.ip  q0, %[dst], 16\n"
        "3:\n"
        "bbci           %[n], 3, 4f\n"
        "ee.vld.l.64.ip q0, %[src], 8\n"
        "ee.vst.128.ip  q0, %[dst], 16\n"
        "4:\n"
        "add.n          %[dst], %[dst], %[skip]\n"
        "addi.n         %[n_px], %[n_px], -1\n"
        "bnez           %[n_px], 1b\n"
        : [dst] "+r" (dst), [src] "+r" (src), [n_px] "+r" (n_px)
        : [n32] "r" (n32), [n] "r" (n), [skip] "r" (skip)
        : "memory");
}

/* Fills the requant rows of n lanes of one block. The bias row holds the
 * filter sums on entry. Returns nonzero if some lane shifts left or has a
 * folded bias outside [-2^29, 2^29), which the fast requant cannot take.
 * In asm, as GCC spilled in this loop and took ~45 cycles per channel. */
static inline uint32_t kxk_req_lanes(int32_t *req, const int32_t *mult, const int32_t *shift,
                                     const int32_t *bias, int n, int32_t input_offset)
{
    uint32_t general;
    int32_t ls, rs, sum, m, t;
    asm volatile(
        "movi.n     %[gen], 0\n"
        "loopgtz    %[n], 1f\n"
        "l32i.n     %[rs], %[shift], 0\n"
        "l32i       %[sum], %[req], 320\n"      /* filter sum */
        "movi.n     %[ls], 0\n"
        "max        %[ls], %[rs], %[ls]\n"      /* left shift */
        "l32i.n     %[m], %[mult], 0\n"
        "sub        %[rs], %[ls], %[rs]\n"      /* right shift */
        "mull       %[sum], %[sum], %[off]\n"
        "s32i       %[m], %[req], 192\n"
        "ssl        %[rs]\n"
        "movi.n     %[m], 1\n"
        "sll        %[m], %[m]\n"
        "s32i       %[rs], %[req], 128\n"
        "addi.n     %[m], %[m], -1\n"           /* (1 << rs) - 1 */
        "s32i       %[ls], %[req], 256\n"
        "srai       %[t], %[m], 1\n"
        "s32i.n     %[m], %[req], 0\n"
        "s32i       %[t], %[req], 64\n"
        "l32i.n     %[t], %[bias], 0\n"
        "add.n      %[sum], %[sum], %[t]\n"     /* bias + input offset * filter sum */
        "s32i       %[sum], %[req], 320\n"
        /* sum >> 29 is 0 or -1 inside [-2^29, 2^29). Plus 1, then >> 1
         * (logical) is 0 for those and nonzero for the rest. */
        "srai       %[t], %[sum], 29\n"
        "addi.n     %[t], %[t], 1\n"
        "srli       %[t], %[t], 1\n"
        "or         %[gen], %[gen], %[ls]\n"
        "or         %[gen], %[gen], %[t]\n"
        "addi.n     %[req], %[req], 4\n"
        "addi.n     %[mult], %[mult], 4\n"
        "addi.n     %[shift], %[shift], 4\n"
        "addi.n     %[bias], %[bias], 4\n"
        "1:\n"
        : [gen] "=&r" (general), [ls] "=&r" (ls), [rs] "=&r" (rs), [sum] "=&r" (sum),
          [m] "=&r" (m), [t] "=&r" (t),
          [req] "+r" (req), [mult] "+r" (mult), [shift] "+r" (shift), [bias] "+r" (bias),
          [n] "+r" (n)
        : [off] "r" (input_offset)
        : "memory");
    return general;
}

/* ----------------------------------------------------------- C driver -- */

typedef struct {
    const int8_t *input;
    int input_wd, input_ht, channels, pad_wd, pad_ht, stride_ht, row_bytes;
    int8_t pad_val;
    const uint8_t *req_general; /* per 16-channel group: some channel shifts left or has a large bias */
    const int8_t *filter;
    const int32_t *req;
    int8_t *out_data;
    int y_begin, y_end;
    int8_t *ring;   /* fh padded rows */
    int8_t *tmp;    /* out_wd * 16 staging bytes */
    int cs, kw, kh, x_step, out_wd;
    int32_t out_offset, clamp_min, clamp_max;
} dw_kxk_job_t;

/* Output rows [y_begin, y_end). Each input row is copied once into a ring of
 * fh padded rows; the kernel gets the fh row pointers it needs. */
static void kxk_rows(dw_kxk_job_t *j)
{
    const int channels = j->channels, cs = j->cs, fh = j->kh;
    const int out_wd = j->out_wd;
    /* The kernel stores 16 bytes at a 16-aligned output, or 8 bytes at a
     * time at an 8-aligned one (channels is a multiple of 8). Other outputs
     * go through tmp. */
    const int direct16 = (cs == channels) && (((uintptr_t)j->out_data & 15) == 0);
    const int direct = direct16 || (((uintptr_t)j->out_data & 7) == 0);

    /* The pad columns of every ring row stay as filled here. The input part
     * is rewritten by every copy, or by a pad-row memset. */
    const int left = j->pad_wd * cs, in_bytes = j->input_wd * cs;
    for (int k = 0; k < fh; k++) {
        int8_t *row = j->ring + k * j->row_bytes;
        memset(row, j->pad_val, left);
        memset(row + left + in_bytes, j->pad_val, j->row_bytes - left - in_bytes);
    }

    const int8_t *rows[ESP_NN_DW_S8_KXK_MAX_TAPS];
    kxk_row_t r = {
        .cs = cs,
        .rows = rows,
        .kw = j->kw,
        .kh = fh,
        .x_step = j->x_step,
        .n_out = out_wd,
        .hback = 0,
        .hstep2 = 0,
        .hback2 = 32,
        .out_offset = j->out_offset,
        .clamp_min = j->clamp_min,
        .clamp_max = j->clamp_max,
        .variant = j->kw == fh && (fh == 5 || fh == 3) ? fh : 0,
    };

    int next = 0; /* first padded row not yet in the ring */
    for (int out_y = j->y_begin; out_y < j->y_end; out_y++) {
        const int r0 = out_y * j->stride_ht;
        for (int k = 0; k < fh; k++) {
            const int pr = r0 + k;
            const int y = pr - j->pad_ht;
            int8_t *dst = j->ring + (pr % fh) * j->row_bytes;
            rows[k] = dst;
            if (pr < next) {
                continue;
            }
            if (y < 0 || y >= j->input_ht) {
                /* the pad columns already hold pad_val */
                memset(dst + left, j->pad_val, in_bytes);
                continue;
            }
            const int8_t *src = j->input + y * j->input_wd * channels;
            if (((uintptr_t)src & 7) == 0) {
                /* 16 bytes per two 8-byte loads. A whole row is one
                 * "pixel" when cs == channels. */
                const int whole = cs == channels;
                kxk_copy(dst + j->pad_wd * cs, src, whole ? 1 : j->input_wd,
                         whole ? in_bytes : channels, cs);
            } else if (cs == channels) {
                memcpy(dst + j->pad_wd * cs, src, j->input_wd * channels);
            } else {
                /* Lanes past channels have zero weights; any value works. */
                for (int x = 0; x < j->input_wd; x++) {
                    memcpy(dst + (j->pad_wd + x) * cs, src + x * channels, channels);
                }
            }
        }
        next = r0 + fh;

        int8_t *out_row = j->out_data + out_y * out_wd * channels;
        for (int ch = 0; ch < cs; ch += r.g_left * 16) {
            const int n_ch = min(16, channels - ch);
            const int general = j->req_general[ch >> 4];
            /* Direct stores run the full groups that share a requant in one
             * call. The 8-channel tail group has its own req_ctl and store. */
            int ch_end = ch + 16;
            if (direct && n_ch == 16) {
                while (ch_end < (channels & ~15) && j->req_general[ch_end >> 4] == general) {
                    ch_end += 16;
                }
            }
            r.g_left = (ch_end - ch) >> 4;
            r.g_off = ch;
            r.g_req = j->req + (ch >> 4) * KXK_REQ_BLOCK;
            r.filter = j->filter + ch;
            /* an 8-channel group requants lanes 0-7 only */
            r.req_ctl = (n_ch > 8 ? 4 : 2) | (general ? 1 << 8 : 0);
            r.hstep = n_ch > 8 ? 32 : 0;
            if (direct) {
                r.out = out_row + ch;
                r.out_step = channels;
                r.store = direct16 ? 0 : (n_ch == 16 ? 1 : 2);
            } else {
                r.out = j->tmp;
                r.out_step = 16;
                r.store = 0;
            }
            r.g_out = r.out;
            const int groups = r.g_left;
            kxk_row(&r);
            /* the kernel counts g_left down to 1 */
            r.g_left = groups;
            if (!direct) {
                for (int x = 0; x < out_wd; x++) {
                    memcpy(out_row + x * channels + ch, j->tmp + x * 16, n_ch);
                }
            }
        }
    }
}

/* Fills the requant blocks and, per 16-channel group, whether it takes the
 * general requant. The fast requant takes 2 * (sum + bias), so a fast
 * group's bias row holds 2 * bias. Choosing per group keeps a few channels
 * that shift left (int4 weights have larger scales) from slowing the whole
 * layer. */
static void kxk_req_build(int32_t *req, uint8_t *general, int channels, int cs,
                          const int32_t *bias, const quant_data_t *q, int32_t input_offset)
{
    static const int32_t zeros[16];
    for (int g = 0; g < cs; g += 16, req += KXK_REQ_BLOCK) {
        const int n = min(16, channels - g);
        general[g >> 4] = kxk_req_lanes(req, q->mult + g, q->shift + g,
                                        bias ? bias + g : zeros, n, input_offset) != 0;
        /* padding lanes: mult 0 and no shift */
        for (int k = n; k < 16; k++) {
            for (int r = 0; r < KXK_REQ_BLOCK; r += 16) {
                req[k + r] = 0;
            }
        }
        if (!general[g >> 4]) {
            for (int k = 0; k < 16; k++) {
                req[KXK_REQ_BIAS + k] *= 2;
            }
        }
    }
}

static void kxk_worker(void *arg)
{
    kxk_rows((dw_kxk_job_t *)arg);
}

void esp_nn_dw_s8_kxk_esp32s3(const data_dims_t *input_dims,
                              const int8_t *input_data,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const dw_conv_params_t *conv_params,
                              const quant_data_t *quant_data,
                              void *scratch)
{
    esp_nn_dw_s8_kxk_layout_t l;
    esp_nn_dw_s8_kxk_layout_esp32s3(input_dims, filter_dims, output_dims, conv_params, &l);
    const int channels = input_dims->channels;
    const int input_wd = input_dims->width, input_ht = input_dims->height;
    const int out_wd = output_dims->width, out_ht = output_dims->height;
    const int fw = filter_dims->width, fh = filter_dims->height;
    const int pad_wd = conv_params->padding.width, pad_ht = conv_params->padding.height;
    const int stride_wd = conv_params->stride.width, stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int cs = l.cs;

    int8_t *base = (int8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);
    int8_t *filt = base;
    int32_t *req = (int32_t *)(filt + l.filter_bytes);
    uint8_t *req_general = (uint8_t *)req + l.req_bytes;
    int8_t *tmp = (int8_t *)req_general + l.flag_bytes;
    /* the fh padded rows the current output row needs, in a ring */
    int8_t *ring = tmp + l.tmp_bytes;

    /* The kernel loads filter taps cs bytes apart, 16-byte aligned. TFLM
     * filters at channels % 16 == 0 already are, so they are used in place. */
    const int8_t *filter = filter_data;
    if (cs != channels || ((uintptr_t)filter_data & 15)) {
        for (int t = 0; t < l.taps; t++) {
            memcpy(filt + t * cs, filter_data + t * channels, channels);
            memset(filt + t * cs + channels, 0, cs - channels);
        }
        filter = filt;
    }

    /* Requant parameters, once per layer. Padding channels get mult 0.
     * The bias takes the offset: sum((q + off) * w) + b =
     * sum(q * w) + (b + off * sum(w)). */
    for (int c = 0; c < cs; c += 16) {
        /* filter sums into the bias slot, 16 lanes per QACC pass */
        esp_nn_qacc_lane_sum_s8_esp32s3(filter + c, cs, l.taps,
                                        req + (c >> 4) * KXK_REQ_BLOCK + KXK_REQ_BIAS);
    }
    /* The fast requant needs |x| < 2^30. The raw sum is below 2^19 (31 taps
     * of -128 * -128), so a bias within 2^29 is enough. */
    kxk_req_build(req, req_general, channels, cs, bias, quant_data, input_offset);

    dw_kxk_job_t job = {
        .input = input_data,
        .input_wd = input_wd,
        .input_ht = input_ht,
        .channels = channels,
        .pad_wd = pad_wd,
        .pad_ht = pad_ht,
        .stride_ht = stride_ht,
        .row_bytes = l.row_bytes,
        .pad_val = (int8_t)(-input_offset),
        .req_general = req_general,
        .filter = filter,
        .req = req,
        .out_data = out_data,
        .y_begin = 0,
        .y_end = out_ht,
        .ring = ring,
        .tmp = tmp,
        .cs = cs,
        .kw = fw,
        .kh = fh,
        .x_step = stride_wd * cs,
        .out_wd = out_wd,
        .out_offset = conv_params->out_offset,
        .clamp_min = conv_params->activation.min - conv_params->out_offset,
        .clamp_max = conv_params->activation.max - conv_params->out_offset,
    };

    /* When this call claims the dual-core worker, the worker takes the top
     * half of the output rows with its own ring. Filter and requant blocks
     * are shared. */
    if (out_ht >= 2 && esp_nn_dual_core_begin()) {
        int8_t *wbuf = esp_nn_dual_core_scratch(16 + l.tmp_bytes + l.ring_bytes);
        if (wbuf != NULL) {
            dw_kxk_job_t wjob = job;
            wjob.y_end = out_ht / 2;
            wjob.tmp = (int8_t *)(((uintptr_t)wbuf + 15) & ~(uintptr_t)15);
            wjob.ring = wjob.tmp + l.tmp_bytes;
            if (esp_nn_dual_core_run(kxk_worker, &wjob)) {
                job.y_begin = out_ht / 2;
                kxk_rows(&job);
                esp_nn_dual_core_wait();
                esp_nn_dual_core_end();
                return;
            }
        }
        esp_nn_dual_core_end();
    }
    kxk_rows(&job);
}
