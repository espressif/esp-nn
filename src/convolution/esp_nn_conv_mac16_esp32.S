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
 * ESP32 (MAC16) helpers for conv. Pure .S: the requant dots use all 14
 * address registers, stack slots and a call per output channel. Inline asm
 * would leave the compiler too few registers.
 */

/*
 * One s16 filter row against n_pix s16 input rows, with MAC16.
 * Based on the esp-dsp macro dotprod_s16_ae32 (dsps_dotprod_s16_m_ae32.S),
 * rescheduled so that every ldinc result is used 3 instructions later.
 * A use 2 instructions after the load stalls one cycle on the ESP32.
 *
 * void esp_nn_conv_dot_mac16_esp32(const int16_t *filter,   a2
 *                                  const int16_t *input,    a3
 *                                  int32_t *out,            a4
 *                                  int32_t n_pix,           a5 (>= 1)
 *                                  int32_t len4,            a6 (row length / 4, >= 1)
 *                                  int32_t bias)            a7
 *
 * Layout, with L = len4 * 4:
 *  - input: n_pix rows of L samples, word aligned, then 2 spare words.
 *  - filter: one zero sample, the L filter samples, then 3 spare samples.
 * The filter is one sample ahead of the input, so each m register meets
 * two different partners and a load can sit 3 instructions before its use.
 * The loop reads 2 words past the end of each row. Those values are not used.
 *
 * out[p] = bias + dot(filter, input row p), low 32 bits of ACC.
 * No stack slots are used.
 */

    .text
    .align  4
    .global esp_nn_conv_dot_mac16_esp32
    .type   esp_nn_conv_dot_mac16_esp32,@function

esp_nn_conv_dot_mac16_esp32:
    entry   a1, 32

    addi    a9, a3, -4          // input pointer, ldinc pre-increments
    addi    a2, a2, -4

1:  // pixel loop, two pixels per pass
    wsr.acclo a7                // ACCLO = bias. ACCHI does not affect ACCLO.
    mov     a8, a2              // filter pointer
    ldinc   m3, a8              // [0, f0]
    ldinc   m0, a9              // [x0, x1]
    ldinc   m2, a8              // [f1, f2]
    ldinc   m1, a9              // [x2, x3]
    loop    a6, 2f
    mula.dd.lh.ldinc m3, a8, m0, m3     // x0 * f0, load [f3, f4]
    mula.dd.hl.ldinc m0, a9, m0, m2     // x1 * f1, load [x4, x5]
    mula.dd.lh.ldinc m2, a8, m1, m2     // x2 * f2, load [f5, f6]
    mula.dd.hl.ldinc m1, a9, m1, m3     // x3 * f3, load [x6, x7]
2:
    addi    a9, a9, -8          // the loop read 2 words into the next row
    addi    a5, a5, -1
    rsr.acclo a11
    s32i    a11, a4, 0
    addi    a4, a4, 4
    beqz    a5, 4f
    wsr.acclo a7                // ACCLO = bias. ACCHI does not affect ACCLO.
    mov     a8, a2              // filter pointer
    ldinc   m3, a8              // [0, f0]
    ldinc   m0, a9              // [x0, x1]
    ldinc   m2, a8              // [f1, f2]
    ldinc   m1, a9              // [x2, x3]
    loop    a6, 3f
    mula.dd.lh.ldinc m3, a8, m0, m3     // x0 * f0, load [f3, f4]
    mula.dd.hl.ldinc m0, a9, m0, m2     // x1 * f1, load [x4, x5]
    mula.dd.lh.ldinc m2, a8, m1, m2     // x2 * f2, load [f5, f6]
    mula.dd.hl.ldinc m1, a9, m1, m3     // x3 * f3, load [x6, x7]
3:
    addi    a9, a9, -8          // the loop read 2 words into the next row
    addi    a5, a5, -1
    rsr.acclo a11
    s32i    a11, a4, 0
    addi    a4, a4, 4
    bnez    a5, 1b
4:
    retw.n

    .size   esp_nn_conv_dot_mac16_esp32, . - esp_nn_conv_dot_mac16_esp32

/*
 * The dot above with the requant fused in, for the common case: right shift
 * >= 1 and activation range [-128, 127]. Each sum is requantized and stored
 * as s8 right after its ACC readout. The multiply latency hides behind the
 * preloads of the next pixel, and no int32 sums go through memory.
 *
 * void esp_nn_conv_dot_rq_mac16_esp32(const int16_t *filter,   a2
 *                                     const int16_t *input,    a3
 *                                     int8_t *out,             a4
 *                                     int32_t n_pix,           a5 (>= 1)
 *                                     int32_t len4,            a6 (>= 1)
 *                                     const int32_t *p)        a7
 *
 * p[] = { bias, mult (>= 0), n + 1, k2, out_step }, with right shift n in
 * [1, 20] and k2 = 1 + 2^n + out_offset * 2^(n+1).
 * Same layout rules as esp_nn_conv_dot_mac16_esp32. The loop stops one step
 * early and the last 4 MACs load no input, so the input pointer ends on the
 * next row. The preload after the last pixel reads the 2 spare words after
 * the band.
 *
 * Requant, exact, with hi:lo = x * mult:
 *  - r = (x * mult + 2^30) >> 31 = 2 * hi + (((lo >> 30) + 1) >> 1).
 *  - The rounding shift (half away from zero) plus out_offset is
 *    (r - (x < 0) + 2^(n-1) + out_offset * 2^n) >> n. Using x < 0 instead of
 *    r < 0 only differs when r == 0, and both give 0 then.
 *  - Both steps in one: (4 * hi + (lo >> 30) + k2 - 2 * (x < 0)) >> (n + 1).
 *  - The caller keeps |x| <= 2^29 and out_offset in [-128, 127],
 *    so |4 * hi| <= 2^30 and |k2| < 2^29.
 *    The sum does not overflow.
 * No stack slots are used. a3 is a temp after the setup.
 */
    .align  4
    .global esp_nn_conv_dot_rq_mac16_esp32
    .type   esp_nn_conv_dot_rq_mac16_esp32,@function

esp_nn_conv_dot_rq_mac16_esp32:
    entry   a1, 32

    l32i    a10, a7, 4          // mult
    l32i    a11, a7, 8          // n + 1
    l32i    a13, a7, 12         // k2
    l32i    a14, a7, 16         // out_step
    l32i    a7, a7, 0           // bias
    ssr     a11
    addi    a9, a3, -4          // input pointer, ldinc pre-increments
    addi    a2, a2, -4
    addi    a6, a6, -1          // the loop leaves the last 4 samples out

    wsr.acclo a7                // ACCLO = bias. ACCHI does not affect ACCLO.
    mov     a8, a2              // filter pointer
    ldinc   m3, a8              // [0, f0]
    ldinc   m0, a9              // [x0, x1]
    ldinc   m2, a8              // [f1, f2]
    ldinc   m1, a9              // [x2, x3]
    bbsi    a5, 0, 3f           // odd n: start with the second half
1:  // pixel loop, two pixels per pass
    loopnez a6, 2f
    mula.dd.lh.ldinc m3, a8, m0, m3     // x0 * f0, load [f3, f4]
    mula.dd.hl.ldinc m0, a9, m0, m2     // x1 * f1, load [x4, x5]
    mula.dd.lh.ldinc m2, a8, m1, m2     // x2 * f2, load [f5, f6]
    mula.dd.hl.ldinc m1, a9, m1, m3     // x3 * f3, load [x6, x7]
2:
    mula.dd.lh.ldinc m3, a8, m0, m3     // the last 4 samples, no input loads
    mula.dd.hl m0, m2
    mula.dd.lh m1, m2
    mula.dd.hl m1, m3
    rsr.acclo a11               // x
    mov     a8, a2              // next pixel
    wsr.acclo a7
    mull    a12, a11, a10       // lo, 3 after the ACC read: closer stalls
    srai    a3, a11, 31         // -(x < 0)
    mulsh   a11, a11, a10       // hi
    ldinc   m3, a8
    srli    a12, a12, 30
    ldinc   m0, a9
    addx2   a3, a3, a13         // k2 - 2 * (x < 0)
    ldinc   m2, a8
    addx4   a11, a11, a12       // 4 * hi + (lo >> 30)
    ldinc   m1, a9
    add     a11, a11, a3
    sra     a11, a11            // >> (n + 1)
    clamps  a11, a11, 7
    addi    a5, a5, -1
    s8i     a11, a4, 0
    add     a4, a4, a14
3:
    loopnez a6, 5f
    mula.dd.lh.ldinc m3, a8, m0, m3     // x0 * f0, load [f3, f4]
    mula.dd.hl.ldinc m0, a9, m0, m2     // x1 * f1, load [x4, x5]
    mula.dd.lh.ldinc m2, a8, m1, m2     // x2 * f2, load [f5, f6]
    mula.dd.hl.ldinc m1, a9, m1, m3     // x3 * f3, load [x6, x7]
5:
    mula.dd.lh.ldinc m3, a8, m0, m3     // the last 4 samples, no input loads
    mula.dd.hl m0, m2
    mula.dd.lh m1, m2
    mula.dd.hl m1, m3
    rsr.acclo a11               // x
    mov     a8, a2              // next pixel
    wsr.acclo a7
    mull    a12, a11, a10       // lo, 3 after the ACC read: closer stalls
    srai    a3, a11, 31         // -(x < 0)
    mulsh   a11, a11, a10       // hi
    ldinc   m3, a8
    srli    a12, a12, 30
    ldinc   m0, a9
    addx2   a3, a3, a13         // k2 - 2 * (x < 0)
    ldinc   m2, a8
    addx4   a11, a11, a12       // 4 * hi + (lo >> 30)
    ldinc   m1, a9
    add     a11, a11, a3
    sra     a11, a11            // >> (n + 1)
    clamps  a11, a11, 7
    addi    a5, a5, -1
    s8i     a11, a4, 0
    add     a4, a4, a14
    bnez    a5, 1b
    retw.n

    .size   esp_nn_conv_dot_rq_mac16_esp32, . - esp_nn_conv_dot_rq_mac16_esp32

/* The 4 MACs of one pass of the lanes MAC loop. */
    .macro  rq256s_mac4
    mula.dd.lh.ldinc m3, a8, m0, m3     // f0 * x0, load [x3, x4]
    mula.dd.hl.ldinc m0, a9, m0, m2     // f1 * x1, load [f4, f5]
    mula.dd.lh.ldinc m2, a8, m1, m2     // f2 * x2, load [x5, x6]
    mula.dd.hl.ldinc m1, a9, m1, m3     // f3 * x3, load [f6, f7]
    .endm

/* Requant and store for out_offset -128, with no pixel count. */
    .macro  rq256s_store_m128
    rsr.acclo a11
    sub     a9, a9, a2
    wsr.acclo a7
    mulsh   a12, a11, a10       // 3 after the ACC read
    add     a4, a4, a14
    add     a12, a12, a13
    sra     a12, a12
    clamps  a12, a12, 7
    s8i     a12, a4, 0
    .endm

/*
 * The fused dot of esp_nn_conv_rq256s_oc_esp32, for a filter widened as
 * f * 256 by esp_nn_widen_lanes_esp32 (in_ch a multiple of 4).
 *
 * Layout, with L = len4 * 4. The roles are swapped from the dots above.
 *  - filter: L samples, then its first 4 samples again.
 *  - input: the band starts one sample after a3, and the rows follow each
 *    other with no gap. The sample before each row is not used.
 * The MAC loop always loads 4 words ahead. Its last pass loads the word
 * [x(L-1), x0 of the next row] and the next pixel's first words: the input
 * runs on, and the filter copy gives f0..f3 again. So there are no
 * preloads between pixels, only a filter rewind. The loads after the last
 * pixel read the 2 spare words after the band.
 *
 * Requant: ACC holds A = 256 * x, and one mulsh gives H = (A * mult) >> 32
 * = floor(x * mult / 2^24). The requant of esp_nn_conv_dot_rq_mac16_esp32 is
 * (Q + k2 - 2s) >> (n + 1) with Q = floor(x * mult / 2^30) = floor(H / 64),
 * so it becomes (H + K - 128 * s) >> (n + 7), with K = 64 * k2 =
 * 64 + 2^(n+6) + out_offset * 2^(n+7). No mull and no low word.
 * The caller keeps |x| < 2^23, so A fits in ACCLO, |H| < 2^30 and |K| < 2^30.
 */

/*
 * The loop over the output channels for the fused dot above. Per channel it
 * widens the filter row, appends its first 4 samples and builds the requant
 * values. A C loop with two calls per channel costs much more, which shows
 * on layers with few pixels.
 *
 * With out_offset -128 (ReLU, ReLU6) the sign term is not needed. For
 * x < 0 both (H + K) >> (n + 7) and the exact value are <= -128, so clamps
 * gives -128 either way. Those loops drop 2 instructions per pixel.
 *
 * void esp_nn_conv_rq256s_oc_esp32(int16_t *filter,       a2 (word aligned, row scratch)
 *                                  const int16_t *input,  a3 (as for the dot)
 *                                  int8_t *out,           a4 (output channel 0)
 *                                  int32_t n_pix,         a5 (>= 1)
 *                                  int32_t len4,          a6 (>= 1)
 *                                  const int32_t *q)      a7
 *
 * q[] = { s8 filter, bias (never NULL), bias step (4 or 0), mult, shift,
 *         out_ch (>= 1), out_offset }. Every shift is in [-15, -1], every
 * bias keeps |x| < 2^23 and out_offset is in [-128, 127]. The caller checks.
 *
 * Stack slots (entry a1, 112): 0 filter row, 4 input, 8 out, 12 n_pix,
 * 16 s8 filter, 20 bias, 24 bias step, 28 mult, 32 shift, 36 channels left,
 * 40 len4, 44 out step, 48 out_offset. 52..60 are free.
 * The pixel loops use the registers of the dot above: a2 filter rewind,
 * a4 out, a7 256 * bias, a8 input, a9 filter, a10 mult, a13 K, a14 out step.
 */
    .align  4
    .global esp_nn_conv_rq256s_oc_esp32
    .type   esp_nn_conv_rq256s_oc_esp32,@function

esp_nn_conv_rq256s_oc_esp32:
    entry   a1, 112

    s32i    a2, a1, 0
    s32i    a3, a1, 4
    s32i    a4, a1, 8
    s32i    a5, a1, 12
    s32i    a6, a1, 40
    l32i    a8, a7, 0
    l32i    a9, a7, 4
    l32i    a10, a7, 8
    l32i    a11, a7, 12
    l32i    a12, a7, 16
    l32i    a13, a7, 20
    s32i    a8, a1, 16
    s32i    a9, a1, 20
    s32i    a10, a1, 24
    s32i    a11, a1, 28
    s32i    a12, a1, 32
    l32i    a8, a7, 24
    s32i    a13, a1, 36
    s32i    a13, a1, 44
    s32i    a8, a1, 48
20: // channel loop
    l32i    a10, a1, 0
    l32i    a11, a1, 16
    l32i    a12, a1, 40
    call8   esp_nn_widen_lanes_esp32
    l32i    a9, a1, 0           // filter row
    l32i    a6, a1, 40          // len4
    l32i    a3, a1, 16
    slli    a2, a6, 3           // filter rewind per pixel (bytes)
    add     a11, a9, a2
    srli    a12, a2, 1          // s8 filter row (bytes)
    l32i    a14, a9, 0          // the first 4 samples again after the row
    l32i    a15, a9, 4
    add     a3, a3, a12
    s32i    a14, a11, 0
    s32i    a15, a11, 4
    s32i    a3, a1, 16
    l32i    a3, a1, 20          // bias
    l32i    a12, a1, 24
    l32i    a7, a3, 0
    add     a3, a3, a12
    s32i    a3, a1, 20
    l32i    a3, a1, 28          // mult
    l32i    a12, a1, 32         // shift
    l32i    a10, a3, 0
    l32i    a15, a12, 0
    addi    a3, a3, 4
    addi    a12, a12, 4
    s32i    a3, a1, 28
    s32i    a12, a1, 32
    movi    a3, 6
    sub     a12, a3, a15        // n + 6
    ssl     a12
    movi    a3, 1
    l32i    a15, a1, 48         // out_offset
    sll     a3, a3              // 2^(n+6)
    sll     a13, a15            // out_offset * 2^(n+6)
    addi    a3, a3, 64
    addx2   a13, a13, a3        // K
    addi    a12, a12, 1
    ssr     a12                 // >> (n + 7)
    slli    a7, a7, 8           // 256 * bias
    l32i    a4, a1, 8
    l32i    a14, a1, 44
    l32i    a5, a1, 12
    addi    a3, a4, 1
    s32i    a3, a1, 8
    sub     a4, a4, a14         // the store pointer moves before each store
    l32i    a8, a1, 4
    addi    a9, a9, -4          // filter pointer
    addi    a8, a8, -4          // input pointer, ldinc pre-increments
    wsr.acclo a7
    ldinc   m3, a8              // [-, x0]
    ldinc   m0, a9              // [f0, f1]
    ldinc   m2, a8              // [x1, x2]
    ldinc   m1, a9              // [f2, f3]
    addi    a3, a15, 128
    addi    a15, a13, -128      // K - 128
    bnez    a3, 25f             // out_offset is not -128: sign term
    mov     a15, a13            // no sign term: the loop at 25 picks K either way
    beqi    a6, 2, 10f
    beqi    a6, 4, 12f
    beqi    a6, 8, 14f
    beqi    a6, 16, 16f
    beqi    a6, 32, 18f
    j       25f

    /*
     * Rows of 8 to 128 samples: the MACs are unrolled, so the pixel loop can
     * be the zero-overhead loop. A loop setup and a branch per pixel cost
     * about 5 cycles, which is a lot next to 8 to 128 MACs.
     */
10:
    loop    a5, 11f
    .rept   2
    rq256s_mac4
    .endr
    rq256s_store_m128
11:
    j       30f
12:
    loop    a5, 13f
    .rept   4
    rq256s_mac4
    .endr
    rq256s_store_m128
13:
    j       30f
14:
    loop    a5, 15f
    .rept   8
    rq256s_mac4
    .endr
    rq256s_store_m128
15:
    j       30f
16:
    loop    a5, 17f
    .rept   16
    rq256s_mac4
    .endr
    rq256s_store_m128
17:
    j       30f
18:
    loop    a5, 19f
    .rept   32
    rq256s_mac4
    .endr
    rq256s_store_m128
19:
    j       30f

25: // other row lengths, or any out_offset: one pixel per pass
    loopnez a6, 2f
    rq256s_mac4
2:
    rsr.acclo a11               // A
    sub     a9, a9, a2          // back to [f2, f3] of the row
    mov     a3, a13
    wsr.acclo a7
    mulsh   a12, a11, a10       // H, 3 after the ACC read: closer stalls
    movltz  a3, a15, a11        // K - 128 * (A < 0)
    add     a4, a4, a14
    addi    a5, a5, -1
    add     a12, a12, a3
    sra     a12, a12            // >> (n + 7)
    clamps  a12, a12, 7
    s8i     a12, a4, 0
    bnez    a5, 25b
30:
    l32i    a3, a1, 36
    addi    a3, a3, -1
    s32i    a3, a1, 36
    bnez    a3, 20b
    retw.n

    .size   esp_nn_conv_rq256s_oc_esp32, . - esp_nn_conv_rq256s_oc_esp32

/*
 * esp_nn_conv_rq256s_oc_esp32 for rows past 256 samples, where
 * A = 256 * x needs the 40-bit ACC. Same layout and arguments, with
 * p[] = { (256 * bias) low word, mult, n + 1, k2, out_step,
 *         (256 * bias) >> 32, act_min, act_max },
 * n in [1, 20] and k2 = 1 + 2^n + out_offset * 2^(n+1).
 *
 * The caller keeps |x| < 2^29, so A >> 6 = 4x fits a word. mulsh(4x, mult)
 * is Q = floor(x * mult / 2^30), and the requant is the one of
 * esp_nn_conv_dot_rq_mac16_esp32: (Q + k2 - 2s) >> (n + 1), s = (x < 0).
 * It clamps to any activation range. The rows are long, so the few extra
 * instructions per pixel do not matter.
 * Slot 0: act_max.
 */
    .align  4
    .global esp_nn_conv_dot_rq256w_mac16_esp32
    .type   esp_nn_conv_dot_rq256w_mac16_esp32,@function

esp_nn_conv_dot_rq256w_mac16_esp32:
    entry   a1, 64

    l32i    a10, a7, 4          // mult
    l32i    a11, a7, 8          // n + 1
    l32i    a13, a7, 12         // k2
    l32i    a14, a7, 16         // out_step
    l32i    a15, a7, 20         // ACCHI seed
    l32i    a12, a7, 28         // act_max
    addi    a8, a3, -4          // input pointer, ldinc pre-increments
    l32i    a3, a7, 24          // act_min
    l32i    a7, a7, 0           // ACCLO seed
    ssr     a11
    s32i    a12, a1, 0
    addi    a9, a2, -4          // filter pointer
    slli    a2, a6, 3           // filter rewind per pixel (bytes)
    sub     a4, a4, a14         // the store pointer moves before each store

    wsr.acclo a7
    wsr.acchi a15
    ldinc   m3, a8              // [-, x0]
    ldinc   m0, a9              // [f0, f1]
    ldinc   m2, a8              // [x1, x2]
    ldinc   m1, a9              // [f2, f3]
1:
    loopnez a6, 2f
    rq256s_mac4
2:
    rsr.acclo a11
    rsr.acchi a12
    sub     a9, a9, a2          // back to [f2, f3] of the row
    wsr.acclo a7
    wsr.acchi a15
    srli    a11, a11, 6
    slli    a12, a12, 26        // ACC bit 37 is the sign
    or      a11, a11, a12       // 4x = A >> 6
    mulsh   a12, a11, a10       // Q
    add     a4, a4, a14
    srai    a11, a11, 31        // -s
    add     a12, a12, a13
    addx2   a12, a11, a12       // Q + k2 - 2s
    l32i    a11, a1, 0          // act_max
    sra     a12, a12            // >> (n + 1)
    max     a12, a12, a3
    min     a12, a12, a11
    addi    a5, a5, -1
    s8i     a12, a4, 0
    bnez    a5, 1b
    retw.n

    .size   esp_nn_conv_dot_rq256w_mac16_esp32, . - esp_nn_conv_dot_rq256w_mac16_esp32

/*
 * void esp_nn_widen_lanes_esp32(int16_t *dst,     a2 (word aligned)
 *                               const int8_t *src, a3 (word aligned)
 *                               int32_t n4)        a4 (words, >= 1)
 *
 * Widen 4 bytes [b0, b1, b2, b3] to [b0, b2, b1, b3] * 256 with two masks:
 * a byte in the high half of an s16 is that byte times 256, sign included.
 * The input rows must use the same order, see esp_nn_widen_perm_esp32.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_widen_lanes_esp32
    .type   esp_nn_widen_lanes_esp32,@function

esp_nn_widen_lanes_esp32:
    entry   a1, 32

    movi    a5, 0xff00ff00
    srli    a6, a4, 2
    loopnez a6, 1f              // 4 words per pass: fewer pointer adds
    l32i    a8, a3, 0
    l32i    a9, a3, 4
    l32i    a10, a3, 8
    l32i    a11, a3, 12
    slli    a12, a8, 8
    and     a8, a8, a5          // [b1, b3]
    and     a12, a12, a5        // [b0, b2]
    slli    a13, a9, 8
    s32i    a12, a2, 0
    and     a9, a9, a5
    s32i    a8, a2, 4
    and     a13, a13, a5
    slli    a12, a10, 8
    s32i    a13, a2, 8
    and     a10, a10, a5
    s32i    a9, a2, 12
    and     a12, a12, a5
    slli    a13, a11, 8
    s32i    a12, a2, 16
    and     a11, a11, a5
    s32i    a10, a2, 20
    and     a13, a13, a5
    s32i    a13, a2, 24
    addi    a3, a3, 16
    s32i    a11, a2, 28
    addi    a2, a2, 32
1:
    bbci    a4, 1, 3f
    l32i    a8, a3, 0
    l32i    a9, a3, 4
    addi    a3, a3, 8
    slli    a10, a8, 8
    and     a8, a8, a5
    slli    a11, a9, 8
    and     a9, a9, a5
    and     a10, a10, a5
    and     a11, a11, a5
    s32i    a10, a2, 0
    s32i    a8, a2, 4
    s32i    a11, a2, 8
    s32i    a9, a2, 12
    addi    a2, a2, 16
3:
    bbci    a4, 0, 2f
    l32i    a8, a3, 0
    slli    a10, a8, 8
    and     a8, a8, a5
    and     a10, a10, a5
    s32i    a10, a2, 0
    s32i    a8, a2, 4
2:
    retw.n

    .size   esp_nn_widen_lanes_esp32, . - esp_nn_widen_lanes_esp32

/*
 * void esp_nn_widen_perm_esp32(int16_t *dst,     a2 (2-byte aligned)
 *                              const int8_t *src, a3 (any alignment)
 *                              int32_t n,         a4 (multiple of 4)
 *                              int32_t offset)    a5
 *
 * esp_nn_widen_s8_s16_esp32 in the order of esp_nn_widen_lanes_esp32:
 * dst[4k..4k+3] = [s0, s2, s1, s3] + offset.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_widen_perm_esp32
    .type   esp_nn_widen_perm_esp32,@function

esp_nn_widen_perm_esp32:
    entry   a1, 32

    srli    a6, a4, 2
    loopnez a6, 1f
    l8ui    a8, a3, 0
    l8ui    a9, a3, 1
    l8ui    a10, a3, 2
    l8ui    a11, a3, 3
    sext    a8, a8, 7
    sext    a9, a9, 7
    sext    a10, a10, 7
    sext    a11, a11, 7
    add     a8, a8, a5
    add     a9, a9, a5
    add     a10, a10, a5
    add     a11, a11, a5
    s16i    a8, a2, 0
    s16i    a9, a2, 4
    s16i    a10, a2, 2
    s16i    a11, a2, 6
    addi    a3, a3, 4
    addi    a2, a2, 8
1:
    retw.n

    .size   esp_nn_widen_perm_esp32, . - esp_nn_widen_perm_esp32

/*
 * esp_nn_widen_perm_esp32 for offset 128 and a word aligned source.
 *
 * void esp_nn_widen_u8_esp32(int16_t *dst,     a2 (odd sample: dst % 4 == 2)
 *                            const int8_t *src, a3 (word aligned)
 *                            int32_t n,         a4 (multiple of 4)
 *                            int32_t offset)    a5 (128, not used)
 *
 * s + 128 is the byte s ^ 0x80 read as unsigned. Masks split a word into
 * P = [s0, s2] and Q = [s1, s3] + 128. dst starts one sample after a word
 * boundary, so each stored word is (next:prev) >> 16, one src per word.
 * The first and last samples are halfword stores: the samples next to them
 * belong to other rows.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_widen_u8_esp32
    .type   esp_nn_widen_u8_esp32,@function

esp_nn_widen_u8_esp32:
    entry   a1, 32

    srli    a7, a4, 2
    beqz    a7, 2f
    movi    a5, 0x80808080
    movi    a6, 0x00ff00ff
    ssai    16
    l32i    a8, a3, 0
    addi    a3, a3, 4
    addi    a7, a7, -1
    xor     a8, a8, a5
    srli    a9, a8, 8
    and     a8, a8, a6          // P
    and     a9, a9, a6          // Q
    s16i    a8, a2, 0           // s0
    addi    a2, a2, 2           // word aligned
    src     a10, a9, a8         // [s2, s1]
    s32i    a10, a2, 0
    srli    a4, a7, 1
    loopnez a4, 1f              // two words per pass: no Q copy
    l32i    a8, a3, 0
    l32i    a13, a3, 4
    addi    a3, a3, 8
    xor     a8, a8, a5
    xor     a13, a13, a5
    srli    a11, a8, 8
    and     a8, a8, a6          // P
    and     a11, a11, a6        // Q
    src     a10, a8, a9         // [last Q.h, s0]
    src     a12, a11, a8        // [s2, s1]
    s32i    a10, a2, 4
    s32i    a12, a2, 8
    srli    a9, a13, 8
    and     a13, a13, a6        // P of the second word
    and     a9, a9, a6          // its Q, the last Q for the next pass
    src     a10, a13, a11
    src     a12, a9, a13
    s32i    a10, a2, 12
    s32i    a12, a2, 16
    addi    a2, a2, 16
1:
    bbci    a7, 0, 3f
    l32i    a8, a3, 0
    xor     a8, a8, a5
    srli    a11, a8, 8
    and     a8, a8, a6
    and     a11, a11, a6
    src     a10, a8, a9
    src     a12, a11, a8
    mov     a9, a11
    s32i    a10, a2, 4
    s32i    a12, a2, 8
    addi    a2, a2, 8
3:
    srli    a9, a9, 16
    s16i    a9, a2, 4           // the last s3
2:
    retw.n

    .size   esp_nn_widen_u8_esp32, . - esp_nn_widen_u8_esp32

/*
 * void esp_nn_widen_s8_s16_esp32(int16_t *dst,     a2 (2-byte aligned)
 *                                const int8_t *src, a3 (any alignment)
 *                                int32_t n,         a4 (>= 0)
 *                                int32_t offset)    a5
 *
 * dst[i] = src[i] + offset. Four bytes per step, each byte loaded 4
 * instructions before its first use to avoid load-use stalls.
 * The loop without the add is for filters (offset 0).
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_widen_s8_s16_esp32
    .type   esp_nn_widen_s8_s16_esp32,@function

esp_nn_widen_s8_s16_esp32:
    entry   a1, 32

    srli    a6, a4, 2
    beqz    a5, 3f
    loopnez a6, 4f
    l8ui    a8, a3, 0
    l8ui    a9, a3, 1
    l8ui    a10, a3, 2
    l8ui    a11, a3, 3
    sext    a8, a8, 7
    sext    a9, a9, 7
    sext    a10, a10, 7
    sext    a11, a11, 7
    add     a8, a8, a5
    add     a9, a9, a5
    add     a10, a10, a5
    add     a11, a11, a5
    s16i    a8, a2, 0
    s16i    a9, a2, 2
    s16i    a10, a2, 4
    s16i    a11, a2, 6
    addi    a3, a3, 4
    addi    a2, a2, 8
4:
    j       1f
3:
    loopnez a6, 1f
    l8ui    a8, a3, 0
    l8ui    a9, a3, 1
    l8ui    a10, a3, 2
    l8ui    a11, a3, 3
    addi    a3, a3, 4
    sext    a8, a8, 7
    sext    a9, a9, 7
    sext    a10, a10, 7
    sext    a11, a11, 7
    s16i    a8, a2, 0
    s16i    a9, a2, 2
    s16i    a10, a2, 4
    s16i    a11, a2, 6
    addi    a2, a2, 8
1:
    extui   a6, a4, 0, 2
    loopnez a6, 2f
    l8ui    a8, a3, 0
    addi    a3, a3, 1
    sext    a8, a8, 7
    add     a8, a8, a5
    s16i    a8, a2, 0
    addi    a2, a2, 2
2:
    retw.n

    .size   esp_nn_widen_s8_s16_esp32, . - esp_nn_widen_s8_s16_esp32

/*
 * Requantize n int32 sums of one output channel and store them as s8.
 *
 * void esp_nn_conv_requant_store_esp32(const int32_t *sums,   a2
 *                                      int32_t n,             a3 (>= 1)
 *                                      int8_t *out,           a4
 *                                      int32_t out_step,      a5
 *                                      const int32_t *p)      a6
 *
 * p[] = { mult (>= 0), left_shift, right_shift, out_offset, act_min, act_max }
 * left_shift and right_shift are never both non-zero.
 *
 * Same result as esp_nn_multiply_by_quantized_mult():
 *  - with mult >= 0 the doubling high multiply is (x * mult + 2^30) >> 31,
 *    computed as 2 * hi + ((lo >> 1) + 2^29) >> 30 without a carry;
 *  - the rounding shift by n >= 1 (half away from zero) is
 *    (((r - (r < 0)) >> (n - 1)) + 1) >> 1. It cannot overflow: r <= 2^31 - 2.
 * No stack slots are used.
 */
    .align  4
    .global esp_nn_conv_requant_store_esp32
    .type   esp_nn_conv_requant_store_esp32,@function

esp_nn_conv_requant_store_esp32:
    entry   a1, 32

    l32i    a7, a6, 0           // mult
    l32i    a8, a6, 4           // left_shift
    l32i    a9, a6, 8           // right_shift
    l32i    a10, a6, 12         // out_offset
    l32i    a11, a6, 16         // act_min
    l32i    a12, a6, 20         // act_max
    movi    a15, 0x20000000
    beqz    a9, 3f              // shift >= 0: the left shift loop

    addi    a9, a9, -1
    ssr     a9
    movi    a9, -128
    bne     a11, a9, 1f
    movi    a9, 127
    beq     a12, a9, 5f         // full s8 range: clamps
1:
    loopnez a3, 2f
    l32i    a3, a2, 0
    addi    a2, a2, 4
    mull    a6, a3, a7
    mulsh   a3, a3, a7
    srli    a6, a6, 1
    add     a6, a6, a15
    srli    a6, a6, 30
    addx2   a3, a3, a6          // r = (x * mult + 2^30) >> 31
    extui   a8, a3, 31, 1       // r < 0
    sub     a3, a3, a8
    sra     a3, a3
    addi    a3, a3, 1
    srai    a3, a3, 1
    add     a3, a3, a10
    max     a3, a3, a11
    min     a3, a3, a12
    s8i     a3, a4, 0
    add     a4, a4, a5
2:
    retw.n

3:  // left shift (maybe 0), no right shift
    ssl     a8
    loopnez a3, 4f
    l32i    a3, a2, 0
    addi    a2, a2, 4
    sll     a3, a3
    mull    a6, a3, a7
    mulsh   a3, a3, a7
    srli    a6, a6, 1
    add     a6, a6, a15
    srli    a6, a6, 30
    addx2   a3, a3, a6
    add     a3, a3, a10
    max     a3, a3, a11
    min     a3, a3, a12
    s8i     a3, a4, 0
    add     a4, a4, a5
4:
    retw.n

5:  // right shift, activation range [-128, 127]: clamps, two sums per pass
    srli    a9, a3, 1
    extui   a13, a3, 0, 1       // one sum left over
    loopnez a9, 7f
    l32i    a3, a2, 0
    l32i    a11, a2, 4
    addi    a2, a2, 8
    mull    a6, a3, a7
    mulsh   a3, a3, a7
    mull    a12, a11, a7
    mulsh   a11, a11, a7
    srli    a6, a6, 1
    srli    a12, a12, 1
    add     a6, a6, a15
    add     a12, a12, a15
    srli    a6, a6, 30
    srli    a12, a12, 30
    addx2   a3, a3, a6
    addx2   a11, a11, a12
    extui   a8, a3, 31, 1
    extui   a9, a11, 31, 1
    sub     a3, a3, a8
    sub     a11, a11, a9
    sra     a3, a3
    sra     a11, a11
    addi    a3, a3, 1
    addi    a11, a11, 1
    srai    a3, a3, 1
    srai    a11, a11, 1
    add     a3, a3, a10
    add     a11, a11, a10
    clamps  a3, a3, 7
    clamps  a11, a11, 7
    s8i     a3, a4, 0
    add     a4, a4, a5
    s8i     a11, a4, 0
    add     a4, a4, a5
7:
    beqz    a13, 6f
    l32i    a3, a2, 0
    mull    a6, a3, a7
    mulsh   a3, a3, a7
    srli    a6, a6, 1
    add     a6, a6, a15
    srli    a6, a6, 30
    addx2   a3, a3, a6
    extui   a8, a3, 31, 1
    sub     a3, a3, a8
    sra     a3, a3
    addi    a3, a3, 1
    srai    a3, a3, 1
    add     a3, a3, a10
    clamps  a3, a3, 7
    s8i     a3, a4, 0
6:
    retw.n

    .size   esp_nn_conv_requant_store_esp32, . - esp_nn_conv_requant_store_esp32
