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

// Shared 8-lane int8 MUL step for the ESP32-S3 elementwise and broadcast
// kernels.
//
// Frame: each kernel sets MUL_S8_FRAME (its entry size) before using these
// macros. activation_min and activation_max are its stack arguments at
// a1 + MUL_S8_FRAME + 8 and + 12. The frame keeps:
//   a1+0..47: the kernel's scalar slots. a1+44 is scalar scratch for the
//   macros.
//   a1+48..271: the constant block, 16-byte aligned, at MUL_S8_CONST_PTR.
//   ee.vld.128.ip clears the low 4 address bits, so it must be aligned.
//   - MUL_S8_SETUP writes out_offset, activation_max and activation_min
//     there (48 bytes) for MUL_S8_STEP8.
//   - MUL_S8_SETUP_Q writes the 208-byte block for MUL_S8_STEP8_Q over it.
//     So a path that runs MUL_S8_STEP8_Q never falls back to MUL_S8_STEP8.
//   The top 16 bytes of the frame are the windowed-ABI base save area. The
//   window overflow handler writes there, so the kernels never do. The block
//   ends at a1+271. The kernels use 336, which leaves a1+272..319 free for new
//   slots without moving the stack arguments.
//
// Registers while the vector loop runs:
//   q0: zero              q1: input1_offset (s16)
//   q7: input2_offset (s16)
//   q6: K = 2^13 + (to_add << 14) (s32), or 1 in 8 s16 lanes after
//       MUL_S8_SETUP_Q
//   q2..q5: scratch
//
// The vector path needs right_shift <= 16:
//   - right_shift <= 16 keeps m + K below 2^31 (|m| < 2^30, K <= 2^29 + 2^13),
//     and 14 + right_shift <= 30 is a valid shift.
//   - right_shift >= 1 keeps the pre-offset result within s16, so the
//     offset and clamp run on 8 s16 lanes. right_shift = 0 uses rs0 = 1,
//     which saturates the result to s16 instead.
// A right_shift above 16 always gives 0, as |SRDHM(x, mult)| < 2^16. The
// callers run those with mult = 0 and right_shift = 15, which also give 0.

// \r = start of the 16-byte aligned constant block, (a1 + 63) & ~15.
.macro MUL_S8_CONST_PTR r
    addi            \r, a1, 63
    srli            \r, \r, 4
    slli            \r, \r, 4
.endm

// Build the loop constants. input1_offset, input2_offset and out_offset
// must already be saved at a1+24, a1+28 and a1+40.
// Clobbers q2, \t0, \t1 and a1+44.
.macro MUL_S8_SETUP rs, t0, t1
    // to_add = 1 << (right_shift - 1), K = 2^13 + (to_add << 14)
    movi.n          \t0, 1
    ssl             \rs
    sll             \t0, \t0
    srli            \t0, \t0, 1
    slli            \t1, \t0, 14
    addmi           \t1, \t1, 8192
    s32i            \t1, a1, 44

    ee.zero.q       q0
    addi            \t0, a1, 24
    ee.vldbc.16     q1, \t0
    addi            \t0, a1, 28
    ee.vldbc.16     q7, \t0
    MUL_S8_CONST_PTR \t1
    addi            \t0, a1, 40
    ee.vldbc.16     q2, \t0
    ee.vst.128.ip   q2, \t1, 16
    movi            \t0, MUL_S8_FRAME + 12   // activation_max
    add             \t0, \t0, a1
    ee.vldbc.16     q2, \t0
    ee.vst.128.ip   q2, \t1, 16
    addi            \t0, \t0, -4             // activation_min
    ee.vldbc.16     q2, \t0
    ee.vst.128.ip   q2, \t1, 0
    addi            \t0, a1, 44
    ee.vldbc.32     q6, \t0
.endm

// Multiply 8 int8 pairs and store 8 int8 results. Advances \p1, \p2 and \po
// by 8. All three must be 8-byte aligned. \rs14 holds right_shift + 14, or
// 16 with rs0 = 1. \t0..\t7 are scratch; eight of them let the mulsh
// latencies overlap.
// \cp points at the constant block (from MUL_S8_CONST_PTR) and is
// unchanged on exit.
.macro MUL_S8_STEP8 p1, p2, po, rs14, mult, t0, t1, t2, t3, t4, t5, t6, t7, rs0=0, cp
    ee.vld.l.64.ip  q2, \p1, 8
    ee.vld.h.64.ip  q2, \p2, 8
    ssai            16
    ee.vcmp.lt.s8   q5, q2, q0              // sign extend
    ee.vzip.8       q2, q5                  // q2: input1, q5: input2 (s16)
    ee.vadds.s16    q4, q2, q1
    ee.vadds.s16    q5, q5, q7
    ee.vmul.s16     q3, q4, q5              // high half of x
    ssai            0
    ee.vmul.s16     q2, q4, q5              // low half of x

    // |x| <= 255 * 255 < 2^16, so x << 15 fits in 32 bits and
    // m = mulsh(x << 15, mult) = floor(x * mult / 2^17) exactly. The SRDHM
    // nudge 2^30 is a multiple of 2^17, so SRDHM(x, mult) = (m + 2^13) >> 14.
    ssai            15
    ee.vzip.16      q2, q3
    ee.vsl.32       q2, q2
    ee.vsl.32       q3, q3

    ee.movi.32.a    q2, \t0, 0
    ee.movi.32.a    q2, \t1, 1
    ee.movi.32.a    q2, \t2, 2
    ee.movi.32.a    q2, \t3, 3
    ee.movi.32.a    q3, \t4, 0
    ee.movi.32.a    q3, \t5, 1
    ee.movi.32.a    q3, \t6, 2
    ee.movi.32.a    q3, \t7, 3
    mulsh           \t0, \mult, \t0
    mulsh           \t1, \mult, \t1
    mulsh           \t2, \mult, \t2
    mulsh           \t3, \mult, \t3
    mulsh           \t4, \mult, \t4
    mulsh           \t5, \mult, \t5
    mulsh           \t6, \mult, \t6
    mulsh           \t7, \mult, \t7

    // Rounding divide by 2^right_shift, half away from zero:
    // (v + to_add - (v < 0)) >> right_shift, with v = SRDHM(x, mult).
    // K folds to_add into the SRDHM rounding: (m + K) >> 14 = v + to_add.
    // (v < 0) may be replaced by (x < 0): mult >= 0, so x < 0 gives v <= 0,
    // and for v = 0 both give 0, as 0 <= to_add - 1 < 2^right_shift.
    // So out = (m + K - (x < 0) * 2^14) >> (14 + right_shift).
.if \rs0 == 0
    ee.vcmp.lt.s32  q4, q2, q0
    ee.vcmp.lt.s32  q5, q3, q0
    ssai            14
    ee.vsl.32       q4, q4
    ee.vsl.32       q5, q5
    ee.vadds.s32    q4, q4, q6
    ee.vadds.s32    q5, q5, q6
.endif
    ee.movi.32.q    q2, \t0, 0
    ee.movi.32.q    q2, \t1, 1
    ee.movi.32.q    q2, \t2, 2
    ee.movi.32.q    q2, \t3, 3
    ee.movi.32.q    q3, \t4, 0
    ee.movi.32.q    q3, \t5, 1
    ee.movi.32.q    q3, \t6, 2
    ee.movi.32.q    q3, \t7, 3

.if \rs0 == 0
    ee.vadds.s32    q2, q2, q4
    ee.vadds.s32    q3, q3, q5
.else
    // right_shift = 0: v = (m + 2^13) >> 14 can reach 65025, past s16.
    // K is 2^13 here. Saturating 4 * (m + K) and shifting by 16 gives v
    // saturated to s16. The offset and clamp to int8 then give the same
    // result as on the unsaturated v.
    ee.vadds.s32    q2, q2, q6
    ee.vadds.s32    q3, q3, q6
    ee.vadds.s32    q2, q2, q2
    ee.vadds.s32    q3, q3, q3
    ee.vadds.s32    q2, q2, q2
    ee.vadds.s32    q3, q3, q3
.endif
    ee.vld.128.ip   q5, \cp, 16             // out_offset
    wsr.sar         \rs14
    ee.vsr.32       q2, q2
    ee.vld.128.ip   q4, \cp, 16             // activation_max
    ee.vsr.32       q3, q3

    // Narrow to s16, then add the offset and clamp on 8 lanes at once
    ee.vunzip.16    q2, q3
    ee.vadds.s16    q2, q2, q5
    ee.vld.128.ip   q5, \cp, -32            // activation_min
    ee.vmin.s16     q2, q2, q4
    ee.vmax.s16     q2, q2, q5
    ee.vunzip.8     q2, q3
    ee.vst.l.64.ip  q2, \po, 8
.endm

// Broadcast the low 16 bits of \v to 8 lanes and store them at \cp
// (post-increment 16). \a must hold a1 + 44.
.macro MUL_S8_PUT16 v, cp, a
    s16i            \v, a1, 44
    ee.vldbc.16     q2, \a
    ee.vst.128.ip   q2, \cp, 16
.endm

// Constants for MUL_S8_STEP8_Q (1 <= right_shift <= 15). The 208-byte block
// starts at MUL_S8_CONST_PTR, so it overwrites the MUL_S8_SETUP block and the
// step must not mix with MUL_S8_STEP8. Expects MUL_S8_SETUP to have run.
// Leaves \cp at block + 32, q4 = input1_offset, q6 = 1 in 8 s16 lanes and
// SAR = 1.
// Block: amin, input1_offset, -2^15, 2*M0, M0, 2*M1, M1, 2*M2, M2,
// 2^(rs - 1), 8, out_offset, amax, with M0 = mult & 0x3fff,
// M1 = (mult >> 14) & 0x3fff and M2 = mult >> 28. The step loads them from
// block + 32 onwards, then amax, then wraps to amin and input1_offset.
//
// With \ls (a register holding a left shift of 1..11, and \rs holding 0),
// the step gives SRDHM(x << ls, mult) instead, for the final shift by 3:
// the pieces come from mult << ls, and both rounding constants are 0. For
// right_shift 0, SRDHM(x', mult) = floor((x' * mult + 2^30) / 2^31) for
// either sign of x', and the stages give exactly that. 2 * M2 < 2^(4 + ls)
// fits s16 up to ls = 11. A result past s16 saturates, which the offset and
// the clamp to int8 then absorb.
.macro MUL_S8_SETUP_Q rs, mult, cp, t0, t1, ls=
    MUL_S8_CONST_PTR \cp
    ee.vld.128.ip   q3, \cp, 16             // out_offset
    ee.vld.128.ip   q4, \cp, 16             // activation_max
    ee.vld.128.ip   q5, \cp, -32            // activation_min
    ee.vst.128.ip   q5, \cp, 16
    ee.vst.128.ip   q1, \cp, 16
    addi            \t0, a1, 44
    movi            \t1, -32768
    MUL_S8_PUT16    \t1, \cp, \t0
.ifb \ls
    extui           \t1, \mult, 0, 14
.else
    ssl             \ls
    sll             \t1, \mult              // mult << ls, low 32 bits
    extui           \t1, \t1, 0, 14
.endif
    slli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
    srli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
.ifb \ls
    extui           \t1, \mult, 14, 14
.else
    sll             \t1, \mult              // SAR still holds 32 - ls
    extui           \t1, \t1, 14, 14
.endif
    slli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
    srli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
.ifb \ls
    extui           \t1, \mult, 28, 3
.else
    movi.n          \t1, 28
    sub             \t1, \t1, \ls
    ssr             \t1
    srl             \t1, \mult              // (mult << ls) >> 28
.endif
    slli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
    srli            \t1, \t1, 1
    MUL_S8_PUT16    \t1, \cp, \t0
.ifb \ls
    addi            \t1, \rs, -1
    ssl             \t1
    movi.n          \t1, 1
    sll             \t1, \t1
    MUL_S8_PUT16    \t1, \cp, \t0
    movi.n          \t1, 8
    MUL_S8_PUT16    \t1, \cp, \t0
.else
    movi.n          \t1, 0
    MUL_S8_PUT16    \t1, \cp, \t0           // no 2^(rs - 1) rounding term
    MUL_S8_PUT16    \t1, \cp, \t0           // no (x < 0) term
.endif
    ee.vst.128.ip   q3, \cp, 16
    ee.vst.128.ip   q4, \cp, -160
    movi.n          \t1, 1
    s16i            \t1, a1, 44
    ee.vldbc.16     q6, \t0
    addi            \t0, a1, 24
    ee.vldbc.16     q4, \t0
    ssai            1
.endm

// MUL_S8_STEP8 with the 32x32 high multiply done on QACC (8 lanes of 40
// bits), in place of 8 scalar mulsh and 16 q <-> a moves.
// x = a' * b' does not fit s16, so it is split as x = 2 * xh + xb, with
// xh = x >> 1 (vmul at SAR 1) and xb = x & 1 = a' & b' & 1. mult is split
// in 14-bit pieces M0, M1, M2, so 2 * Mj fits s16. Three MAC stages, with a
// floor shift by 14 between them (srcmb keeps the full 40-bit value in
// QACC), give
//   out = floor((x * mult + 2^30 + 2^(30 + rs) - (x < 0) * 2^31) / 2^(31 + rs)),
// which equals RDBP(SRDHM(x, mult), rs) for rs >= 1 (see MUL_S8_STEP8).
// (-2^15)^2 adds the 2^30 nudge in the first stage.
// The last stage adds 8 * (2^(rs - 1) - (x < 0)), and the final srcmb
// shifts by 3 + rs. QACC never overflows: |x * mult| < 2^47 is only reached
// after two shifts by 14.
// \s14 holds 14 and \srs right_shift + 3.
// A q register used right after the load that fills it stalls the
// pipeline. So every constant is loaded at least two ops before its use,
// rotating through q1, q3 and q5. That needs a third free q: q1 holds
// constants, and input1_offset lives in q4, reloaded at the end of each
// step for the next one.
// With \step set (an AR holding 8), \p1 and \p2 may have any alignment.
// Each 8-byte input is then built with ee.src.q from the aligned blocks that
// hold its first and last byte, so no load touches a block outside the
// buffer. \t is scratch. \po must still be 8-byte aligned.
.macro MUL_S8_STEP8_Q p1, p2, po, s14, srs, cp, step=, t=
.ifb \step
    ee.vld.l.64.ip  q2, \p1, 8
    ee.vld.h.64.ip  q2, \p2, 8
    ee.zero.qacc
    ee.vcmp.lt.s8   q3, q2, q0              // sign extend
    ee.vzip.8       q2, q3                  // q2: input1, q3: input2 (s16)
.else
    ee.ld.128.usar.xp q2, \p1, \step       // SAR_BYTE = p1 & 15
    addi            \t, \p1, -1
    ee.vld.128.ip   q3, \t, 0
    ee.zero.qacc
    ee.src.q        q2, q2, q3
    ee.ld.128.usar.xp q3, \p2, \step
    addi            \t, \p2, -1
    ee.vld.128.ip   q5, \t, 0
    ee.vcmp.lt.s8   q1, q2, q0              // sign extend
    ee.src.q        q3, q3, q5
    ee.vzip.8       q2, q1                  // q2: input1 (s16)
    ee.vcmp.lt.s8   q5, q3, q0
    ee.vzip.8       q3, q5                  // q3: input2 (s16)
.endif
    ee.vadds.s16.ld.incp q1, \cp, q4, q2, q4  // a', q1: -2^15
    ee.vadds.s16.ld.incp q5, \cp, q3, q3, q7  // b', q5: 2*M0
    ee.vmul.s16     q2, q4, q3              // xh (SAR 1)
    ee.andq         q4, q4, q3
    ee.vmulas.s16.qacc.ld.ip q3, \cp, 16, q1, q1   // q3: M0
    ee.andq         q4, q4, q6              // xb
    ee.vmulas.s16.qacc.ld.ip q1, \cp, 16, q2, q5   // q1: 2*M1
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q4, q3   // q5: M1
    ee.srcmb.s16.qacc q3, \s14, 0
    ee.vmulas.s16.qacc.ld.ip q3, \cp, 16, q2, q1   // q3: 2*M2
    ee.vmulas.s16.qacc.ld.ip q1, \cp, 16, q4, q5   // q1: M2
    ee.srcmb.s16.qacc q5, \s14, 0
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q2, q3   // q5: 2^(rs - 1)
    ee.vcmp.lt.s16  q2, q2, q0              // (x < 0) as 0 / -1
    ee.vadds.s16.ld.incp q3, \cp, q2, q2, q5  // q3: 8
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q4, q1   // q5: out_offset
    ee.vmulas.s16.qacc.ld.ip q1, \cp, -192, q2, q3 // q1: activation_max
    ee.srcmb.s16.qacc q2, \srs, 0
    ee.vadds.s16.ld.incp q3, \cp, q2, q2, q5  // q3: activation_min
    ee.vmin.s16.ld.incp q4, \cp, q2, q2, q1   // q4: input1_offset
    ee.vmax.s16     q2, q2, q3
    ee.vunzip.8     q2, q5
    ee.vst.l.64.ip  q2, \po, 8
.endm
