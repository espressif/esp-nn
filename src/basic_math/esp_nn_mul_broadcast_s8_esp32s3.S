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

// Optimized broadcast MUL for SE-block pattern: [H,W,C] * [1,1,C]
// Processes 8 channels at a time using S3 SIMD.

#include "s3_mul/esp_nn_mul_s8_esp32s3_macros.inc"

    // Frame size, and the stack arguments above it
    .set MUL_S8_FRAME, 336
    .set ARG_OUT_MULT, MUL_S8_FRAME + 0
    .set ARG_OUT_SHIFT, MUL_S8_FRAME + 4
    .set ARG_ACT_MIN, MUL_S8_FRAME + 8
    .set ARG_ACT_MAX, MUL_S8_FRAME + 12
    .set ARG_SPATIAL, MUL_S8_FRAME + 16
    .set ARG_CHANNELS, MUL_S8_FRAME + 20

    .text
    .align  4
    .literal_position
    .literal    .LC_nudge, 1073741824   // 1 << 30

    .type   esp_nn_mul_broadcast_channel_s8_esp32s3, @function
    .align  4
    .global esp_nn_mul_broadcast_channel_s8_esp32s3

// void esp_nn_mul_broadcast_channel_s8_esp32s3(
//     const int8_t *input1,           // a2
//     const int8_t *input2_per_ch,    // a3
//     const int32_t input1_offset,    // a4
//     const int32_t input2_offset,    // a5
//     int8_t *output,                 // a6
//     const int32_t output_offset,    // a7
//     const int32_t output_mult,      // ARG_OUT_MULT
//     const int32_t output_shift,     // ARG_OUT_SHIFT
//     const int32_t activation_min,   // ARG_ACT_MIN
//     const int32_t activation_max,   // ARG_ACT_MAX
//     const int32_t total_spatial,    // ARG_SPATIAL
//     const int32_t channels);        // ARG_CHANNELS

// Stack frame layout (entry a1, MUL_S8_FRAME):
//  0: right_shift + 3 on the per-row QACC path
//  4: input2_per_ch (saved)
//  8: output base (saved)
// 12: channels
// 16: input1 base (saved)
// 20: right_shift
// 24: input1_offset (saved)
// 28: input2_offset (saved)
// 32: spatial counter
// 36: vector path: 0 off, -1 right_shift = 0, 1 per-row QACC step,
//     else right_shift + 14
// 40: out_offset (from a7)
// 44: scalar scratch for the macros
// 48..271: vector constants, see s3_mul/esp_nn_mul_s8_esp32s3_macros.inc
// 272..319: free, for new slots without moving the stack arguments
// 320..335: windowed-ABI base save area

esp_nn_mul_broadcast_channel_s8_esp32s3:
    entry   a1, MUL_S8_FRAME

    // Save args
    s32i.n  a3, a1, 4               // input2_per_ch base
    s32i.n  a6, a1, 8               // output base
    s32i.n  a2, a1, 16              // input1 base
    s32i.n  a4, a1, 24              // input1_offset
    s32i.n  a5, a1, 28              // input2_offset
    s32i    a7, a1, 40              // out_offset
    movi.n  a5, 0                   // 1 while the leftover loop runs a row head

    l32i    a8, a1, ARG_SPATIAL             // total_spatial
    l32i    a9, a1, ARG_CHANNELS             // channels
    s32i.n  a9, a1, 12              // save channels

    blti    a8, 1, .Lexit           // no spatial positions
    blti    a9, 1, .Lexit           // no channels

    // Prepare shift values
    l32i    a15, a1, ARG_OUT_SHIFT            // output_shift
    movi.n  a11, 0
    max     a14, a15, a11           // left_shift = max(shift, 0)
    sub     a12, a14, a15           // right_shift = left_shift - shift
    s32i.n  a12, a1, 20             // save right_shift

    l32i    a13, a1, ARG_OUT_MULT            // output_mult

    // The vector path needs at least 8 channels. Left shifts take .Llshift.
    // A right_shift above 16 always gives 0. Run it as mult = 0 and
    // right_shift = 15 (see s3_mul/esp_nn_mul_s8_esp32s3_macros.inc).
    // The scalar leftover keeps the real right_shift from a1+20. With
    // mult = 0 it also gives 0.
    s32i    a11, a1, 36
    blti    a9, 8, .Lsetup_done
    bgei    a15, 1, .Llshift
    movi.n  a8, 16
    bge     a8, a12, 1f
    movi.n  a12, 15
    movi.n  a13, 0
1:
    MUL_S8_SETUP a12, a8, a10
    addi    a8, a12, 14
    movi.n  a10, -1
    moveqz  a8, a10, a12
    s32i    a8, a1, 36

.Lsetup_done:
    l32r    a4, .LC_nudge           // nudge = 1 << 30

    // Init spatial counter
    movi.n  a10, 0
    s32i    a10, a1, 32             // spatial counter = 0

    // With channels a multiple of 8, 8-byte aligned buffers and
    // right_shift 1..15, every row is aligned and has no leftover. Run the
    // rows back to back then: only input2 is reset between rows.
    l32i    a12, a1, 36
    blti    a12, 1, .Lspatial_loop
    or      a10, a2, a3
    or      a10, a10, a6
    or      a10, a10, a9
    extui   a10, a10, 0, 3
    // Right shifts 1..15 take the QACC step (see MUL_S8_STEP8_Q). a12 is
    // right_shift + 14.
    addi    a11, a12, -15
    movi.n  a8, 15
    bgeu    a11, a8, .Lspatial_loop
    addi    a12, a12, -14           // right_shift
    MUL_S8_SETUP_Q a12, a13, a14, a8, a11
    addi    a7, a12, 3
    bnez    a10, .Lrows_q
    // The step uses no scalar temps here, so the row count (a10), the
    // steps per row (a12) and the input2 base (a11) stay in registers.
    // Rows can be as short as 2 steps, so stack reloads per row add up.
.Lrows_back_to_back:
    l32i    a10, a1, ARG_SPATIAL            // total_spatial
    srli    a12, a9, 3
    mov.n   a11, a3
    movi.n  a5, 14
.Lchannel_q:
    mov.n   a3, a11                 // input2_per_ch base
    loopgtz a12, .Lchannel_q_end
    MUL_S8_STEP8_Q a2, a3, a6, a5, a7, a14
.Lchannel_q_end:
    addi.n  a10, a10, -1
    bnez    a10, .Lchannel_q
    j       .Lexit

    // Left shifts 1..11 run the back-to-back rows with right_shift 0 (see
    // MUL_S8_SETUP_Q), when channels are a multiple of 8 and the buffers
    // 8-byte aligned. Other left shifts take the scalar path.
.Llshift:
    bgei    a15, 12, .Lsetup_done
    or      a10, a2, a3
    or      a10, a10, a6
    or      a10, a10, a9
    extui   a10, a10, 0, 3
    bnez    a10, .Lsetup_done
    movi.n  a12, 0                  // right_shift
    MUL_S8_SETUP a12, a8, a10
    MUL_S8_SETUP_Q a12, a13, a14, a8, a11, a15
    movi.n  a7, 3                   // final shift 3 + right_shift
    j       .Lrows_back_to_back

    // Rows that are not all 8-byte aligned run the QACC step row by row,
    // see .Lrow_q. a1+36 = 1 selects that path, a1+0 holds right_shift + 3.
.Lrows_q:
    s32i.n  a7, a1, 0
    movi.n  a12, 1
    s32i    a12, a1, 36
    movi.n  a14, 0                  // left_shift for the leftover loop

    // Pointers: a2 = input1 (current), a3 = input2_per_ch (reloaded each row),
    //           a6 = output (current)

.Lspatial_loop:
    l32i    a8, a1, ARG_SPATIAL             // total_spatial
    l32i    a10, a1, 32             // spatial counter
    bge     a10, a8, .Lexit

    // Reset input2 pointer for each spatial position
    l32i.n  a3, a1, 4               // input2_per_ch base

    // Channel counter
    l32i.n  a9, a1, 12              // channels
    movi.n  a11, 0                  // channel index
    mov.n   a15, a9                 // leftover loop bound

    l32i    a12, a1, 36
    beqz    a12, .Lchannel_leftover
    srli    a8, a9, 3

    // The 64-bit vector loads and stores need 8-byte aligned rows
    or      a10, a2, a3
    or      a10, a10, a6
    extui   a10, a10, 0, 3
    bltz    a12, .Lchannel_rs0
    beqi    a12, 1, .Lrow_q
    bnez    a10, .Lchannel_leftover   // right_shift 16, unaligned row

.Lchannel_simd:
    MUL_S8_CONST_PTR a14
    loopgtz a8, .Lchannel_simd_end
    MUL_S8_STEP8 a2, a3, a6, a12, a13, a4, a5, a7, a8, a9, a10, a11, a15, cp=a14
.Lchannel_simd_end:
    j       .Lchannel_vec_done

    // right_shift = 0 needs its own loop, see MUL_S8_STEP8. Unaligned rows
    // take the scalar path.
.Lchannel_rs0:
    bnez    a10, .Lchannel_leftover
    movi.n  a12, 16
    MUL_S8_CONST_PTR a14
    loopgtz a8, .Lchannel_rs0_end
    MUL_S8_STEP8 a2, a3, a6, a12, a13, a4, a5, a7, a8, a9, a10, a11, a15, 1, a14
.Lchannel_rs0_end:
    j       .Lchannel_vec_done

    // QACC step on one row. A scalar head (the leftover loop with bound
    // a15 and a5 = 1) makes the output 8-byte aligned. The step reads the
    // inputs with ee.src.q, so they may have any alignment.
.Lrow_q:
    neg     a15, a6
    extui   a15, a15, 0, 3          // head size
    movi.n  a5, 1
    j       .Lchannel_leftover
.Lrow_q_body:
    l32i.n  a15, a1, 12             // channels
    sub     a8, a15, a11
    srli    a8, a8, 3
    MUL_S8_CONST_PTR a14
    addi    a14, a14, 32            // where MUL_S8_SETUP_Q left it
    movi.n  a5, 14
    l32i.n  a7, a1, 0               // right_shift + 3
    movi.n  a12, 8
    ssai    1                       // the scalar head changed SAR
    loopgtz a8, .Lrow_q_end
    MUL_S8_STEP8_Q a2, a3, a6, a5, a7, a14, a12, a10
.Lrow_q_end:
    slli    a8, a8, 3
    add.n   a11, a11, a8            // channels done
    movi.n  a5, 0
    movi.n  a14, 0
    j       .Lchannel_leftover

.Lchannel_vec_done:
    movi.n  a5, 0
    movi.n  a14, 0                  // left_shift, 0 on the vector path
    l32r    a4, .LC_nudge
    l32i.n  a15, a1, 12             // channels
    movi.n  a11, -8
    and     a11, a15, a11           // channels done

.Lchannel_leftover:
    // Process channels one by one, up to a15
    bge     a11, a15, .Lleftover_done

    ssl     a14                     // left_shift
    l32i.n  a8, a1, 24              // input1_offset
    l8ui    a10, a2, 0              // *input1
    sext    a10, a10, 7
    add.n   a10, a10, a8            // + input1_offset
    l32i.n  a8, a1, 28              // input2_offset
    l8ui    a12, a3, 0              // *input2_per_ch
    sext    a12, a12, 7
    add.n   a12, a12, a8            // + input2_offset
    mull    a10, a10, a12           // multiply

    // Requantize
    sll     a10, a10                // left shift

    l32i.n  a9, a1, 20              // right_shift
    mulsh   a8, a10, a13
    mull    a12, a10, a13
    ssai    31
    add.n   a12, a4, a12
    saltu   a10, a12, a4
    add.n   a10, a10, a8
    src     a10, a10, a12           // result

    blti    a9, 1, .Lskip_div_scalar

    addi    a8, a9, -1
    ssl     a8
    movi    a7, 1
    sll     a7, a7                  // to_add
    extui   a8, a10, 31, 1          // sign bit (1 if neg, 0 if pos)
    sub     a10, a10, a8            // val -= sign (fast rounding)
    add     a10, a10, a7
    ssr     a9
    sra     a10, a10

.Lskip_div_scalar:
    l32i    a8, a1, 40              // output_offset
    l32i    a7, a1, ARG_ACT_MIN             // activation_min
    l32i    a12, a1, ARG_ACT_MAX            // activation_max
    add.n   a10, a10, a8
    min     a10, a10, a12
    max     a10, a10, a7
    s8i     a10, a6, 0              // store

    addi    a2, a2, 1               // input1++
    addi    a3, a3, 1               // input2++
    addi    a6, a6, 1               // output++
    addi    a11, a11, 1             // channel index++
    j       .Lchannel_leftover

.Lleftover_done:
    bnez    a5, .Lrow_q_body
.Lspatial_next:
    l32i    a10, a1, 32             // spatial counter
    addi    a10, a10, 1
    s32i    a10, a1, 32
    j       .Lspatial_loop

.Lexit:
    retw.n

    .size   esp_nn_mul_broadcast_channel_s8_esp32s3, . - esp_nn_mul_broadcast_channel_s8_esp32s3
