// Copyright 2021-2026 Espressif Systems (Shanghai) PTE LTD
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

#include "s3_mul/esp_nn_mul_s8_esp32s3_macros.inc"

    // Frame size, and the stack arguments above it
    .set MUL_S8_FRAME, 336
    .set ARG_OUT_MULT, MUL_S8_FRAME + 0
    .set ARG_OUT_SHIFT, MUL_S8_FRAME + 4
    .set ARG_ACT_MIN, MUL_S8_FRAME + 8
    .set ARG_ACT_MAX, MUL_S8_FRAME + 12
    .set ARG_SIZE, MUL_S8_FRAME + 16

    .text
    .align  4
    .literal_position
    .literal    .LC0_26_123, 1073741824 // `1 << 30`

    .type   esp_nn_mul_elementwise_s8_esp32s3, @function
    .align   4
    .global esp_nn_mul_elementwise_s8_esp32s3

// void esp_nn_mul_elementwise_s8_esp32s3(
//     const int8_t *input1_data,      // a2
//     const int8_t *input2_data,      // a3
//     const int32_t input1_offset,    // a4
//     const int32_t input2_offset,    // a5
//     int8_t *output,                 // a6
//     const int32_t out_offset,       // a7
//     const int32_t out_mult,         // ARG_OUT_MULT
//     const int32_t out_shift,        // ARG_OUT_SHIFT
//     const int32_t activation_min,   // ARG_ACT_MIN
//     const int32_t activation_max,   // ARG_ACT_MAX
//     const int32_t size);            // ARG_SIZE

// Stack frame layout (entry a1, MUL_S8_FRAME):
//  4: input2 (saved)
//  8: output (saved)
// 16: input1 (saved)
// 20: right_shift
// 24: input1_offset (saved)
// 28: input2_offset (saved)
// 40: out_offset (from a7)
// 44: scalar scratch for the macros
// 48..271: vector constants, see s3_mul/esp_nn_mul_s8_esp32s3_macros.inc
// 272..319: free, for new slots without moving the stack arguments
// 320..335: windowed-ABI base save area

esp_nn_mul_elementwise_s8_esp32s3:

    entry   a1, MUL_S8_FRAME
    s32i.n  a4,a1,24                // input1_offset
    s32i.n  a5,a1,28                // input2_offset

    s32i.n  a3,a1,4                 // input2
    mov.n   a10,a3                  // input2 pointer
    l32i    a3,a1,ARG_SIZE
    mov.n   a9,a6                   // out_addr
    blti    a3,1,.exit              // exit

    s32i.n  a2,a1,16                // input1
    s32i    a7,a1,40                // out_offset
    movi.n  a11,0
    mov.n   a12,a2
    movi.n  a5,0                        // 1 while the leftover loop runs the unaligned head

    l32i    a15,a1,ARG_OUT_SHIFT
    l32i    a13,a1,ARG_OUT_MULT
    s32i.n  a6,a1,8                 // out_addr
    max     a14,a15,a11             // left_shift
    sub     a4,a14,a15              // right_shift
    s32i.n  a4,a1,20
    // a4 is the SRDHM nudge (1 << 30) from here on. It must be loaded before
    // the two early branches below: .process_leftover reads it when entered
    // directly (size < 8, or an input not 16-byte aligned).
    l32r    a4,.LC0_26_123

    blti    a3,8,.process_leftover
    bgei    a15,1,.Llshift
    // A right_shift above 16 always gives 0. Run it as mult = 0 and
    // right_shift = 15 (see s3_mul/esp_nn_mul_s8_esp32s3_macros.inc).
    // The scalar leftover keeps the real right_shift from a1+20. With
    // mult = 0 it also gives 0.
    l32i.n  a8,a1,20
    movi.n  a6,16
    bge     a6,a8,1f
    movi.n  a8,15
    movi.n  a13,0
1:
    MUL_S8_SETUP a8, a6, a2
    addi    a15,a8,14                   // right_shift + 14
    srli    a2,a3,3

    // The 64-bit vector loads and stores need 8-byte aligned buffers
    or          a6,a12,a10
    or          a6,a6,a9
    extui       a6,a6,0,3
    beqz        a8,.Lrs0
    bnez        a6,.Lunaligned

    MUL_S8_CONST_PTR a14
    // Right shifts 1..15 take the QACC step, which is about 20 instructions
    // shorter. Its setup pays off from 64 elements.
    bgeui   a3,64,.Lmul8_try_q
.Lmul8_k15:
    loopgtz a2,.Lmul8_end
    MUL_S8_STEP8 a12, a10, a9, a15, a13, a2, a3, a4, a5, a6, a7, a8, a11, cp=a14
.Lmul8_end:
    j       .Lmul8_done

.Lmul8_try_q:
    addi    a7,a8,-1
    movi.n  a6,15
    bgeu    a7,a6,.Lmul8_k15
    MUL_S8_SETUP_Q a8, a13, a14, a6, a7
    movi.n  a5,14
    addi    a6,a8,3
    loopgtz a2,.Lmul8_q_end
    MUL_S8_STEP8_Q a12, a10, a9, a5, a6, a14
.Lmul8_q_end:
    j       .Lmul8_done

    // right_shift = 0 needs its own loop, see MUL_S8_STEP8. Unaligned
    // buffers take the scalar path.
.Lrs0:
    bnez    a6,.Lscalar
    movi.n  a15,16
    MUL_S8_CONST_PTR a14
    loopgtz a2,.Lmul8_rs0_end
    MUL_S8_STEP8 a12, a10, a9, a15, a13, a2, a3, a4, a5, a6, a7, a8, a11, 1, a14
.Lmul8_rs0_end:
    j       .Lmul8_done
.Lscalar:                               // a4 still holds the SRDHM nudge
    j       .process_leftover

    // Unaligned buffers, right shift 1..15 and size >= 64: a scalar head
    // makes the output 8-byte aligned, then the QACC step reads the inputs
    // with ee.src.q. Other unaligned cases take the scalar path.
    // The 64 matches the aligned path. Below it, the head and the QACC setup
    // cost more than the steps save.
.Lunaligned:
    addi    a7,a8,-1
    movi.n  a6,15
    bgeu    a7,a6,.Lscalar
    bltui   a3,64,.Lscalar
    neg     a3,a9
    extui   a3,a3,0,3                   // head size
    movi.n  a5,1                        // come back to .Lunaligned_q
    j       .process_leftover
.Lunaligned_q:
    l32i.n  a12,a1,16
    l32i.n  a10,a1,4
    l32i.n  a9,a1,8
    add.n   a12,a12,a11
    add.n   a10,a10,a11
    add.n   a9,a9,a11
    l32i    a2,a1,ARG_SIZE
    sub     a2,a2,a11
    srli    a2,a2,3
    addi    a8,a15,-14                  // right_shift, 15 if it was above 16
    MUL_S8_SETUP_Q a8, a13, a14, a6, a7
    movi.n  a5,14
    addi    a6,a8,3
    movi.n  a7,8
    loopgtz a2,.Lmul8_qu_end
    MUL_S8_STEP8_Q a12, a10, a9, a5, a6, a14, a7, a8
.Lmul8_qu_end:
    slli    a2,a2,3
    add.n   a11,a11,a2                  // elements done
    movi.n  a5,0
    movi.n  a14,0
    l32r    a4,.LC0_26_123
    l32i    a3,a1,ARG_SIZE
    j       .process_leftover

    // Left shifts 1..11 take the QACC step with right_shift 0 (see
    // MUL_S8_SETUP_Q), for 8-byte aligned buffers and 64 elements or more.
    // Other left shifts take the scalar path.
.Llshift:
    bgei    a15,12,.process_leftover
    bltui   a3,64,.process_leftover
    or      a6,a12,a10
    or      a6,a6,a9
    extui   a6,a6,0,3
    bnez    a6,.process_leftover
    movi.n  a8,0                        // right_shift
    MUL_S8_SETUP a8, a6, a2
    srli    a2,a3,3
    MUL_S8_SETUP_Q a8, a13, a14, a6, a7, a15
    movi.n  a5,14
    movi.n  a6,3                        // final shift 3 + right_shift
    loopgtz a2,.Lmul8_ls_end
    MUL_S8_STEP8_Q a12, a10, a9, a5, a6, a14
.Lmul8_ls_end:
    mov.n   a14,a15                     // the leftover keeps the left shift
    j       .Lmul8_tail

.Lmul8_done:
    movi.n  a14,0                       // left_shift, 0 on the vector path
.Lmul8_tail:
    movi.n  a5,0
    l32r    a4,.LC0_26_123              // SRDHM nudge for the leftover

    l32i    a3,a1,ARG_SIZE                   // size
    movi.n  a11,-8
    and     a11,a3,a11                  // elements done
    bge     a11,a3,.exit

.process_leftover:
    sub     a8,a3,a11
    loopgtz a8,.LBB33_esp_nn_mul_elementwise_s8_esp32s3

    ssl     a14                     // left_shift
    l32i.n  a8,a1,24                // input1_offset
    l32i.n  a10,a1,4                // input2
    l32i.n  a12,a1,16               // input1
    add.n   a10,a11,a10             // input2
    add.n   a12,a11,a12             // input1
    l8ui    a12,a12,0
    l8ui    a10,a10,0
    sext    a12,a12,7
    add.n   a12,a12,a8
    l32i.n  a8,a1,28                // input2_offset
    sext    a10,a10,7
    add.n   a10,a10,a8
    mull    a10,a12,a10             // multiplication result

// multiply by quantised mult
    l32i.n  a9,a1,20                // load right_shift

    sll     a10,a10                 // left shift

    mulsh   a3,a10,a13
    mull    a8,a10,a13
    ssai    31
    add.n   a6,a8,a4
    saltu   a8,a6,a8
    add.n   a8,a8,a3
    src     a3,a8,a6                // result

    blti    a9, 1, .skip_div_by_pow_of_2_remains
// divide by power of 2
    // calculate to_add = `1 << (exponent - 1)`
    addi    a6,a9,-1
    ssl     a6
    movi    a7,1
    sll     a7,a7                       // to_add

    extui   a8,a3,31,1              // sign
    // Round half away from zero, as esp_nn_div_by_power_of_two does:
    // (x + half) >> e for x >= 0, (x + half - 1) >> e for x < 0. The
    // vector path adds the (0 / -1) compare mask; here the sign bit must
    // therefore be subtracted, not added.
    sub     a3,a3,a8            // subtract sign
    add     a3,a3,a7            // add to_add

    ssr     a9                      // load right_shift
    sra     a3,a3               // right shift
.skip_div_by_pow_of_2_remains:

    l32i.n  a6,a1,40                // out_offset
    l32i  a8,a1,ARG_ACT_MAX         // act_max
    l32i  a7,a1,ARG_ACT_MIN         // act_min

// add offset and apply activation
    add.n   a3,a3,a6                // offset added
    min     a8,a8,a3
    l32i.n  a3,a1,8                 // load base out_addr
    max     a8,a8,a7

// store
    add.n   a3,a11,a3               // add index from `a11`
    s8i     a8,a3,0                 // store
    addi.n  a11,a11,1               // inc index

.LBB33_esp_nn_mul_elementwise_s8_esp32s3:// 0x2ed
    bnez    a5,.Lunaligned_q
.exit:
    retw.n

    .size   esp_nn_mul_elementwise_s8_esp32s3, . - esp_nn_mul_elementwise_s8_esp32s3
