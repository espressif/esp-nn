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
//     const int32_t output_mult,      // stack+120
//     const int32_t output_shift,     // stack+124
//     const int32_t activation_min,   // stack+128
//     const int32_t activation_max,   // stack+132
//     const int32_t total_spatial,    // stack+136
//     const int32_t channels);        // stack+140

// Stack frame layout (entry a1, 120):
//  0: to_add (for div by power of 2)
//  4: input2_per_ch (saved)
//  8: output base (saved)
// 12: channels
// 16: input1 base (saved)
// 20: right_shift
// 24: input1_offset (saved)
// 28: input2_offset (saved)
// 32: spatial counter
// 36: out_ptr (current)
// 40: out_offset (from a7)
// 44: input1_offset (for vldbc)
// 48: input2_offset (for vldbc)

esp_nn_mul_broadcast_channel_s8_esp32s3:
    entry   a1, 120

    // Save args
    s32i.n  a3, a1, 4               // input2_per_ch base
    s32i.n  a6, a1, 8               // output base
    s32i.n  a2, a1, 16              // input1 base
    s32i.n  a4, a1, 24              // input1_offset
    s32i.n  a5, a1, 28              // input2_offset
    s32i    a7, a1, 40              // out_offset

    l32i    a8, a1, 136             // total_spatial
    l32i    a9, a1, 140             // channels
    s32i.n  a9, a1, 12              // save channels

    blti    a8, 1, .Lexit           // no spatial positions
    blti    a9, 1, .Lexit           // no channels

    // Prepare shift values
    l32i    a15, a1, 124            // output_shift
    movi.n  a11, 0
    max     a14, a15, a11           // left_shift = max(shift, 0)
    sub     a4, a14, a15            // right_shift = left_shift - shift
    s32i.n  a4, a1, 20              // save right_shift

    l32i    a13, a1, 120            // output_mult
    l32r    a4, .LC_nudge           // nudge = 1 << 30

    // Store offsets for vldbc
    l32i    a8, a1, 136             // reload total_spatial
    s32i    a5, a1, 48              // input2_offset for vldbc
    l32i.n  a5, a1, 24              // input1_offset
    s32i    a5, a1, 44              // input1_offset for vldbc

    // Init spatial counter
    movi.n  a10, 0
    s32i    a10, a1, 32             // spatial counter = 0

    // Pointers: a2 = input1 (current), a3 = input2_per_ch (reloaded each row),
    //           a6 = output (current)

.Lspatial_loop:
    l32i    a8, a1, 136             // total_spatial
    l32i    a10, a1, 32             // spatial counter
    bge     a10, a8, .Lexit

    // Reset input2 pointer for each spatial position
    l32i.n  a3, a1, 4               // input2_per_ch base

    // Channel counter
    l32i.n  a9, a1, 12              // channels
    movi.n  a11, 0                  // channel index

    blti    a9, 8, .Lchannel_leftover

    // Check alignment for SIMD path
    or      a8, a2, a3
    or      a8, a8, a6
    extui   a8, a8, 0, 4
    bnez    a8, .Lchannel_leftover

    // Setup SIMD constants
    ee.zero.q   q1                  // zero register
    addi    a8, a1, 44
    ee.vldbc.16 q0, a8              // input1_offset broadcast
    addi    a8, a1, 48
    ee.vldbc.16 q7, a8              // input2_offset broadcast
    st.qr   q0, a1, 64             // save for reload in loop

.Lchannel_simd_loop:
    addi    a8, a9, -7              // channels - 7
    blt     a11, a8, .Lchannel_simd_body
    j       .Lchannel_leftover

.Lchannel_simd_body:
    ld.qr           q4, a1, 64             // input1_offset
    ee.vld.l.64.ip  q2, a2, 8              // load 8 input1 values
    movi.n          a7, 16
    ee.vld.h.64.ip  q2, a3, 8              // load 8 input2 values (per-ch)
    wsr.sar         a7
    ee.vcmp.lt.s8   q5, q2, q1             // sign extend
    ee.vzip.8       q2, q5                 // interleave to 16-bit
    ee.vadds.s16    q5, q5, q7             // add input2_offset
    ee.vadds.s16    q4, q2, q4             // add input1_offset
    ee.vmul.s16     q3, q4, q5             // multiply (high part)
    ssai            0                      // sar = 0
    ee.vmul.s16     q2, q4, q5             // multiply (low part)

    // Requantize 8 results (same pattern as elementwise mul)
    wsr.sar         a14                     // left_shift
    ee.vzip.16      q2, q3
    ee.vsl.32       q6, q2                  // left shift first 4

    ssai            31

    // Element 2 of q6
    ee.movi.32.a    q6, a8, 2
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a5, a8, a4
    add.n           a5, a5, a7
    src             a5, a5, a8
    // Element 3
    ee.movi.32.a    q6, a8, 3
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a12, a8, a4
    add.n           a12, a12, a7
    src             a12, a12, a8
    ee.movi.32.q    q2, a5, 2
    ee.movi.32.q    q2, a12, 3
    // Element 1
    ee.movi.32.a    q6, a8, 1
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a5, a8, a4
    add.n           a5, a5, a7
    src             a5, a5, a8
    // Element 0
    ee.movi.32.a    q6, a8, 0
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a12, a8, a4
    add.n           a12, a12, a7
    src             a12, a12, a8
    ee.movi.32.q    q2, a5, 1
    ee.movi.32.q    q2, a12, 0

    // Second group of 4 (q3)
    wsr.sar         a14                     // left_shift
    ee.vsl.32       q4, q3

    ssai            31

    ee.movi.32.a    q4, a8, 2
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a5, a8, a4
    add.n           a5, a5, a7
    src             a5, a5, a8
    ee.movi.32.a    q4, a8, 3
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a12, a8, a4
    add.n           a12, a12, a7
    src             a12, a12, a8
    ee.movi.32.q    q0, a5, 2
    ee.movi.32.q    q0, a12, 3
    ee.movi.32.a    q4, a8, 1
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a5, a8, a4
    add.n           a5, a5, a7
    src             a5, a5, a8
    ee.movi.32.a    q4, a8, 0
    mulsh           a7, a13, a8
    mull            a8, a13, a8
    add.n           a8, a4, a8
    saltu           a12, a8, a4
    add.n           a12, a12, a7
    src             a12, a12, a8
    ee.movi.32.q    q0, a5, 1
    ee.movi.32.q    q0, a12, 0

    // Divide by power of 2 (right_shift)
    l32i.n          a5, a1, 20              // right_shift
    movi.n          a7, 1

    blti            a5, 1, .Lskip_div

    ee.vcmp.lt.s32  q5, q2, q1
    ee.vcmp.lt.s32  q6, q0, q1
    addi.n          a8, a5, -1
    ssl             a8
    sll             a7, a7                  // to_add = 1 << (right_shift - 1)
    s32i.n          a7, a1, 0
    ee.vldbc.32     q4, a1                  // broadcast to_add
    wsr.sar         a5
    ee.vadds.s32    q5, q4, q5
    ee.vadds.s32    q5, q2, q5
    ee.vsr.32       q2, q5
    wsr.sar         a5
    ee.vadds.s32    q5, q4, q6
    ee.vadds.s32    q5, q0, q5
    ee.vsr.32       q0, q5

.Lskip_div:
    // Add output offset, apply activation
    addi            a8, a1, 132
    ee.vldbc.32     q4, a8                  // activation_max
    addi            a5, a1, 40
    ee.vldbc.32     q6, a5                  // output_offset
    addi            a7, a1, 128
    ee.vadds.s32    q0, q0, q6              // add offset
    ee.vadds.s32    q2, q2, q6
    ee.vldbc.32     q6, a7                  // activation_min
    ee.vmin.s32     q0, q0, q4
    ee.vmin.s32     q2, q2, q4
    ee.vmax.s32     q0, q0, q6
    ee.vmax.s32     q2, q2, q6

    // Pack 32-bit -> 8-bit and store
    ee.vunzip.16    q2, q0
    ee.vunzip.8     q2, q0
    ee.vst.l.64.ip  q2, a6, 8

    addi            a11, a11, 8             // channel index += 8
    j               .Lchannel_simd_loop

.Lchannel_leftover:
    // Process remaining channels one by one
    l32i.n  a9, a1, 12              // channels
    bge     a11, a9, .Lspatial_next

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
    l32i    a7, a1, 128             // activation_min
    l32i    a12, a1, 132            // activation_max
    add.n   a10, a10, a8
    min     a10, a10, a12
    max     a10, a10, a7
    s8i     a10, a6, 0              // store

    addi    a2, a2, 1               // input1++
    addi    a3, a3, 1               // input2++
    addi    a6, a6, 1               // output++
    addi    a11, a11, 1             // channel index++
    j       .Lchannel_leftover

.Lspatial_next:
    l32i    a10, a1, 32             // spatial counter
    addi    a10, a10, 1
    s32i    a10, a1, 32
    j       .Lspatial_loop

.Lexit:
    retw.n

    .size   esp_nn_mul_broadcast_channel_s8_esp32s3, . - esp_nn_mul_broadcast_channel_s8_esp32s3
