// Copyright 2021-2022 Espressif Systems (Shanghai) PTE LTD
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

    .text

 # Requantize the 4 sums of a QACC half, dumped at a1[base..base+19] as 40-bit
 # lanes, into the int32 lanes of \qd. a8 = mult, a3 = left shift. The
 # bias is read from a1[52]. Clobbers a9-a15 and SAR.
 # floor((v * mult + 2^30) / 2^31) is the exact reference result for
 # mult >= 0. With v * mult = hi:lo it is (hi:lo >> 31) + bit 30 of lo.
 # The lanes at base+5/10/15 are read with unaligned l32i, which the S3
 # core supports in hardware.
 .macro REQUANT4 qd, base, lshift, set_sar=1
    l32i    a10,a1,52
    l32i    a12,a1,\base
    addi    a9,a1,\base+5
    l32i    a13,a9,0
    addi    a9,a1,\base+10
    l32i    a14,a9,0
    addi    a9,a1,\base+15
    l32i    a15,a9,0
    add     a12,a12,a10
    add     a13,a13,a10
    add     a14,a14,a10
    add     a15,a15,a10
  .if \lshift
    ssl     a3
    sll     a12,a12
    sll     a13,a13
    sll     a14,a14
    sll     a15,a15
  .endif
  .if \lshift + \set_sar
    ssai    31
  .endif
    mulsh   a9,a8,a12
    mull    a12,a8,a12
    mulsh   a11,a8,a13
    extui   a10,a12,30,1
    src     a12,a9,a12
    mull    a13,a8,a13
    add     a12,a12,a10
    extui   a10,a13,30,1
    src     a13,a11,a13
    mulsh   a9,a8,a14
    add     a13,a13,a10
    mull    a14,a8,a14
    mulsh   a11,a8,a15
    extui   a10,a14,30,1
    src     a14,a9,a14
    mull    a15,a8,a15
    add     a14,a14,a10
    extui   a10,a15,30,1
    src     a15,a11,a15
    add     a15,a15,a10
    ee.movi.32.q    \qd,a12,0
    ee.movi.32.q    \qd,a13,1
    ee.movi.32.q    \qd,a14,2
    ee.movi.32.q    \qd,a15,3
 .endm

 # Fast form of the SRDHM part, used when every channel passes the bound
 # checked at entry. With |v| < 2^29 after the left shift,
 # u = mulsh(v << 2, mult) = floor(v * mult / 2^30) is exact, and the SRDHM
 # result is (u + 1) >> 1. The epilogue folds that halving into the rounding
 # shift. a10 holds 4 * bias, so addx4 gives (acc + bias) << 2. With lshift,
 # SAR holds the left shift. a3, a6 and a11 hold a1 + 1, 2 and 3, so the
 # unaligned lane loads need no address adds.
 .macro FASTREQ4 qd, base, lshift
    l32i    a12,a1,\base
    l32i    a13,a3,\base+4               # a1 + base + 5
    l32i    a14,a6,\base+8               # a1 + base + 10
    l32i    a15,a11,\base+12             # a1 + base + 15
    addx4   a12,a12,a10
    addx4   a13,a13,a10
    addx4   a14,a14,a10
    addx4   a15,a15,a10
  .if \lshift
    sll     a12,a12
    sll     a13,a13
    sll     a14,a14
    sll     a15,a15
  .endif
    mulsh   a12,a8,a12
    mulsh   a13,a8,a13
    mulsh   a14,a8,a14
    mulsh   a15,a8,a15
    ee.movi.32.q    \qd,a12,0
    ee.movi.32.q    \qd,a13,1
    ee.movi.32.q    \qd,a14,2
    ee.movi.32.q    \qd,a15,3
 .endm

    # Program Unit: esp_nn_conv_s8_mult8_1x1_esp32s3
    #
    # Pure .S: this hand-scheduled kernel predates esp-nn's inline-asm
    # kernels and is tuned here in place. Moving it to inline asm in C is
    # a separate change.
    #
    # Requirements:
    #   - size (input_wd * input_ht) must be a multiple of 8. The C driver
    #     runs a lone last position with esp_nn_conv_s8_1x1_1pos_esp32s3.
    #   - in_channels must be a multiple of 8
    #   - input_data must be 8-byte aligned (ee.vld.l/h.64.xp ignore the low 3 address bits)
    #   - buffer (scratch) must be 16-byte aligned, with 16 * in_channels
    #     bytes for the transpose and 16 * out_channels for the per-OC table
    #   - table_ready = 1 only when the last call used the same buffer,
    #     in_channels, out_channels, bias, out_shift, out_mult and out_offset
    #   - filter_data may have any alignment, but it must be readable for
    #     16 bytes past its end. The 8-channel loop (in_channels % 16 == 8)
    #     loads filter blocks in funnel-shift pairs, and the second load of a
    #     pair can read the 16-byte block just past the filter's end. The
    #     16-channel loop reads a 16-byte aligned filter with plain aligned
    #     loads.
    #
    .type   esp_nn_conv_s8_mult8_1x1_esp32s3, @function
    .align   4
    .global esp_nn_conv_s8_mult8_1x1_esp32s3

esp_nn_conv_s8_mult8_1x1_esp32s3:  # 0xdbc
    # Stack frame (240 bytes). Every slot stays below 192: the top 48 bytes
    # are the register-window save area.
    #   0..51    QACC_L and QACC_H dumps (0..19, 32..51)
    #   20       size - 7
    #   24       a3 across the output tile flush
    #   28       end of the per-OC table, ends the OC loop
    #   52       bias of the current output channel (exact path)
    #   56       input
    #   60       filter_data
    #   64       input_offset
    #   68       input_ptr
    #   72       i_out
    #   76       in_channels
    #   80       size
    #   84       16-channel blocks after the first, -1 for in_channels % 16
    #            == 8, bit 30 set for a filter that is not 16-byte aligned
    #   88..100  1, out_offset, activation_min, activation_max
    #   104      bias step (0 for a NULL bias)
    #   108      zero word, the bias of a NULL bias pointer
    #   112..175 8x8 output tile
    #   176      output tile flag
    #   180      free
    #   184      per-OC table, at buffer + 16 * in_channels
    #   188..191 free

 // registers:
 // a2: int8_t *input_data
 // a3: uint16_t input_wd
 // a4: uint16_t input_ht
 // a5: uint16_t in_channels
 // a6: int32_t input_offset
 // a7: int16_t *filter_data

 // on stack:
 // int32_t *bias           // 240
 // int8_t *out_data        // 244
 // uint16_t out_wd         // 248
 // uint16_t out_ht         // 252
 // uint16_t out_channels   // 256
 // int32_t out_offset      // 260
 // int32_t *out_shift      // 264
 // int32_t *out_mult       // 268
 // int32_t activation_min  // 272
 // int32_t activation_max  // 276
 // void *buffer // tmp buf // 280
 // int table_ready         // 284, 1: the per-OC table is still valid

    entry   a1,240                      #
    # The top 48 bytes of the frame (192 and above) are a register-window
    # save area. A window spill on an interrupt may overwrite them, so all
    # slots stay below 192. a1[112..175] is the 8x8 output tile.
    s32i    a5,a1,76                    # [0]  gra_spill_temp_166, in_channels
    s32i    a6,a1,64                    # [2]  id:619 input_offset+0x0
    s32i    a7,a1,60                    # [1]  gra_spill_temp_170, filter_data
    mul16u  a8,a3,a4                    # [3]  size = input_wd * input_ht;
    s32i    a2,a1,56                    # [0]  gra_spill_temp_169, input
    l32i    a4,a1,244                   # [1]  id:624 out_data+0x0
    mov.n   a3,a1                       # scratch_buf
    # 1, out_offset, activation_min and activation_max, next to each other.
    # The epilogue walks them with one pointer.
    l32i    a9,a1,260
    l32i    a10,a1,272
    l32i    a11,a1,276
    s32i    a9,a1,92
    s32i    a10,a1,96
    s32i    a11,a1,100
    movi.n  a9,1
    s32i    a9,a1,88

    s32i    a8,a1,80                    # [4]  gra_spill_temp_176, size
    # Output tile flag: out_data 8-byte aligned and out_channels % 8 == 0
    l16ui   a9,a1,256                   # out_channels
    or      a9,a9,a4
    extui   a9,a9,0,3
    movi.n  a10,1
    movi.n  a11,0
    movnez  a10,a11,a9
    s32i    a10,a1,176
    blti    a8,8,.return_function
    addi    a9,a8,-7                    # [32]
    s32i    a9,a1,20                    # [33]  gra_spill_temp_175, size-7

    s32i    a2,a1,68                    # [2]  gra_spill_temp_172 , input_ptr
    srai    a15,a5,3                    # [7] `in_ch/8` loop_cnt
    movi.n  a11,0                       # [10]
    s32i    a11,a1,72                   # [11]  gra_spill_temp_165
    # 16-channel blocks after the first, or -1 when in_channels % 16 == 8.
    # Bit 30 set: the filter is not 16-byte aligned, so the 16-channel loop
    # takes the funnel-shift loads (.mac16u). An aligned filter takes plain
    # aligned loads, which never touch a block past the filter's end.
    srli    a9,a5,4
    addi.n  a9,a9,-1
    movi.n  a10,-1
    extui   a11,a5,3,1
    movnez  a9,a10,a11
    extui   a12,a7,0,4
    movi.n  a13,1
    slli    a13,a13,30
    or      a13,a9,a13
    movnez  a9,a13,a12
    s32i    a9,a1,84
    # Bias pointer and step. A NULL bias reads a zero word with step 0.
    l32i    a9,a1,240
    movi.n  a10,0
    s32i    a10,a1,108                   # zero word
    movi.n  a11,4
    moveqz  a11,a10,a9
    s32i    a11,a1,104                  # bias step
    # Per-OC table at buffer + 16 * in_channels, so the OC loop only loads
    # its constants. Each OC takes 16 bytes:
    # - mult.
    # - The bias term.
    # - The rounding term plus out_offset << (rs + 1). The rounding term is
    #   1 + 2^rs, or 1 when rs is 0.
    # - The shifts word. Bits 0-4 hold ls. Bit 8 is set unless the channel
    #   takes plain FASTREQ4. Bit 9 is set for the exact path. Bits 16-21
    #   hold rs + 1.
    # FASTREQ4 needs |acc << left_shift| < 2^29 and a right shift <= 21.
    # Then |u| < 2^30 and the rounding term is below 2^30 in magnitude, so
    # the vector adds of the fast path do not saturate.
    # |acc| <= in_channels * 255 * 128 + |bias| < (in_channels << 15) + |bias|.
    # A channel that passes stores 4 * bias (it fits, |bias| < 2^29). The
    # others store the bias and take the exact path. a1[184] points at it.
    addi    a10,a1,108
    moveqz  a9,a10,a9                   # NULL bias reads the zero word
    l32i    a10,a1,264                  # out_shift
    l32i    a7,a1,268                   # out_mult
    l32i    a6,a1,280                   # buffer
    slli    a13,a5,4
    add     a6,a6,a13                   # table
    s32i    a6,a1,184
    l32i    a12,a1,284                  # table_ready
    bnez    a12,.table_end              # same channels as the last call
    l16ui   a12,a1,256                  # out_channels
    slli    a13,a5,15
    loopgtz a12,.table_end
    l32i    a15,a9,0                    # bias
    l32i    a2,a10,0                    # shift
    l32i    a8,a7,0                     # mult
    add     a9,a9,a11
    addi.n  a10,a10,4
    addi.n  a7,a7,4
    s32i    a8,a6,0
    movi.n  a3,0
    max     a3,a2,a3                    # left shift
    sub     a8,a3,a2                    # right shift
    movi    a5,29
    min     a2,a3,a5                    # left shift > 29 shifts by 0: exact path
    sub     a2,a5,a2
    ssr     a2
    abs     a2,a15
    add     a2,a2,a13                   # bound on |acc|, unsigned, never 0
    srl     a2,a2                       # nonzero: bound >= 2^(29 - left shift)
    movi    a5,21
    sub     a5,a5,a8
    srli    a5,a5,31                    # right shift > 21
    or      a2,a2,a5                    # nonzero: exact path
    slli    a5,a15,2
    moveqz  a15,a5,a2
    s32i    a15,a6,4                    # 4 * bias, or the bias
    ssl     a8
    movi.n  a5,1
    sll     a15,a5
    add     a15,a15,a5                  # 1 + 2^rs
    moveqz  a15,a5,a8                   # 1 for rs 0
    l32i    a14,a1,92                   # out_offset
    addi    a12,a8,1
    ssl     a12
    sll     a14,a14
    add     a15,a15,a14                 # + out_offset << (rs + 1)
    s32i    a15,a6,8
    addi    a15,a8,1
    slli    a15,a15,16
    or      a15,a15,a3
    movi    a5,0x100
    or      a5,a15,a5
    movnez  a15,a5,a3                   # bit 8: ls != 0
    movi    a5,0x300
    or      a5,a15,a5
    movnez  a15,a5,a2                   # bits 8, 9: exact path
    s32i    a15,a6,12
    addi    a6,a6,16
.table_end:

.outer_loop: // for (; i_out < size - 7; i_out += 8) {

    l32i    a10,a1,280                  # [1]  gra_spill_temp_165, buffer
    l32i    a11,a1,76                   # [1]  gra_spill_temp_166, input_channels
    l32i    a8,a1,68                    # [2]  gra_spill_temp_172, input_ptr
    srai    a9,a11,3                    # [7] `in_ch/8` loop_cnt for transpose loop

    ee.zero.q   q7                      # [0]
    addi        a12,a1,64               # [6]
    ee.vldbc.16 q5,a12                  # [0*II+16]  id:638 input_offset

    // load and transose 8 lines of input 8xchannels,
    // add input offset and store 16 bit data to tmp buffer
    loopgtz a9,.transpose_loop_end  # [10]
    mov.n                   a9,a8
    ee.vld.l.64.xp          q0,a9,a11
    ee.vld.l.64.xp          q1,a9,a11
    ee.vld.h.64.xp          q0,a9,a11
    ee.vld.h.64.xp          q1,a9,a11
    ee.vld.l.64.xp          q2,a9,a11
    ee.vzip.8               q0,q1
    ee.vld.l.64.xp          q3,a9,a11
    ee.vld.h.64.xp          q2,a9,a11
    ee.vld.h.64.ip          q3,a9,0
    ee.vzip.16              q0,q1
    ee.vzip.8               q2,q3
    ee.vzip.16              q2,q3
    ee.vzip.32              q0,q2
    ee.vcmp.lt.s8           q4,q2,q7
    ee.vzip.8               q2,q4
    ee.vcmp.lt.s8           q6,q0,q7
    ee.vzip.8               q0,q6
    ee.vadds.s16            q0,q0,q5
    ee.vadds.s16.st.incp    q0,a10,q6,q6,q5
    ee.vadds.s16.st.incp    q6,a10,q2,q2,q5
    ee.vadds.s16.st.incp    q2,a10,q4,q4,q5
    ee.vst.128.ip           q4,a10,16
    ee.vzip.32              q1,q3
    ee.vcmp.lt.s8           q4,q3,q7
    ee.vzip.8               q3,q4
    ee.vcmp.lt.s8           q6,q1,q7
    ee.vzip.8               q1,q6
    ee.vadds.s16            q1,q1,q5
    ee.vadds.s16.st.incp    q1,a10,q6,q6,q5
    ee.vadds.s16.st.incp    q6,a10,q3,q3,q5
    ee.vadds.s16.st.incp    q3,a10,q4,q4,q5
    ee.vst.128.ip           q4,a10,16
    addi.n                  a8,a8,8
.transpose_loop_end:    # 0xeeb

    l32i    a5,a1,60                    # filter
    l16ui   a8,a1,256                   # out_channels
    l32i    a2,a1,184                   # per-OC table, walked in a2
    blti    a8,1,.outer_ch_loop_end
    slli    a10,a8,4
    add     a10,a10,a2
    s32i    a10,a1,28                   # table end, ends the OC loop
    addi    a3,a1,1                     # lane pointers of FASTREQ4
    addi    a6,a1,2
    addi    a11,a1,3
    ee.zero.q   q5                      # sign compare zero, kept across channels

.out_ch_loop:   # 0xf3e
    # Loads sit a few instructions ahead of the branches that use them:
    # a load followed by a branch on it stalls.
    l32i    a9,a1,84                    # 16-channel blocks after the first
    l32i    a8,a1,280                   # [4]  gra_spill_temp_165, buffer_ptr
    ee.zero.qacc                        # [3]
    bltz    a9,.mac8                    # in_channels % 16 == 8
    bbsi    a9,30,.mac16u               # filter not 16-byte aligned
    # 16 channels per step: one filter load gives two s16 halves
    ee.vld.128.ip                   q7,a5,16
    ee.vld.128.ip                   q0,a8,16
    ee.vld.128.ip                   q1,a8,16
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6       # q7: channels 0-7, q6: 8-15
    loopgtz a9,.mac16_loop_end
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7
    # next filter block; the last block is peeled so no load runs past it
    ee.vld.128.ip                   q7,a5,16
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,7
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6
.mac16_loop_end:
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,5
    ee.vsmulas.s16.qacc q2,q6,6
    ee.vsmulas.s16.qacc q3,q6,7

.mac_done:

 # Per output channel requant constants, from the table built at entry.
 # a2 walks the table. a7 holds the packed shifts word.
    mov.n   a9,a1
    ee.st.qacc_l.l.128.ip   a9,16       # QACC_L to a1[0..19]
    ee.st.qacc_l.h.32.ip    a9,16
    ee.st.qacc_h.l.128.ip   a9,16       # QACC_H to a1[32..51]
    ee.st.qacc_h.h.32.ip    a9,0
    l32i.n  a7,a2,12                    # shifts
    l32i.n  a8,a2,0                     # mult
    l32i.n  a10,a2,4                    # 4 * bias (the bias, exact path)
    addi    a9,a2,8                     # rounding term
    addi    a2,a2,16
    bbsi    a7,8,.fast_other
    FASTREQ4 q0,0,0
    FASTREQ4 q4,32,0
.fast_requant_done:
 # With u from FASTREQ4 and rs the right shift, the reference result is
 # (u + 1 + 2^rs - 2 * (u < -1)) >> (rs + 1), and (u + 1) >> 1 for rs 0.
 # The table term also holds out_offset << (rs + 1). Adding a multiple of
 # 2^(rs + 1) before the floor shift adds out_offset after it.
    ee.vldbc.32     q1,a9               # rounding term + out_offset
    extui   a10,a7,16,6                 # rs + 1
    beqi    a10,1,.fast_rs0
    ee.vcmp.eq.s32  q6,q5,q5            # -1
    ee.vcmp.lt.s32  q3,q0,q6
    ee.vcmp.lt.s32  q2,q4,q6
    ee.vadds.s32    q0,q0,q1
    ee.vadds.s32    q4,q4,q1
    ee.vadds.s32    q0,q0,q3
    ee.vadds.s32    q4,q4,q2
    ee.vadds.s32    q0,q0,q3
    ee.vadds.s32    q4,q4,q2
.fast_shift:
    wsr.sar a10
    ee.vsr.32       q0,q0
    ee.vsr.32       q4,q4
    addi    a9,a1,96
.round_done:
    l32i    a10,a1,176                  # tile flag
    l32i    a15,a1,28                   # table end, ends the OC loop

# Broadcast the activation bounds. a9 points at activation_min.
    ee.vldbc.32.ip  q2,a9,4             # activation_min
    ee.vldbc.32     q3,a9               # activation_max

 # activation
    ee.vmin.s32     q0,q0,q3
    ee.vmax.s32     q0,q0,q2
    ee.vmin.s32     q4,q4,q3
    ee.vmax.s32     q4,q4,q2

# With the tile, the 8 bytes of this OC go to tile row oc % 8. After the
# 8th OC the tile is transposed and stored as 8 bytes per position.
    beqz    a10,.store_bytes
    ee.vunzip.16    q0,q4
    ee.vunzip.8     q0,q4               # q0 low 8 bytes: the 8 positions
    extui   a10,a4,0,3                  # oc % 8, out_data is 8-byte aligned
    addx8   a9,a10,a1
    addi    a9,a9,112
    ee.vst.l.64.ip  q0,a9,0
    beqi    a10,7,.tile_flush
.store_done:
    addi.n  a4,a4,1                     # [29] out_data++;
    bne     a2,a15,.out_ch_loop

.outer_ch_loop_end:
    l16ui   a9,a1,256                   # out_channels

    subx8   a11,a9,a9                   # (7 * out_channels);
    l32i    a10,a1,76                   # in_channels
    slli    a10,a10,3                   # in_channels * 8
    l32i    a15,a1,72                   # [4]  gra_spill_temp_165
    l32i    a9,a1,68                    # [2]  gra_spill_temp_172
    l32i    a8,a1,20                    # [0]  gra_spill_temp_175, size-7
    add.n   a4,a4,a11                   # [5] out_data += (7 * out_channels);
    addi.n  a15,a15,8
    s32i    a15,a1,72                   # [7]  gra_spill_temp_165
    add.n   a9,a9,a10                   # [8]
    s32i    a9,a1,68                    # [9]  gra_spill_temp_172
    blt     a15,a8,.outer_loop          # [10]

.return_function:
    retw.n              # [9]

# 16-channel MAC loop for a filter that is not 16-byte aligned, out of the
# common path. The funnel-shift pair reads the block after each filter block;
# that block still holds filter bytes, because the filter is not aligned.
.mac16u:
    extui   a9,a9,0,16                  # drop bit 30; the block count fits 16 bits
    ee.ld.128.usar.ip               q7,a5,16
    ee.ld.128.usar.ip               q6,a5,0
    ee.src.q                        q7,q7,q6
    ee.vld.128.ip                   q0,a8,16
    ee.vld.128.ip                   q1,a8,16
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6       # q7: channels 0-7, q6: 8-15
    loopgtz a9,.mac16u_loop_end
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7
    # next filter block. With the filter not 16-byte aligned, the block
    # after the one that holds the filter's last byte is never loaded.
    ee.ld.128.usar.ip               q7,a5,16
    ee.ld.128.usar.ip               q4,a5,0
    ee.src.q                        q7,q7,q4
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,7
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6
.mac16u_loop_end:
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,0
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,1
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q6,2
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q6,3
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q6,4
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q6,5
    ee.vsmulas.s16.qacc q2,q6,6
    ee.vsmulas.s16.qacc q3,q6,7

    j       .mac_done

# 8-channel MAC loop for in_channels % 16 == 8, out of the common path
.mac8:
    l32i    a10,a1,76                   # in_channels
    srli    a9,a10,3
    addi.n  a9,a9,-1                    # in_channels/8 - 1
    addi    a10,a10,-7                  # in_channels - 7
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip               q7,a5,16
    ee.ld.128.usar.ip               q6,a5,0
    addi                            a5,a5,-8     # net advance = 8
    ee.src.q                        q7,q7,q6
    ee.vld.128.ip                   q0,a8,16
    ee.vld.128.ip                   q1,a8,16
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6

    ee.vsmulas.s16.qacc.ld.incp     q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp     q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp     q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp     q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp     q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp     q3,a8,q1,q7,5
    blti    a10,8,.inner_loop_end           # [16]

    loopgtz a9,.inner_loop_end  # [3]

    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6   # [0*II+0]  id:657
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7   # [0*II+1]  id:658
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip           q7,a5,16
    ee.ld.128.usar.ip           q6,a5,0
    addi                        a5,a5,-8
    ee.src.q                    q7,q7,q6
    ee.vcmp.lt.s8               q6,q7,q5
    ee.vzip.8                   q7,q6
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0   # [0*II+4]  id:660
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1   # [0*II+5]  id:661
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2   # [0*II+6]  id:662
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3   # [0*II+7]  id:663
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4   # [0*II+8]  id:664
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5   # [0*II+9]  id:665
.inner_loop_end:    # 0xfaf

    ee.vsmulas.s16.qacc q2,q7,6     # [2]
    ee.vsmulas.s16.qacc q3,q7,7     # [3]
    j       .mac_done

.fast_other:
    bbsi    a7,9,.requant_slow
    ssl     a7                          # uses bits 0-4: ls
    FASTREQ4 q0,0,1
    FASTREQ4 q4,32,1
    j       .fast_requant_done

.fast_rs0:
    ee.vadds.s32    q0,q0,q1
    ee.vadds.s32    q4,q4,q1
    j       .fast_shift

# The exact 64-bit requant, for channels that fail the FASTREQ4 bound.
# a8 = mult and a10 = bias from the table. REQUANT4 reads the bias from
# a1[52] and clobbers a9-a15, so a11 is set again before the join.
.requant_slow:
    s32i    a10,a1,52                   # bias
    extui   a11,a7,0,5                  # left shift
    extui   a7,a7,16,6
    addi.n  a7,a7,-1                    # right shift
    bnez    a11,.requant_lshift
    REQUANT4 q0,0,0
    REQUANT4 q4,32,0,0                 # SAR is still 31
.requant_done:

 # Rounding right shift: (x + (1 << (rs - 1)) - (x < 0)) >> rs
    beqz.n  a7,.round_skip
    addi.n  a10,a7,-1
    wsr.sar a10
    addi    a9,a1,88
    ee.vldbc.32.ip  q1,a9,4             # 1
    ee.vsl.32       q1,q1               # to_add = 1 << (rs - 1)
    ee.vcmp.lt.s32  q3,q0,q5            # q5 is zero
    ee.vcmp.lt.s32  q2,q4,q5
    ee.vadds.s32    q3,q3,q1
    ee.vadds.s32    q2,q2,q1
    ee.vadds.s32    q0,q0,q3
    ee.vadds.s32    q4,q4,q2
    wsr.sar         a7
    ee.vsr.32       q0,q0
    ee.vsr.32       q4,q4
.round_skip:
    addi    a9,a1,92
    ee.vldbc.32.ip  q1,a9,4             # out_offset
    ee.vadds.s32    q0,q0,q1
    ee.vadds.s32    q4,q4,q1
    addi    a11,a1,3
    j       .round_done

.requant_lshift:
    # REQUANT4 takes the left shift in a3, so park the lane pointer.
    s32i    a3,a1,24
    mov.n   a3,a11
    REQUANT4 q0,0,1
    REQUANT4 q4,32,1
    l32i    a3,a1,24
    j       .requant_done

.store_bytes:
# store the 8 positions, out_channels apart, one byte each
    l16ui   a9,a1,256                   # out_channels
    ee.movi.32.a    q0,a14,0
    ee.movi.32.a    q0,a13,1
    add     a10,a4,a9
    s8i     a14,a4,0
    s8i     a13,a10,0
    ee.movi.32.a    q0,a14,2
    ee.movi.32.a    q0,a13,3
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a13,a10,0
    ee.movi.32.a    q4,a14,0
    ee.movi.32.a    q4,a13,1
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a13,a10,0
    ee.movi.32.a    q4,a14,2
    ee.movi.32.a    q4,a13,3
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a13,a10,0
    j       .store_done

 # Transpose the 8x8 tile (rows = OCs) and store 8 OC bytes per position.
.tile_flush:
    addi    a13,a1,112
    l16ui   a9,a1,256                   # out_channels
    addi    a12,a4,-7
    ee.vld.l.64.ip  q0,a13,8
    ee.vld.l.64.ip  q1,a13,8
    ee.vld.h.64.ip  q0,a13,8
    ee.vld.h.64.ip  q1,a13,8
    ee.vld.l.64.ip  q2,a13,8
    ee.vld.l.64.ip  q3,a13,8
    ee.vld.h.64.ip  q2,a13,8
    ee.vld.h.64.ip  q3,a13,0
    ee.vzip.8       q0,q1
    ee.vzip.16      q0,q1
    ee.vzip.8       q2,q3
    ee.vzip.16      q2,q3
    ee.vzip.32      q0,q2
    ee.vzip.32      q1,q3
    ee.vst.l.64.xp  q0,a12,a9
    ee.vst.h.64.xp  q0,a12,a9
    ee.vst.l.64.xp  q2,a12,a9
    ee.vst.h.64.xp  q2,a12,a9
    ee.vst.l.64.xp  q1,a12,a9
    ee.vst.h.64.xp  q1,a12,a9
    ee.vst.l.64.xp  q3,a12,a9
    ee.vst.h.64.xp  q3,a12,a9
    j       .store_done

    .size   esp_nn_conv_s8_mult8_1x1_esp32s3, . - esp_nn_conv_s8_mult8_1x1_esp32s3
