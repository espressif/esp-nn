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
    .literal_position
    .literal  .nudge_val, 1073741824

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

    # Program Unit: esp_nn_conv_s8_mult8_1x1_esp32s3
    #
    # Pure .S: this hand-scheduled kernel predates esp-nn's inline-asm
    # kernels and is tuned here in place. Moving it to inline asm in C is
    # a separate change.
    #
    # Requirements:
    #   - in_channels must be a multiple of 8
    #   - input_data must be 8-byte aligned (ee.vld.l/h.64.xp ignore the low 3 address bits)
    #   - buffer (scratch) must be 16-byte aligned
    #   - filter_data may have any alignment, but it must be readable for
    #     16 bytes past its end. The 8-channel loop (in_channels % 16 == 8)
    #     and the leftover path load filter blocks in funnel-shift pairs, and
    #     the second load of a pair can read the 16-byte block just past the
    #     filter's end. The 16-channel loop reads a 16-byte aligned filter
    #     with plain aligned loads.
    #
    .type   esp_nn_conv_s8_mult8_1x1_esp32s3, @function
    .align   4
    .global esp_nn_conv_s8_mult8_1x1_esp32s3

esp_nn_conv_s8_mult8_1x1_esp32s3:  # 0xdbc
    # Stack frame (240 bytes). Every slot stays below 192: the top 48 bytes
    # are the register-window save area.
    #   0..51    QACC_L and QACC_H dumps (0..19, 32..51); the leftover path
    #            reuses 0..35 after the main loop
    #   20       size - 7
    #   24       a3 across the output tile flush
    #   28       out_mult end, ends the OC loop
    #   36       leftover path: input channel iterations
    #   52       bias of the current output channel
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
    #   180..191 free

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
    blti    a8,8,.prepare_leftover      # [5] // process remaining lines one by one
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

 # 468          uint32_t bias_ptr = (uint32_t) bias;
 # 469          uint32_t filter_ptr = (uint32_t) (filter_data);
 # 470          const int32_t *out_mult_ptr = out_mult;
 # 471          const int32_t *out_shift_ptr = out_shift;
    l32i    a6,a1,264                   # [0]  out_shift
    l32i    a2,a1,268                   # [1]  out_mult
    l32i    a5,a1,60                    # [2]  gra_spill_temp_170, filter
    l32i    a9,a1,240                   # bias pointer
    addi    a10,a1,108
    moveqz  a9,a10,a9
 # 472          for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
    l16ui   a8,a1,256                   # [5]  id:620 out_channels+0x0
    mov.n   a3,a9                       # bias pointer, kept in a3
    blti    a8,1,.outer_ch_loop_end

    addx4   a10,a8,a2
    s32i    a10,a1,28                   # out_mult end, ends the OC loop
    ee.zero.q   q5                      # sign compare zero, kept across channels

.out_ch_loop:   # 0xf3e
    l32i    a8,a1,280                   # [4]  gra_spill_temp_165, buffer_ptr
    ee.zero.qacc                        # [3]
    l32i    a9,a1,84                   # 16-channel blocks after the first
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

 # Per output channel requant constants. They are set up once for all 8
 # positions, and the requant runs inline, without helper calls.
 # a11 = left shift and a7 = right shift stay in registers. The bias goes
 # to a1[52] for REQUANT4. a3 is the bias pointer across the OC loop.
    l32i.n  a9,a6,0                     # shift
    l32i.n  a8,a2,0                     # mult
    addi.n  a6,a6,4                     # out_shift_ptr++
    addi.n  a2,a2,4                     # out_mult_ptr++
    l32i.n  a12,a3,0                    # bias
    l32i    a10,a1,104                  # bias step
    movi.n  a13,0
    s32i    a12,a1,52
    max     a11,a9,a13                  # left_shift
    sub     a7,a11,a9                   # right_shift
    add.n   a3,a3,a10
    mov.n   a9,a1
    ee.st.qacc_l.l.128.ip   a9,16       # QACC_L to a1[0..19]
    ee.st.qacc_l.h.32.ip    a9,16
    ee.st.qacc_h.l.128.ip   a9,16       # QACC_H to a1[32..51]
    ee.st.qacc_h.h.32.ip    a9,0
    # A left shift is rare (shift > 0), so it runs out of line.
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
.round_done:

# Broadcast out_offset and the activation bounds. a9 points at out_offset.
    ee.vldbc.32.ip  q1,a9,4             # out_offset
    ee.vldbc.32.ip  q2,a9,4             # activation_min
    ee.vldbc.32     q3,a9               # activation_max

# add offset
    ee.vadds.s32    q0,q0,q1
    ee.vadds.s32    q4,q4,q1

 # activation
    ee.vmin.s32     q0,q0,q3
    ee.vmax.s32     q0,q0,q2
    ee.vmin.s32     q4,q4,q3
    ee.vmax.s32     q4,q4,q2

# With the tile, the 8 bytes of this OC go to tile row oc % 8. After the
# 8th OC the tile is transposed and stored as 8 bytes per position.
    l32i    a10,a1,176                  # tile flag
    beqz    a10,.store_bytes
    ee.vunzip.16    q0,q4
    ee.vunzip.8     q0,q4               # q0 low 8 bytes: the 8 positions
    extui   a10,a4,0,3                  # oc % 8, out_data is 8-byte aligned
    addx8   a11,a10,a1
    addi    a11,a11,112
    ee.vst.l.64.ip  q0,a11,0
    beqi    a10,7,.tile_flush
.store_done:
    addi.n  a4,a4,1                     # [29] out_data++;
    l32i    a10,a1,28                   # out_mult end
    bne     a2,a10,.out_ch_loop

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

 # check if leftover
    l32i    a15,a1,72
    l32i    a13,a1,80                   # [1]  gra_spill_temp_176, size
    l32i    a8,a1,76                    # [0]  gra_spill_temp_166, in_channels
    bge     a15, a13, .return_function  # no leftover

// This block below processes one input channel line at a time.
.process_leftover:
    mov.n   a3,a1                       # scratch_buf
    l32i    a15,a1,72                   # [1]  gra_spill_temp_165, i_out
    l32i    a14,a1,56                   # [2]  gra_spill_temp_169, input
    mull    a15,a15,a8                  # [3] in_channels * i_out
    addi.n  a8,a8,-1                    # [4] in_channels - 1
    add.n   a14,a14,a15                 # [5] input_ptr = in_channels * i_out + input
    srai    a8,a8,3                     # [6] iterations, (in_channels - 1) >> 3
    s32i    a8,a1,36                    # [7]  gra_spill_temp_164, iterations
    s32i    a14,a1,68                   # [8]  gra_spill_temp_172, in_channels * i_out + input
    addi            a12,a1,64
    ee.vldbc.16     q4,a12              # [8]  id:716 input_offset

.leftover_outer_loop:

    l32i    a15,a1,264                  # [0]  out_shift
    l32i    a2,a1,268                   # [1]  out_mult
    l32i    a8,a1,60                    # [3]  gra_spill_temp_170, filter_data
    l32i    a5,a1,240                   # [0]  gra_spill_temp_170, bias
    movi.n  a11,0                       # [2]

.leftover_out_ch_loop:

    ee.zero.qacc                            # [0]
    ee.zero.q       q3                      # [1]
    l32i            a9,a1,68                # [4]  gra_spill_temp_172, input_ptr
    l32i            a10,a1,36               # [1]  gra_spill_temp_164, iterations, (in_channels - 1) >> 3
    ee.vld.l.64.ip          q0,a9,8         # [7]  id:717, input
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip       q1,a8,16
    ee.ld.128.usar.ip       q7,a8,0
    addi                    a8,a8,-8
    ee.src.q                q1,q1,q7
    ee.vcmp.lt.s8           q6,q0,q3
    ee.vcmp.lt.s8           q7,q1,q3
    ee.vzip.8               q0,q6
    ee.vzip.8               q1,q7
    ee.vadds.s16            q0,q0,q4  # [11]  id:718, add offset

    loopgtz a10,.leftover_inner_loop_end        # [3]

    ee.vmulas.s16.qacc          q0,q1  # mula(q0,q1)
    ee.vld.l.64.ip              q0,a9,8         # load 8 input values
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip           q1,a8,16
    ee.ld.128.usar.ip           q7,a8,0
    addi                        a8,a8,-8
    ee.src.q                    q1,q1,q7
    ee.vcmp.lt.s8               q2,q0,q3        # sign
    ee.vcmp.lt.s8               q7,q1,q3
    ee.vzip.8                   q0,q2           # 16 bit input
    ee.vzip.8                   q1,q7           # 16 bit filter
    ee.vadds.s16                q0,q0,q4        # add offset
.leftover_inner_loop_end:   # 0x1262

# re-arrange data from qacc in 32 bit q registers
    ee.vmulas.s16.qacc      q0,q1       # [3]
    ee.st.qacc_l.l.128.ip   a3,16       # [5]  id:722
    ee.st.qacc_l.h.32.ip    a3,0        # [6]  id:723
    l8ui    a10,a1,5                    # [11]  scratch_buf+5
    l8ui    a12,a1,6                    # [10]  scratch_buf+6
    l16ui   a14,a1,10                   # [8]  scratch_buf+10
    l8ui    a9,a1,15                    # [7]  scratch_buf+15
    l8ui    a13,a1,16                   # [9]  scratch_buf+16
    s8i     a10,a1,2                    # [12]  scratch_buf+2
    s8i     a12,a1,3                    # [13]  scratch_buf+3
    s16i    a14,a1,4                    # [15]  scratch_buf+4
    s8i     a9,a1,6                     # [16]  scratch_buf+6
    s8i     a13,a1,7                    # [14]  scratch_buf+7

    ee.st.qacc_h.l.128.ip   a3,16       # [17]  id:724
    ee.st.qacc_h.h.32.ip    a3,-32      # [18]  id:725
    l16ui   a13,a1,16                   # [30]  scratch_buf+16
    l8ui    a14,a1,21                   # [23]  scratch_buf+21
    l8ui    a9,a1,22                    # [22]  scratch_buf+22
    l16ui   a10,a1,26                   # [21]  scratch_buf+26
    s16i    a13,a1,8                    # [31]  scratch_buf+8
    l8ui    a12,a1,31                   # [20]  scratch_buf+31
    l8ui    a13,a1,32                   # [19]  scratch_buf+32
    s8i     a14,a1,10                   # [24]  scratch_buf+10
    s8i     a9,a1,11                    # [25]  scratch_buf+11
    s16i    a10,a1,12                   # [26]  scratch_buf+12
    s8i     a12,a1,14                   # [27]  scratch_buf+14
    s8i     a13,a1,15                   # [28]  scratch_buf+15
    movi.n  a12,16

# get data now
    ee.vld.128.ip       q0,a3,0
    ee.srcmb.s16.qacc   q1,a12,0
    ee.vzip.16          q0,q1

    ee.vadds.s32    q0,q0,q1
    ee.movi.32.a    q0,a10,3
    ee.movi.32.a    q0,a9,2
    ee.movi.32.a    q0,a14,0
    add             a9,a9,a10
    ee.movi.32.a    q0,a10,1
    add             a14,a14,a10
    add             a14,a14,a9

# a14 contains conv_out
    l32i    a9,a1,240                   # [43]  gra_spill_temp_170, bias ptr
    l32i.n  a6,a15,0                    # [44]  id:730, shift
    beqz.n  a9,.leftover_multiply_by_quant_mult             # [45]

# load and add bias
    l32i.n  a9,a5,0
    add.n   a14,a14,a9

.leftover_multiply_by_quant_mult:   # 0x12e7
    l32i.n  a9,a2,0                 # [0]  id:729, mult
    movi.n  a10,0                   # [1]
    max     a10,a6,a10              # [2]  left_shift
    ssl     a10                     # [3]
    sll     a14,a14                 # [4] (value << left_shift)

    sub     a7,a10,a6               # right_shift

    l32r    a13,.nudge_val
    mulsh   a12,a9,a14
    mull    a14,a9,a14
    ssai    31

    addi.n  a2,a2,4                 # [0] mult
    addi.n  a15,a15,4               # [1] shift
    addi.n  a5,a5,4                 # [2] bias
    addi.n  a11,a11,1               # [3]

    add     a13,a14,a13             # low part
    saltu   a14,a13,a14
    add     a9,a12,a14              # high part
    src     a12,a9,a13

    blti    a7,1,.leftover_skip_div_by2

    addi.n  a14,a7,-1
    ssl     a14
    movi.n  a10,1
    sll     a10,a10                     # 1 << (exponent - 1)
    extui   a14,a12,31,1
    ssr     a7
    sub     a10,a10,a14                 # 1 << (exponent - 1) - (val < 0)
    add     a12,a12,a10                 # val += to_add
    sra     a12,a12

.leftover_skip_div_by2:
    l32i    a10,a1,260                  # [26]  id:733 out_offset+0x0
    l32i    a9,a1,272                   # [29]  id:732 activation_min+0x0
    l16ui   a13,a1,256                  # [5]  id:620 out_channels+0x0
    l32i    a14,a1,276                  # [31]  id:731 activation_max+0x0

// add offset, apply activation and store
    add.n   a10,a10,a12
    max     a9,a9,a10
    min     a14,a14,a9
    s8i     a14,a4,0
    addi.n  a4,a4,1

    bne     a11,a13,.leftover_out_ch_loop

    l32i    a15,a1,76                   # [0]  gra_spill_temp_166, in_channels
    l32i    a14,a1,68                   # [1]  gra_spill_temp_172, input_ptr
    l32i    a13,a1,72                   # [2]  gra_spill_temp_165, i_out
    l32i    a12,a1,80                   # [3]  gra_spill_temp_176, size
    addi.n  a13,a13,1                   # [4]
    s32i    a13,a1,72                   # [5]  gra_spill_temp_165, i_out
    add     a14,a14,a15                 # [7]  input_ptr += in_channels
    s32i    a14,a1,68                   # [8]  gra_spill_temp_172, input_ptr
    blt     a13,a12,.leftover_outer_loop

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

.round_skip:
    addi    a9,a1,92
    j       .round_done

.requant_lshift:
    # REQUANT4 takes the left shift in a3, so park the bias pointer.
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
    ee.movi.32.a    q0,a11,1
    add     a10,a4,a9
    s8i     a14,a4,0
    s8i     a11,a10,0
    ee.movi.32.a    q0,a14,2
    ee.movi.32.a    q0,a11,3
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a11,a10,0
    ee.movi.32.a    q4,a14,0
    ee.movi.32.a    q4,a11,1
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a11,a10,0
    ee.movi.32.a    q4,a14,2
    ee.movi.32.a    q4,a11,3
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0
    s8i     a11,a10,0
    j       .store_done

 # Transpose the 8x8 tile (rows = OCs) and store 8 OC bytes per position.
.tile_flush:
    addi    a11,a1,112
    l16ui   a9,a1,256                   # out_channels
    addi    a12,a4,-7
    ee.vld.l.64.ip  q0,a11,8
    ee.vld.l.64.ip  q1,a11,8
    ee.vld.h.64.ip  q0,a11,8
    ee.vld.h.64.ip  q1,a11,8
    ee.vld.l.64.ip  q2,a11,8
    ee.vld.l.64.ip  q3,a11,8
    ee.vld.h.64.ip  q2,a11,8
    ee.vld.h.64.ip  q3,a11,0
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

.prepare_leftover:
    l32i    a8,a1,76                    # [0]  gra_spill_temp_166, in_channels
    movi.n  a15,0
    s32i    a15,a1,72                   # [7]  gra_spill_temp_165, i_out
    j   .process_leftover

    .size   esp_nn_conv_s8_mult8_1x1_esp32s3, . - esp_nn_conv_s8_mult8_1x1_esp32s3
