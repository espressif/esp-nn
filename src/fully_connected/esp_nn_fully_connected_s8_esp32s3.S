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
    .align  4
    .literal_position
    .literal    .LC3_26_101, 1073741824 // nudge (1 << 30)

    # Program Unit: esp_nn_fully_connected_s8_esp32s3
    .type   esp_nn_fully_connected_s8_esp32s3, @function
    .align   4
    .global esp_nn_fully_connected_s8_esp32s3

// a2: input_data
// a3: input_offset
// a4: row_len
// a5: filter_data
// a6: filter_offset
// a7: bias
// on stack: out_data
// on stack: out_channels
// on stack: out_offset
// on stack: out_shift
// on stack: out_mult
// on stack: activation_min
// on stack: activation_max

esp_nn_fully_connected_s8_esp32s3:  # 0x4
    # qacc_scratch = 0
    // 40, filter_offset
    // 44, input_offset
    # gra_spill_temp_7 = 48
    # gra_spill_temp_0 = 52
    # gra_spill_temp_1 = 56
    # gra_spill_temp_2 = 60
    # gra_spill_temp_3 = 64
    # gra_spill_temp_4 = 68
    # gra_spill_temp_5 = 72
    # gra_spill_temp_6 = 76

    entry   a1,112                      #
    s32i.n  a5,a1,60                # [0]  gra_spill_temp_2, filter_data
    s32i    a7,a1,48                    # [1]  gra_spill_temp_7, bias
    s32i    a6,a1,40                    # [2]  id:252 filter_offset+0x0
    s32i    a3,a1,44                    # [3]  id:251 input_offset+0x0
    mov.n   a13,a2                      # [5]
    mov.n   a12,a4                      # [6]

 // out_channel loop
    l16ui       a2,a1,116                   # [7]  id:255 out_channels+0x0
    addi        a4,a1,40                # [8]
    addi        a8,a1,44                # [9]
    ee.vldbc.16 q5,a8               # [10]  id:253 input_offset
    ee.vldbc.16 q6,a4               # [12]  id:254 filter_offset
    beqz.n      a2,.Lt_0_7938           # [13]

    ee.zero.q   q7                      # [0]
    srai        a11,a12,3                   # [2]
    l32i        a10,a1,128                  # [5]  id:257 out_mult+0x0
    l32i        a8,a1,112                   # [6]  id:259 out_data+0x0
    addi        a9,a12,-7                   # [7]
    s32i        a9,a1,76                    # [8]  gra_spill_temp_6
    s32i        a8,a1,72                    # [9]  gra_spill_temp_5
    s32i        a11,a1,64                   # [14]  gra_spill_temp_3
    slli        a11,a11,3                   # [16]
    s32i        a11,a1,68                   # [18]  gra_spill_temp_4
    l32i        a10,a1,124                  # [25]  id:256 out_shift+0x0
    movi.n      a15,0                   # [17]
    mov.n       a14,a7                      # [15]
    max         a11,a10,a15                 # [29]
    s32i        a11,a1,52                   # [30]  gra_spill_temp_0 // left_shift
    sub         a10,a11,a10                 #  // right_shift
    s32i.n      a10,a1,56                   # [28]  gra_spill_temp_1 // right_shift
    mov.n       a11,a5                      # [31]
    movi.n      a10,0                   # [32]
    mov.n       a2,a11                      # [33]

.Lt_0_8450: # 0x12b

    extui           a5,a11,0,3                  # [34]
    ee.zero.qacc                    # [0]
    mov.n           a8,a1                       # [1]
    l32i            a9,a1,76                    # [2]  gra_spill_temp_6
    slli            a5,a5,1                     # [3]
    ee.vld.l.64.ip  q4,a2,8         # [4]  id:267
    wur.sar_byte    a5                  # [5]
    mov.n           a5,a10                      # [6]
    ee.vcmp.lt.s8   q2,q4,q7            # [7]
    ee.vzip.8       q4,q2                   # [8]
    bgei            a9,0,.LBB6_esp_nn_fully_connected_s8_esp32s3            # [9]

    movi.n  a2,0                    # [0]
    j       .Lt_0_8706                      # [1]

.LBB6_esp_nn_fully_connected_s8_esp32s3:    # 0x147
    l32i    a4,a1,64                    # [0]  gra_spill_temp_3
    mov.n   a3,a13                      # [1]
    addx8   a5,a4,a10                   # [2]
    loopgtz a4,.LBB45_esp_nn_fully_connected_s8_esp32s3     # [3]

    ee.vld.l.64.ip      q0,a2,8         # [0*II+0]  id:268
    ee.vld.l.64.ip      q1,a3,8         # [0*II+1]  id:270
    ee.vcmp.lt.s8       q2,q0,q7            # [0*II+2]
    ee.vcmp.lt.s8       q3,q1,q7            # [0*II+3]
    ee.vzip.8           q0,q2                   # [0*II+4]
    ee.vzip.8           q1,q3                   # [0*II+5]
    ee.vadds.s16        q1,q1,q5            # [0*II+6]
    ee.src.q.qup        q2,q4,q0            # [0*II+7]
    ee.vadds.s16        q2,q2,q6            # [0*II+8]
    ee.vmulas.s16.qacc  q1,q2       # [0*II+9]

.LBB45_esp_nn_fully_connected_s8_esp32s3:   # 0x170
    l32i    a2,a1,68                    # [0]  gra_spill_temp_4

.Lt_0_8706: # 0x173
    ee.st.qacc_l.l.128.ip   a8,16       # [0]  id:272
    ee.st.qacc_l.h.32.ip    a8,0        # [1]  id:273
    l8ui            a9,a1,16                    # [2]  qacc_scratch+16
    l8ui            a6,a1,6                     # [3]  qacc_scratch+6
    s8i             a6,a1,3                     # [4]  qacc_scratch+3
    s8i             a9,a1,7                     # [5]  qacc_scratch+7
    l8ui            a6,a1,15                    # [6]  qacc_scratch+15
    l8ui            a9,a1,5                     # [7]  qacc_scratch+5
    s8i             a9,a1,2                     # [8]  qacc_scratch+2
    s8i             a6,a1,6                     # [9]  qacc_scratch+6
    l16ui           a9,a1,10                    # [10]  qacc_scratch+10
    s16i            a9,a1,4                     # [11]  qacc_scratch+4
    ee.st.qacc_h.l.128.ip   a8,16       # [12]  id:283
    ee.st.qacc_h.h.32.ip    a8,-32      # [13]  id:284
    l8ui            a6,a1,32                    # [15]  qacc_scratch+32
    l8ui            a9,a1,31                    # [14]  qacc_scratch+31
    s8i             a6,a1,15                    # [16]  qacc_scratch+15
    s8i             a9,a1,14                    # [17]  qacc_scratch+14
    l8ui            a9,a1,22                    # [19]  qacc_scratch+22
    l16ui           a6,a1,26                    # [18]  qacc_scratch+26
    s8i             a9,a1,11                    # [20]  qacc_scratch+11
    s16i            a6,a1,12                    # [21]  qacc_scratch+12
    l8ui            a6,a1,21                    # [23]  qacc_scratch+21
    l16ui           a9,a1,16                    # [22]  qacc_scratch+16
    s8i             a6,a1,10                    # [24]  qacc_scratch+10
    s16i            a9,a1,8                     # [25]  qacc_scratch+8
    movi.n          a6,16                   # [26]
    ee.vld.128.ip   q1,a8,0             # [27]  id:296
    ee.srcmb.s16.qacc   q2,a6,0         # [28]
    ee.vzip.16      q1,q2               # [29]
    ee.vadds.s32    q1,q1,q2            # [30]
    ee.movi.32.a    q1,a9,3             # [31]
    ee.movi.32.a    q1,a8,2             # [32]
    ee.movi.32.a    q1,a6,0             # [33]
    add.n           a8,a8,a9                    # [34]
    ee.movi.32.a    q1,a9,1             # [35]
    add.n           a6,a6,a9                    # [36]
    add.n           a6,a6,a8                    # [37]
    bge             a2,a12,.Lt_0_9730           # [38]

// prepare remaining loop
    l32i    a8,a1,44                    # [0]  id:251 input_offset+0x0
    l32i    a7,a1,40                    # [1]  id:252 filter_offset+0x0
    sub     a3,a12,a2                   # [2]
    l32i.n  a4,a1,60                # [3]  gra_spill_temp_2
    add.n   a2,a2,a13                   # [4]
    add.n   a4,a4,a5                    # [5]
    loopgtz a3,.LBB60_esp_nn_fully_connected_s8_esp32s3     # [6]

// remaining c loop
    l8ui    a3,a2,0                     # [0*II+0]  id:299
    l8ui    a5,a4,0                     # [0*II+1]  id:300
    sext    a3,a3,7                     # [0*II+2]
    sext    a5,a5,7                     # [0*II+3]
    add.n   a5,a5,a7                    # [0*II+5]
    add.n   a3,a3,a8                    # [0*II+6]
    mull    a3,a3,a5                    # [0*II+7]
    addi.n  a2,a2,1                 # [0*II+8]
    addi.n  a4,a4,1                 # [0*II+4]
    add.n   a6,a6,a3                    # [0*II+9]

.LBB60_esp_nn_fully_connected_s8_esp32s3:   # 0x20f

// add bias
.Lt_0_9730: # 0x20f
    l32i    a8,a1,48                    # [0]  gra_spill_temp_7, bias
    beqz.n  a8,.Lt_0_10754          # [2], skip_bias

    l32i.n  a9,a14,0                # [0]  id:301
    add.n   a6,a6,a9                    # [2]

// apply quantization
.Lt_0_10754:    # 0x218
    l32i    a2,a1,52                    # [1]  gra_spill_temp_0 // left_shift
    l32i    a5,a1,56                    # [2]  gra_spill_temp_1 // right_shift
    ssl     a2                          # [3]
    sll     a6,a6                       # [5] // x * (1 << left_shift)

    l32r    a3,.LC3_26_101              # [0]

    add.n   a10,a10,a12                 # [0]
    addi.n  a14,a14,4               # [1]

    l32i    a4,a1,128                   # [2]  gra_spill_temp_10 //out_mult
    add.n   a11,a11,a12                 # [6]

// multiply add nudge and pick high32
    ssai    31
    mulsh   a7,a4,a6                    # [4]
    mull    a4,a4,a6                    # [5]

    mov.n   a2,a11                      # [27]
    add     a4,a4,a3
    saltu   a8,a4,a3
    add.n   a7,a7,a8
    src     a3,a7,a4

// divide_by_power_of2_step
    blti    a5,1,.skip_divide_by2
    movi.n  a8,1                    # [28]
    addi    a4,a5,-1
    ssl     a4          // load left_shift
    sll     a8,a8       // to_add factor ( 1 << (exponent - 1))
    extui   a6,a3,31,1                  # [33]
    sub     a8,a8,a6        // modified to_add factor ( 1 << (exponent - 1) - (val < 0))
    add     a3,a3,a8    // val + to_add
    ssr     a5                          # [29] //load right_shift
    sra     a3,a3                       # [31]
.skip_divide_by2:

    l32i    a8,a1,120                   # [41]  out_offset
    l32i    a7,a1,132                   # [44] // activation_min
    l32i    a4,a1,136                   # [45] // activation_max

    add.n   a8,a8,a3                    # [46] // add out_offset
    l32i    a6,a1,72                    # [47]  gra_spill_temp_5
    l32i.n  a3,a1,116                   # [48]  out_channels
    max     a7,a7,a8                    # [49]
    add.n   a6,a15,a6                   # [50]
    min     a4,a4,a7                    # [51]
    addi.n  a15,a15,1               # [52]
    s8i     a4,a6,0                     # [53]  id:302
    bne     a3,a15,.Lt_0_8450               # [55]

.Lt_0_7938: # 0x25c
    retw.n                          # [0]

    .size   esp_nn_fully_connected_s8_esp32s3, . - esp_nn_fully_connected_s8_esp32s3
