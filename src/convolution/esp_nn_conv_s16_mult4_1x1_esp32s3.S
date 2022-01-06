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
    .literal    .LC14_29_106, 1073741824

    # Program Unit: esp_nn_conv_s16_mult4_1x1_esp32s3
    .type   esp_nn_conv_s16_mult4_1x1_esp32s3, @function
    .align   4
    .global esp_nn_conv_s16_mult4_1x1_esp32s3
esp_nn_conv_s16_mult4_1x1_esp32s3:  # 0xa62
    # scratch_buf = 0
    # to_add = 32
    # gra_spill_temp_139 = 36
    # gra_spill_temp_140 = 40
    # gra_spill_temp_141 = 44
    # gra_spill_temp_142 = 48
    # gra_spill_temp_143 = 52
    # gra_spill_temp_144 = 56
    # gra_spill_temp_145 = 60
    # gra_spill_temp_146 = 64
    # gra_spill_temp_147 = 68
    # gra_spill_temp_148 = 72
    # gra_spill_temp_149 = 76
    # gra_spill_temp_150 = 80
    # gra_spill_temp_151 = 84
    # gra_spill_temp_152 = 88
    # gra_spill_temp_153 = 92
    # gra_spill_temp_154 = 96
    # gra_spill_temp_155 = 100
    # gra_spill_temp_156 = 104
    # gra_spill_temp_157 = 108
    # gra_spill_temp_158 = 112
    # gra_spill_temp_159 = 116
    # gra_spill_temp_160 = 120
    # gra_spill_temp_161 = 128
    # gra_spill_temp_162 = 144
    # lgra_spill_temp_163 = 160
    # lgra_spill_temp_164 = 176
    # lgra_spill_temp_165 = 180
    # lgra_spill_temp_166 = 184
    # lgra_spill_temp_167 = 188
    # lgra_spill_temp_168 = 192
    # lra_spill_temp_169 = 196

 // registers:
 // a2: int16_t *input_data
 // a3: uint16_t input_wd
 // a4: uint16_t input_ht
 // a5: uint16_t in_channels
 // a6: int16_t *filter_data
 // a7: int32_t *bias

 // on stack:
 // int8_t *out_data
 // uint16_t out_wd
 // uint16_t out_ht
 // uint16_t out_channels
 // int32_t out_offset
 // int32_t *out_shift
 // int32_t *out_mult
 // int32_t activation_min
 // int32_t activation_max
 // *buffer /* scratch buffer */


    entry   a1,240                      #
    s32i.n  a2,a1,40                # [0]  gra_spill_temp_140
    s32i    a6,a1,68                    # [1]  gra_spill_temp_147
    s32i    a7,a1,116                   # [2]  gra_spill_temp_159
 # 301
 # 302      int16_t *transposed_input = (int16_t *) buffer;
 # 303      int32_t size = input_wd * input_ht;
 # 304
 # 305      for (int32_t i_out = 0; i_out < size - 3; i_out += 4) {
    mul16u  a3,a3,a4                # [3]
    addi    a10,a1,112                  # [4]
    addmi   a11,a1,256                  # [5]
    addmi   a8,a1,256                   # [6]
    addmi   a9,a1,256                   # [7]
    addi.n  a9,a9,12                # [8]
    addi    a8,a8,16                    # [9]
    ee.vldbc.32 q0,a11              # [10]  id:268 out_offset
    st.qr   q0,a10,32                   # [11]  gra_spill_temp_162-112
    ee.vldbc.32 q7,a8               # [12]  id:270 activation_max
    ee.vldbc.32 q0,a9               # [13]  id:269 activation_min
    blti    a3,4,.Lt_3_6402             # [14]

.LBB3_esp_nn_conv_s16_mult4_1x1_esp32s3:    # 0xa90
    l32i    a13,a1,240                  # [0]  id:280 out_data+0x0
    srai    a8,a5,2                     # [1]
    addi    a10,a3,-3                   # [2]
    addi    a9,a5,-3                    # [3]
    movi.n  a12,0                   # [4]
    slli    a11,a5,2                    # [5]
    slli    a15,a5,1                    # [6]
    l16ui   a14,a1,252                  # [7]  id:271 out_channels+0x0
    s32i    a14,a1,108                  # [8]  gra_spill_temp_157
    s32i.n  a15,a1,36               # [9]  gra_spill_temp_139
    s32i.n  a11,a1,56               # [10]  gra_spill_temp_144
    s32i    a12,a1,84                   # [11]  gra_spill_temp_151
    s32i    a9,a1,104                   # [12]  gra_spill_temp_156
    s32i.n  a10,a1,60               # [13]  gra_spill_temp_145
    s32i    a8,a1,88                    # [14]  gra_spill_temp_152
    movi.n  a10,0                   # [15]
    l32i    a8,a1,276                   # [16]  id:281 buffer+0x0
    l32i    a9,a1,260                   # [17]  id:279 out_shift+0x0
    addi    a12,a1,112                  # [18]
    slli    a11,a11,1                   # [19]
    l32i    a15,a1,264                  # [20]  id:272 out_mult+0x0
    s32i.n  a15,a1,48               # [21]  gra_spill_temp_142
    s32i    a11,a1,64                   # [22]  gra_spill_temp_146
    st.qr   q0,a12,16                   # [23]  gra_spill_temp_161-112
    s32i.n  a9,a1,52                # [24]  gra_spill_temp_143
    s32i    a8,a1,112                   # [25]  gra_spill_temp_158
    s32i    a10,a1,92                   # [26]  gra_spill_temp_153
    movi.n  a8,0                    # [27]
    movi.n  a10,0                   # [28]
    l32r    a9,.LC14_29_106             # [29]
    s32i    a9,a1,96                    # [30]  gra_spill_temp_154
    s32i    a10,a1,80                   # [31]  gra_spill_temp_150
    s32i    a8,a1,76                    # [32]  gra_spill_temp_149
    mov.n   a10,a1                      # [33]
    slli    a8,a14,1                    # [34]
    addx2   a9,a14,a14                  # [35]
    s32i    a9,a1,72                    # [36]  gra_spill_temp_148
    s32i.n  a8,a1,44                # [37]  gra_spill_temp_141
    addx4   a14,a14,a15                 # [38]
    s32i    a14,a1,100                  # [39]  gra_spill_temp_155
    j   .Lt_3_6914                      # [40]

.Lt_3_8194: # 0xb00
#<loop> Part of loop body line 305, head labeled .Lt_3_6914
    l32i.n  a12,a1,60               # [0]  gra_spill_temp_145
    l32i.n  a9,a1,56                # [1]  gra_spill_temp_144
    l32i    a8,a1,76                    # [2]  gra_spill_temp_149
    l32i    a15,a1,64                   # [3]  gra_spill_temp_146
    l32i    a11,a1,72                   # [4]  gra_spill_temp_148
    l32i    a14,a1,84                   # [5]  gra_spill_temp_151
    add.n   a13,a13,a11                 # [6]
    l32i    a11,a1,80                   # [7]  gra_spill_temp_150
    add.n   a14,a14,a15                 # [8]
    add.n   a8,a8,a9                    # [9]
    s32i    a8,a1,76                    # [10]  gra_spill_temp_149
    s32i    a14,a1,84                   # [11]  gra_spill_temp_151
    addi.n  a11,a11,4               # [12]
    s32i    a11,a1,80                   # [13]  gra_spill_temp_150
    bge     a11,a12,.Lt_3_6402          # [14]

.Lt_3_6914: # 0xb27
    l32i    a12,a1,104                  # [0]  gra_spill_temp_156
    l32i    a4,a1,112                   # [1]  gra_spill_temp_158
    blti    a12,1,.Lt_3_7170            # [2]

.LBB6_esp_nn_conv_s16_mult4_1x1_esp32s3:    # 0xb30
    l32i    a3,a1,88                    # [0]  gra_spill_temp_152
    l32i.n  a5,a1,40                # [1]  gra_spill_temp_140
    mv.qr   q0,q3                       # [2]
    l32i    a2,a1,84                    # [3]  gra_spill_temp_151
    mv.qr   q3,q4                       # [4]
    mv.qr   q4,q2                       # [5]
    mv.qr   q2,q0                       # [6]
    add.n   a2,a2,a5                    # [7]
    mv.qr   q0,q1                       # [8]
    l32i.n  a5,a1,36                # [9]  gra_spill_temp_139
    mv.qr   q1,q4                       # [10]
    loopgtz a3,.LBB38_esp_nn_conv_s16_mult4_1x1_esp32s3     # [11]

    mov.n   a3,a2                       # [0*II+0]
    addi.n  a2,a2,8                 # [0*II+1]
    ee.vld.l.64.xp  q0,a3,a5        # [0*II+2]  id:282
    ee.vld.l.64.xp  q1,a3,a5        # [0*II+3]  id:283
    ee.vld.l.64.xp  q2,a3,a5        # [0*II+4]  id:284
    ee.vld.l.64.xp  q3,a3,a5        # [0*II+5]  id:285
    ee.vzip.16      q0,q1               # [0*II+6]
    ee.vzip.16      q2,q3               # [0*II+7]
    ee.vzip.32      q0,q2               # [0*II+8]
    ee.vst.128.ip   q0,a4,16            # [0*II+9]  id:286
    ee.vst.128.ip   q2,a4,16            # [0*II+10]  id:287
.LBB38_esp_nn_conv_s16_mult4_1x1_esp32s3:   # 0xb70

    mv.qr   q4,q3                       # [0]
    mv.qr   q3,q2                       # [1]
    mv.qr   q2,q1                       # [2]
    mv.qr   q1,q0                       # [3]

.Lt_3_7170: # 0xb7c
    l32i    a2,a1,68                    # [0]  gra_spill_temp_147
    l32i    a9,a1,116                   # [1]  gra_spill_temp_159
    l32i    a8,a1,108                   # [2]  gra_spill_temp_157
    s32i    a9,a1,120                   # [3]  gra_spill_temp_160
    beqz.n  a8,.Lt_3_8194           # [4]

    l32i.n  a9,a1,52                # [0]  gra_spill_temp_143
    l32i.n  a11,a1,48               # [1]  gra_spill_temp_142
    l32i    a15,a1,72                   # [2]  gra_spill_temp_148
    l32i.n  a14,a1,44               # [3]  gra_spill_temp_141
    add.n   a15,a15,a13                 # [4]
    add.n   a14,a14,a13                 # [5]
    j   .Lt_3_8706                      # [6]

.Lt_3_10754:    # 0xb9a

    movi.n  a3,0                    # [0]

.Lt_3_10498:    # 0xb9c

// esp_nn_multiply_by_quantized_mult_esp32s3
    ee.zero.q   q0                      # [0]
    l32i        a5,a1,92                    # [1]  gra_spill_temp_153
    s32i        a2,a1,180                   # [2]  lgra_spill_temp_165
    s32i        a11,a1,188                  # [3]  lgra_spill_temp_167
    s32i        a13,a1,192                  # [4]  lgra_spill_temp_168
    s32i        a9,a1,184                   # [5]  lgra_spill_temp_166

    movi.n          a13,0                   # [6]
    max             a12,a12,a13                 # [7]
    wsr.sar         a12                     # [8]
    ee.vsl.32       q1,q1                   # [9]
    ssai            31                          # [10]
    ee.movi.32.a    q1,a7,0             # [11]
    ee.movi.32.a    q1,a8,1             # [12]
    ee.movi.32.a    q1,a6,3             # [13]
    ee.movi.32.a    q1,a9,2             # [14]
    mulsh           a12,a4,a9                   # [15]
    mulsh           a11,a4,a6                   # [16]
    mulsh           a2,a4,a8                    # [17]
    mulsh           a13,a7,a4                   # [18]
    mull            a8,a4,a8                    # [19]
    mull            a7,a7,a4                    # [20]
    add.n           a11,a5,a11                  # [21]
    add.n           a12,a5,a12                  # [22]
    add.n           a2,a5,a2                    # [23]
    mull            a6,a4,a6                    # [24]
    add.n           a5,a5,a13                   # [25]
    l32i            a13,a1,96                   # [26]  gra_spill_temp_154
    mull            a9,a4,a9                    # [27]
    add.n           a6,a13,a6                   # [28]
    add.n           a9,a13,a9                   # [29]
    add.n           a7,a13,a7                   # [30]
    s32i            a7,a1,196                   # [31]  lra_spill_temp_169
    add.n           a8,a13,a8                   # [32]
    saltu           a7,a7,a13                   # [33]
    add.n           a7,a7,a5                    # [34]
    saltu           a5,a8,a13                   # [35]
    add.n           a5,a5,a2                    # [36]
    src             a5,a5,a8                    # [37]
    saltu           a2,a9,a13                   # [38]
    l32i            a8,a1,196                   # [39]  lra_spill_temp_169
    add.n           a2,a2,a12                   # [40]
    saltu           a13,a6,a13                  # [41]
    addi.n          a12,a3,-1               # [42]
    src             a2,a2,a9                    # [43]
    add.n           a13,a13,a11                 # [44]
    addi            a9,a1,32                    # [45]  to_add
    movi.n          a11,1                   # [46]
    src             a7,a7,a8                    # [47]
    src             a13,a13,a6                  # [48]
    addi            a8,a1,112                   # [49]
    ee.movi.32.q    q5,a7,0             # [50]
    ee.movi.32.q    q5,a5,1             # [51]
    l32i            a7,a1,100                   # [52]  gra_spill_temp_155
    l32i            a5,a1,108                   # [53]  gra_spill_temp_157
    ee.movi.32.q    q5,a2,2             # [54]
    ssl             a12                         # [55]
    sll             a11,a11                     # [56]
    ee.movi.32.q    q5,a13,3            # [57]
    wsr.sar         a3                      # [58]
    ee.vcmp.lt.s32  q0,q5,q0        # [59]
    l32i            a13,a1,192                  # [60]  lgra_spill_temp_168
    s32i.n          a11,a1,32               # [61]  to_add
    ee.vldbc.32     q1,a9               # [62]  id:317 to_add
    add.n           a5,a5,a13                   # [63]
    l32i            a9,a1,184                   # [64]  lgra_spill_temp_166
    ee.vadds.s32    q1,q1,q0            # [65]
    addi.n          a9,a9,4                 # [66]
    ee.vadds.s32    q1,q5,q1            # [67]
    ld.qr           q0,a8,32                    # [68]  gra_spill_temp_162-112
    ee.vsr.32       q1,q1                   # [69]
    ee.vadds.s32    q1,q1,q0            # [70]
    ld.qr           q0,a8,16                    # [71]  gra_spill_temp_161-112
    ee.vmin.s32     q1,q1,q7            # [72]
    ee.vmax.s32     q1,q1,q0            # [73]
    ee.vst.128.ip   q1,a10,0            # [74]  id:320
    l8ui        a6,a1,0                     # [75]  scratch_buf
    s8i         a6,a13,0                    # [76]
    addi.n      a13,a13,1               # [77]
    l8ui        a2,a1,4                     # [78]  scratch_buf+4
    s8i         a2,a5,0                     # [79]
    l8ui        a12,a1,8                    # [80]  scratch_buf+8
    l32i        a2,a1,180                   # [81]  lgra_spill_temp_165
    s8i         a12,a14,0                   # [82]
    addi.n      a14,a14,1               # [83]
    l8ui        a11,a1,12                   # [84]  scratch_buf+12
    s8i         a11,a15,0                   # [85]
    l32i        a11,a1,188                  # [86]  lgra_spill_temp_167
    addi.n      a15,a15,1               # [87]
    addi.n      a11,a11,4               # [88]
    sub         a7,a11,a7                   # [89]
    beqz        a7,.Lt_3_8194               # [90]

.Lt_3_8706: # 0xc97
    ee.zero.qacc                    # [0]
    l32i    a8,a1,104                   # [1]  gra_spill_temp_156
    l32i    a3,a1,112                   # [2]  gra_spill_temp_158
    blti    a8,1,.Lt_3_8962             # [3]

    l32i    a4,a1,88                    # [0]  gra_spill_temp_152
    loopgtz a4,.LBB53_esp_nn_conv_s16_mult4_1x1_esp32s3     # [2]

    ee.vld.l.64.ip  q6,a2,8         # [0*II+0]  id:289
    ee.vld.l.64.ip  q1,a3,8         # [0*II+1]  id:290
    ee.vld.l.64.ip  q2,a3,8         # [0*II+2]  id:291
    ee.vsmulas.s16.qacc q1,q6,0     # [0*II+3]
    ee.vld.l.64.ip  q3,a3,8         # [0*II+4]  id:292
    ee.vsmulas.s16.qacc q2,q6,1     # [0*II+5]
    ee.vld.l.64.ip  q4,a3,8         # [0*II+6]  id:293
    ee.vsmulas.s16.qacc q3,q6,2     # [0*II+7]
    ee.vsmulas.s16.qacc q4,q6,3     # [0*II+8]

.LBB53_esp_nn_conv_s16_mult4_1x1_esp32s3:   # 0xcc4

.Lt_3_8962: # 0xcc4

// extract data:
    ee.st.qacc_l.l.128.ip   a10,16      # [0]  id:298
    ee.st.qacc_l.h.32.ip    a10,-16     # [1]  id:299
    l8ui    a12,a1,16                   # [2]  scratch_buf+16
    l8ui    a8,a1,6                     # [3]  scratch_buf+6
    s8i     a8,a1,3                     # [4]  scratch_buf+3
    s8i     a12,a1,7                    # [5]  scratch_buf+7
    l8ui    a8,a1,15                    # [6]  scratch_buf+15
    l8ui    a12,a1,5                    # [7]  scratch_buf+5
    s8i     a12,a1,2                    # [8]  scratch_buf+2
    s8i     a8,a1,6                     # [9]  scratch_buf+6
    l16ui   a12,a1,10                   # [10]  scratch_buf+10
    movi.n  a8,16                   # [11]
    ee.srcmb.s16.qacc   q2,a8,0         # [12]
    s16i    a12,a1,4                    # [13]  scratch_buf+4
    ee.vld.l.64.ip  q1,a10,0        # [14]  id:309
    l32i    a12,a1,116                  # [15]  gra_spill_temp_159, bias
    ee.vzip.16  q1,q2               # [16]

    beqz.n  a12,.Lt_3_9986          # [17] // skip bias
 // add bias:
    l32i            a8,a1,120                   # [0]  gra_spill_temp_160
    ee.vldbc.32.ip  q0,a8,4         # [2]  id:311
    s32i            a8,a1,120                   # [3]  gra_spill_temp_160
    ee.vadds.s32    q1,q1,q0            # [4]
.Lt_3_9986: # 0xd04

    l32i.n  a12,a9,0                # [0]  id:313
    l32i.n  a4,a11,0                # [1]  id:312
    bgei    a12,1,.Lt_3_10754           # [2]

    neg     a3,a12                      # [0]
    j       .Lt_3_10498                     # [1]

.Lt_3_6402: # 0xd11
    retw.n                          # [0]

    .size   esp_nn_conv_s16_mult4_1x1_esp32s3, . - esp_nn_conv_s16_mult4_1x1_esp32s3
