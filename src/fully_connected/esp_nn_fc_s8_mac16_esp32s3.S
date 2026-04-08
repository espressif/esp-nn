//
// SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
//
// SPDX-License-Identifier: Apache-2.0
//

//
// s8 dot product for FC with 2x loop unrolling and QUP for unaligned filter.
// Pattern adapted from esp-dsp dspi_dotprod_s8_aes3.S.
// Input must be 16-byte aligned. Filter can be unaligned.
//

    .text
    .align  4

    .type   esp_nn_fc_s8_mac16_esp32s3, @function
    .align  4
    .global esp_nn_fc_s8_mac16_esp32s3

// a2: input_data (16-byte aligned)
// a3: filter_data (may be unaligned)
// a4: row_len_div16 (>= 1)
// Returns: int32_t dot product in a2

esp_nn_fc_s8_mac16_esp32s3:
    entry   a1, 32

    ee.zero.accx
    beqz    a4, .Ldone

    // Prime: first unaligned filter load (sets SAR_BYTE)
    ee.ld.128.usar.ip   q0, a3, 16         // filter chunk 0

    // Check if we can do 2x unrolled (need >= 2 iterations)
    srai    a5, a4, 1                       // a5 = row_len_div16 / 2
    beqz    a5, .Lsingle

    // Load first input + filter pair for unrolled loop
    ee.vld.128.ip       q1, a2, 16         // input[0]
    ee.ld.128.usar.ip   q2, a3, 16         // filter chunk 1

    // 2x unrolled main loop: 2 MACs per iteration
    loopgtz a5, .Lloop2_end

    ee.src.q.qup        q4, q0, q2         // align filter[i]
    ee.vld.128.ip       q3, a2, 16         // input[i+1]
    ee.vmulas.s8.accx   q4, q1             // MAC filter[i] * input[i]
    ee.ld.128.usar.ip   q0, a3, 16         // filter chunk[i+2]
    ee.src.q.qup        q5, q2, q0         // align filter[i+1]
    ee.vld.128.ip       q1, a2, 16         // input[i+2] (primed for next)
    ee.vmulas.s8.accx   q5, q3             // MAC filter[i+1] * input[i+1]
    ee.ld.128.usar.ip   q2, a3, 16         // filter chunk[i+3]

.Lloop2_end:

    // Check if there's a remaining single iteration
    bbci    a4, 0, .Llast_qup              // if row_len_div16 is even, skip single

.Lsingle:
    // Single iteration: load input, QUP filter, MAC
    ee.vld.128.ip       q1, a2, 16         // input
    ee.ld.128.usar.ip   q2, a3, 16         // next filter chunk
    ee.src.q.qup        q4, q0, q2         // align filter
    ee.vmulas.s8.accx   q4, q1             // MAC
    j                   .Ldone_mac

.Llast_qup:
    // After 2x loop: need to back up pointers since we loaded one extra pair
    addi    a2, a2, -16
    addi    a3, a3, -16

.Ldone_mac:
.Ldone:
    // 2-cycle gap before ACCX read
    movi.n  a3, 0
    nop
    ee.srs.accx a2, a3, 0

    retw.n

    .size   esp_nn_fc_s8_mac16_esp32s3, . - esp_nn_fc_s8_mac16_esp32s3
