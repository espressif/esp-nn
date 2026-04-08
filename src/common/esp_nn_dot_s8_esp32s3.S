//
// SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
//
// SPDX-License-Identifier: Apache-2.0
//

//
// Reusable s8 dot product kernels for ESP32-S3.
// Used by conv im2col, FC, and any kernel that reduces to a dot product.
//
// esp_nn_dot_s8_aligned_esp32s3:
//   Both input and filter 16-byte aligned. Uses ee.vld.128.ip + fused MAC.
//
// esp_nn_dot_s8_unaligned_esp32s3:
//   Input aligned, filter may be unaligned. Uses USAR+QUP for filter.
//

    .text
    .align  4

// ============================================================
// esp_nn_dot_s8_aligned_esp32s3
// Both pointers must be 16-byte aligned.
// a2: input_data (aligned)
// a3: filter_data (aligned)
// a4: len (must be multiple of 16, >= 16)
// Returns: int32_t dot product in a2
// ============================================================
    .type   esp_nn_dot_s8_aligned_esp32s3, @function
    .align  4
    .global esp_nn_dot_s8_aligned_esp32s3

esp_nn_dot_s8_aligned_esp32s3:
    entry   a1, 32

    ee.zero.accx
    beqz    a4, .Lalign_done

    // Compute loop count and remainder
    srli    a5, a4, 4               // a5 = len / 16
    beqz    a5, .Lalign_done

    // Prime: load first pair
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a3, 16
    addi            a5, a5, -1
    beqz            a5, .Lalign_last

    // Main loop: fused MAC + load
    loopgtz a5, .Lalign_loop_end
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    ee.vld.128.ip   q1, a3, 16
.Lalign_loop_end:

.Lalign_last:
    // Final MAC
    ee.vmulas.s8.accx  q0, q1

.Lalign_done:
    // 2-cycle gap before ACCX read
    movi.n  a3, 0
    nop
    ee.srs.accx a2, a3, 0

    retw.n

    .size   esp_nn_dot_s8_aligned_esp32s3, . - esp_nn_dot_s8_aligned_esp32s3


// ============================================================
// esp_nn_dot_s8_unaligned_esp32s3
// Input must be 16-byte aligned. Filter can be unaligned.
// Uses USAR+QUP pattern for filter loads.
// a2: input_data (aligned)
// a3: filter_data (may be unaligned)
// a4: len_div16 (>= 1)
// Returns: int32_t dot product in a2
// ============================================================
    .type   esp_nn_dot_s8_unaligned_esp32s3, @function
    .align  4
    .global esp_nn_dot_s8_unaligned_esp32s3

esp_nn_dot_s8_unaligned_esp32s3:
    entry   a1, 32

    ee.zero.accx
    beqz    a4, .Lunalign_done

    // Prime: first unaligned filter load (sets SAR_BYTE)
    ee.ld.128.usar.ip   q0, a3, 16

    // Check if we can do 2x unrolled (need >= 2 iterations)
    srai    a5, a4, 1               // a5 = len_div16 / 2
    beqz    a5, .Lunalign_single

    // Load first input + filter pair for unrolled loop
    ee.vld.128.ip       q1, a2, 16
    ee.ld.128.usar.ip   q2, a3, 16

    // 2x unrolled main loop
    loopgtz a5, .Lunalign_loop2_end

    ee.src.q.qup        q4, q0, q2         // align filter[i]
    ee.vld.128.ip       q3, a2, 16         // input[i+1]
    ee.vmulas.s8.accx   q4, q1             // MAC filter[i] * input[i]
    ee.ld.128.usar.ip   q0, a3, 16         // filter chunk[i+2]
    ee.src.q.qup        q5, q2, q0         // align filter[i+1]
    ee.vld.128.ip       q1, a2, 16         // input[i+2] (primed for next)
    ee.vmulas.s8.accx   q5, q3             // MAC filter[i+1] * input[i+1]
    ee.ld.128.usar.ip   q2, a3, 16         // filter chunk[i+3]

.Lunalign_loop2_end:

    // Check if there's a remaining single iteration
    bbci    a4, 0, .Lunalign_last_qup

.Lunalign_single:
    // Single iteration
    ee.vld.128.ip       q1, a2, 16
    ee.ld.128.usar.ip   q2, a3, 16
    ee.src.q.qup        q4, q0, q2
    ee.vmulas.s8.accx   q4, q1
    j                   .Lunalign_done_mac

.Lunalign_last_qup:
    // After 2x loop: back up pointers (loaded one extra pair)
    addi    a2, a2, -16
    addi    a3, a3, -16

.Lunalign_done_mac:
.Lunalign_done:
    // 2-cycle gap before ACCX read
    movi.n  a3, 0
    nop
    ee.srs.accx a2, a3, 0

    retw.n

    .size   esp_nn_dot_s8_unaligned_esp32s3, . - esp_nn_dot_s8_unaligned_esp32s3
