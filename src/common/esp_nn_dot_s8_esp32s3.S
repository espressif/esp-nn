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
//   First operand 16-byte aligned, second arbitrary. USAR+QUP with bounded
//   loads: no load touches a 16-byte block holding no live bytes.
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
    // Read lower 32 bits of ACCX (sufficient for int8 dot products)
    nop
    nop
    rur.accx_0 a2

    retw.n

    .size   esp_nn_dot_s8_aligned_esp32s3, . - esp_nn_dot_s8_aligned_esp32s3


// ============================================================
// esp_nn_dot_s8_unaligned_esp32s3
// Fast USAR+QUP path when the first operand is 16-byte aligned (every
// in-tree caller passes the aligned operand first); bounded scalar
// fallback otherwise. Loads are bounded either way: no load ever touches
// a 16-byte block that holds no live bytes of either operand, so callers
// need no readable padding before or after either object.
// a2: input_data (any alignment; 16-byte aligned for the fast path)
// a3: filter_data (any alignment)
// a4: len_div16 (>= 1)
// Returns: int32_t dot product in a2
// ============================================================
    .type   esp_nn_dot_s8_unaligned_esp32s3, @function
    .align  4
    .global esp_nn_dot_s8_unaligned_esp32s3

esp_nn_dot_s8_unaligned_esp32s3:
    entry   a1, 32

    ee.zero.accx
    extui   a7, a2, 0, 4            // first-operand misalignment
    extui   a6, a3, 0, 4            // second-operand misalignment (0..15)
    bnez    a7, .Lunalign_scalar
    beqz    a4, .Lunalign_done

    // Prime: aligned load of the block holding b[0..15]; sets SAR_BYTE
    ee.ld.128.usar.ip   q0, a3, 16
    addi    a5, a4, -1
    beqz    a5, .Lunalign_last

    // Chunks 0 .. N-2. The lookahead block loaded in iteration i is
    // block i+1 <= N-1, which always holds live bytes of b.
    loopgtz a5, .Lunalign_loop_end
    ee.vld.128.ip       q1, a2, 16
    ee.ld.128.usar.ip   q2, a3, 16
    ee.src.q.qup        q4, q0, q2
    ee.vmulas.s8.accx   q4, q1
.Lunalign_loop_end:

.Lunalign_last:
    // Final chunk N-1. A misaligned b spills into one more block that
    // still holds live bytes; an aligned b already has the whole chunk
    // in q0 and SAR_BYTE=0 makes qup pass q0 through untouched.
    ee.vld.128.ip       q1, a2, 16
    beqz    a6, .Lunalign_pass
    ee.ld.128.usar.ip   q2, a3, 16
.Lunalign_pass:
    ee.src.q.qup        q4, q0, q2
    ee.vmulas.s8.accx   q4, q1

.Lunalign_done:
    // 2-cycle gap before ACCX read
    movi.n  a3, 0
    nop
    ee.srs.accx a2, a3, 0

    retw.n

.Lunalign_scalar:
    // Unaligned first operand: bounded scalar loop
    slli    a4, a4, 4
    movi.n  a7, 0
    beqz    a4, .Lunalign_scalar_done
    loopgtz a4, .Lunalign_scalar_loop_end
        l8ui    a5, a2, 0
        l8ui    a6, a3, 0
        sext    a5, a5, 7
        sext    a6, a6, 7
        mull    a5, a5, a6
        add     a7, a7, a5
        addi.n  a2, a2, 1
        addi.n  a3, a3, 1
.Lunalign_scalar_loop_end:
.Lunalign_scalar_done:
    mov.n   a2, a7
    retw.n

    .size   esp_nn_dot_s8_unaligned_esp32s3, . - esp_nn_dot_s8_unaligned_esp32s3
