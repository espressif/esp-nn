/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fast 2-wide requantization for ESP32-P4 (RISC-V).
 * Interleaves mul/mulh across 2 elements for better pipeline utilization.
 * Uses a0-a7 and t0-t6 only (no callee-saved registers needed).
 *
 * void esp_nn_requant_2x_esp32p4(
 *     int32_t x0,       // a0
 *     int32_t x1,       // a1
 *     int32_t mult0,    // a2
 *     int32_t mult1,    // a3
 *     int32_t shift0,   // a4
 *     int32_t shift1,   // a5
 *     int32_t *out      // a6: pointer to store 2 results
 * );
 */

    .text
    .align  4
    .global esp_nn_requant_2x_esp32p4
    .type   esp_nn_requant_2x_esp32p4, @function

esp_nn_requant_2x_esp32p4:
    /* Compute left_shift and apply */
    mv      t0, a0              /* x0 */
    mv      t1, a1              /* x1 */
    bgez    a4, .Lls0_pos
    mv      t6, zero            /* ls0 = 0 */
    j       .Lls0_done
.Lls0_pos:
    sll     t0, t0, a4          /* x0 <<= shift0 (positive = left shift) */
    mv      t6, a4              /* ls0 = shift0 */
.Lls0_done:
    sub     a4, t6, a4          /* rs0 = ls0 - shift0 */

    bgez    a5, .Lls1_pos
    mv      t6, zero
    j       .Lls1_done
.Lls1_pos:
    sll     t1, t1, a5
    mv      t6, a5
.Lls1_done:
    sub     a5, t6, a5          /* rs1 = ls1 - shift1 */

    /* ---- Interleaved 64-bit multiply ---- */
    /* mulh first (both elements), then mul (both elements) */
    mulh    t2, t0, a2          /* hi0 */
    mulh    t3, t1, a3          /* hi1 */
    mul     t0, t0, a2          /* lo0 */
    mul     t1, t1, a3          /* lo1 */

    /* Add nudge and combine: result = ((hi:lo) + (1<<30)) >> 31 */
    li      t4, 0x40000000      /* nudge = 1 << 30 */

    add     t5, t0, t4          /* lo0 + nudge */
    sltu    t6, t5, t0          /* carry0 */
    add     t2, t2, t6          /* hi0 += carry0 */
    srli    t5, t5, 31          /* (lo0+nudge) >> 31 */
    slli    t0, t2, 1           /* hi0 << 1 */
    or      t0, t0, t5          /* result0 */

    add     t5, t1, t4          /* lo1 + nudge */
    sltu    t6, t5, t1          /* carry1 */
    add     t3, t3, t6          /* hi1 += carry1 */
    srli    t5, t5, 31
    slli    t1, t3, 1
    or      t1, t1, t5          /* result1 */

    /* ---- Right shift with rounding ---- */
    li      t4, 1

    beqz    a4, .Lskip_rs0
    addi    t5, a4, -1
    sll     t5, t4, t5          /* round0 = 1 << (rs0-1) */
    srai    t6, t0, 31          /* -1 if negative, 0 otherwise */
    add     t5, t5, t6          /* round0 += sign */
    add     t0, t0, t5
    sra     t0, t0, a4
.Lskip_rs0:

    beqz    a5, .Lskip_rs1
    addi    t5, a5, -1
    sll     t5, t4, t5
    srai    t6, t1, 31
    add     t5, t5, t6
    add     t1, t1, t5
    sra     t1, t1, a5
.Lskip_rs1:

    /* Store results */
    sw      t0, 0(a6)
    sw      t1, 4(a6)
    ret

    .size   esp_nn_requant_2x_esp32p4, . - esp_nn_requant_2x_esp32p4
