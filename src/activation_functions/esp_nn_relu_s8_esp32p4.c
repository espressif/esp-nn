/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

/**
 * In-place ReLU6 for s8 data using ESP32-P4 PIE SIMD.
 * Clamps each element to [0, 6].
 * Processes 16 elements per iteration via 128-bit vector ops.
 */
void esp_nn_relu6_s8_esp32p4(int8_t *data, uint16_t size)
{
    /* Enable PIE */
    asm volatile (
        "csrsi  0x7f2, 0b01        \n\t"
        "li     x29, 0b10          \n\t"
        "esp.movx.w.cfg x29        \n\t"
        ::: "x29"
    );

    int i = 0;

    if (size >= 16) {
        /* Broadcast 0 into q2 and 6 into q3 */
        const int8_t zero_val = 0;
        const int8_t six_val = 6;

        asm volatile (
            "esp.vldbc.8.ip  q2, %0, 0   \n\t"
            "esp.vldbc.8.ip  q3, %1, 0   \n\t"
            :: "r"(&zero_val), "r"(&six_val)
        );

        int count = size >> 4;
        int stride = 16;

        asm volatile (
            "mv     x30, %[ptr]             \n\t"
            "mv     x31, %[cnt]             \n\t"

            "1:                             \n\t"
            "esp.vld.128.ip   q0, x30, 0    \n\t"  /* load 16 bytes, no auto-increment */
            "esp.vmax.s8      q0, q0, q2    \n\t"  /* max(val, 0) */
            "esp.vmin.s8      q0, q0, q3    \n\t"  /* min(val, 6) */
            "esp.vst.128.xp   q0, x30, %[stride] \n\t"  /* store and advance ptr by 16 */
            "addi   x31, x31, -1            \n\t"
            "bnez   x31, 1b                 \n\t"

            :
            : [ptr] "r"(data), [cnt] "r"(count), [stride] "r"(stride)
            : "x30", "x31", "memory"
        );

        i = count << 4;
    }

    /* Handle remaining elements scalar */
    for (; i < size; i++) {
        int32_t val = data[i];
        if (val < 0) val = 0;
        if (val > 6) val = 6;
        data[i] = (int8_t) val;
    }
}
