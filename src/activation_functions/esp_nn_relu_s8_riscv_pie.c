/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * In-place ReLU6 for s8 data using ESP32-P4 PIE SIMD.
 * Clamps each element to [0, 6].
 * Processes 16 elements per iteration via 128-bit vector ops.
 */
void esp_nn_relu6_s8_riscv_pie(int8_t *data, uint16_t size)
{
    ESP_NN_PIE_ENABLE();

    int i = 0;

    /* esp.vld.128/esp.vst.128 need a 16-byte aligned address (verified on
     * ESP32-S31: unaligned pointers get processed lane-shifted, see
     * github.com/espressif/esp-nn issue #21 for the ESP32-S3 counterpart).
     * Consume bytes scalar until `data` is aligned. */
    int head = (16 - ((uintptr_t)data & 15)) & 15;
    if (head > size) {
        head = size;
    }
    for (; i < head; i++) {
        int32_t val = data[i];
        if (val < 0) val = 0;
        if (val > 6) val = 6;
        data[i] = (int8_t) val;
    }

    if (size - i >= 16) {
        /* Broadcast 0 into q2 and 6 into q3 */
        const int8_t zero_val = 0;
        const int8_t six_val = 6;

        /* esp.* GPR operands must be x26-x31 (required on S31) */
        asm volatile (
            "mv     x30, %0              \n\t"
            "mv     x31, %1              \n\t"
            "esp.vldbc.8.ip  q2, x30, 0  \n\t"
            "esp.vldbc.8.ip  q3, x31, 0  \n\t"
            :: "r"(&zero_val), "r"(&six_val)
            : "x30", "x31"
        );

        int count = (size - i) >> 4;
        int stride = 16;

        asm volatile (
            "mv     x30, %[ptr]             \n\t"
            /* esp.vst.128.xp stride register must be x26-x31 (required on S31) */
            "mv     x29, %[stride]          \n\t"

            /* zero-overhead hardware loop; end label sits ON the last insn */
            "esp.lp.setup     0, %[cnt], 1f \n\t"
            "esp.vld.128.ip   q0, x30, 0    \n\t"  /* load 16 bytes, no auto-increment */
            "esp.vmax.s8      q0, q0, q2    \n\t"  /* max(val, 0) */
            "esp.vmin.s8      q0, q0, q3    \n\t"  /* min(val, 6) */
            "1:                             \n\t"
            "esp.vst.128.xp   q0, x30, x29  \n\t"  /* store and advance ptr by 16 */

            :
            : [ptr] "r"(data + i), [cnt] "r"(count), [stride] "r"(stride)
            : "x29", "x30", "memory"
        );

        i += count << 4;
    }

    /* Handle remaining elements scalar */
    for (; i < size; i++) {
        int32_t val = data[i];
        if (val < 0) val = 0;
        if (val > 6) val = 6;
        data[i] = (int8_t) val;
    }
}
