/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>

/**
 * Optimized elementwise add for s8 on ESP32-P4.
 * Uses fast multiply-by-quantized-mult and 2x unrolling.
 */

/* Inline the core requantization to avoid function call overhead */
/* Inlined fast requant using explicit RISC-V mul/mulh to avoid
 * compiler generating 64-bit multiply helper calls */
static inline __attribute__((always_inline))
int32_t add_requant(int32_t val, int32_t mult, int32_t neg_shift)
{
    /* Use C 64-bit multiply - compiler already generates mul+mulh pair at -O2 */
    int64_t prod64 = (int64_t)val * mult + ((int64_t)1 << 30);
    int32_t result = (int32_t)(prod64 >> 31);

    if (neg_shift > 0) {
        int32_t rnd = (1 << (neg_shift - 1)) - (result < 0);
        result = (result + rnd) >> neg_shift;
    }
    return result;
}

void esp_nn_add_elementwise_s8_esp32p4(const int8_t *input1_data,
                                        const int8_t *input2_data,
                                        const int32_t input1_offset,
                                        const int32_t input2_offset,
                                        const int32_t input1_mult,
                                        const int32_t input2_mult,
                                        const int32_t input1_shift,
                                        const int32_t input2_shift,
                                        const int32_t left_shift,
                                        int8_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const int32_t size)
{
    const int32_t neg_in1_shift = -input1_shift;
    const int32_t neg_in2_shift = -input2_shift;
    const int32_t neg_out_shift = -out_shift;

    int i = 0;
    /* Process 2 at a time - C inline requant lets compiler optimize across calls */
    for (; i <= size - 2; i += 2) {
        int32_t a0 = (input1_data[i + 0] + input1_offset) << left_shift;
        int32_t b0 = (input2_data[i + 0] + input2_offset) << left_shift;

        a0 = add_requant(a0, input1_mult, neg_in1_shift);
        b0 = add_requant(b0, input2_mult, neg_in2_shift);
        int32_t out0 = add_requant(a0 + b0, out_mult, neg_out_shift) + out_offset;
        out0 = max(activation_min, min(out0, activation_max));

        int32_t a1 = (input1_data[i + 1] + input1_offset) << left_shift;
        int32_t b1 = (input2_data[i + 1] + input2_offset) << left_shift;

        a1 = add_requant(a1, input1_mult, neg_in1_shift);
        b1 = add_requant(b1, input2_mult, neg_in2_shift);
        int32_t out1 = add_requant(a1 + b1, out_mult, neg_out_shift) + out_offset;
        out1 = max(activation_min, min(out1, activation_max));

        output[i + 0] = (int8_t) out0;
        output[i + 1] = (int8_t) out1;
    }

    for (; i < size; i++) {
        int32_t tmp1 = (input1_data[i] + input1_offset) << left_shift;
        int32_t tmp2 = (input2_data[i] + input2_offset) << left_shift;

        tmp1 = add_requant(tmp1, input1_mult, neg_in1_shift);
        tmp2 = add_requant(tmp2, input2_mult, neg_in2_shift);

        int32_t out = add_requant(tmp1 + tmp2, out_mult, neg_out_shift) + out_offset;
        out = max(activation_min, min(out, activation_max));
        output[i] = (int8_t) out;
    }
}
