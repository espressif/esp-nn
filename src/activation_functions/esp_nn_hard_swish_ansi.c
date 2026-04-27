/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * HardSwish activation function: y = x * relu6(x + 3) / 6
 * Quantized int8 implementation using fixed-point arithmetic.
 */

#include <stdint.h>
#include <common_functions.h>

/*
 * Saturating left shift for int16
 */
static inline int16_t sat_left_shift_s16(int16_t val, int shift)
{
    int32_t result = (int32_t)val << shift;
    if (result > 32767) return 32767;
    if (result < -32768) return -32768;
    return (int16_t)result;
}

/*
 * SaturatingRoundingDoublingHighMul for int16: (a * b + (1<<14)) >> 15
 */
static inline int16_t sat_round_dbl_high_mul_s16(int16_t a, int16_t b)
{
    if (a == b && a == -32768) return 32767;
    int32_t ab = (int32_t)a * (int32_t)b;
    return (int16_t)((ab + (1 << 14)) >> 15);
}

/*
 * SaturatingDoublingHighMul (NOT rounding): (a * b) >> 15
 */
static inline int16_t sat_dbl_high_mul_s16(int16_t a, int16_t b)
{
    if (a == b && a == -32768) return 32767;
    return (int16_t)(((int32_t)a * (int32_t)b) / (1 << 15));
}

/*
 * RoundingDivideByPOT for int16
 */
static inline int16_t rounding_div_pot_s16(int16_t val, int exponent)
{
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = val & mask;
    int32_t threshold = (mask >> 1) + (val < 0 ? 1 : 0);
    return (int16_t)((val >> exponent) + (remainder > threshold ? 1 : 0));
}

void esp_nn_hard_swish_s8_ansi(const int8_t *input,
                                int8_t *output,
                                const int32_t size,
                                const int16_t input_zero_point,
                                const int16_t output_mult_fxp,
                                const int16_t reluish_mult_fxp,
                                const int32_t reluish_mult_exp,
                                const int32_t output_mult_exp,
                                const int16_t output_zero_point)
{
    for (int i = 0; i < size; i++) {
        const int16_t in_val = input[i] - input_zero_point;
        const int16_t in_hires = in_val * 128; /* << 7 */

        /* Scale input to output scale */
        const int16_t in_on_out_scale = sat_round_dbl_high_mul_s16(in_hires, output_mult_fxp);

        /* Compute reluish value: maps input from [-3,3] to [-1,1] */
        int16_t reluish = in_hires;
        if (reluish_mult_exp > 0) {
            reluish = sat_left_shift_s16(reluish, reluish_mult_exp - 1);
        }
        reluish = sat_round_dbl_high_mul_s16(reluish, reluish_mult_fxp);
        if (reluish_mult_exp > 0) {
            reluish = sat_left_shift_s16(reluish, 1);
        }
        if (reluish_mult_exp < 0) {
            reluish = rounding_div_pot_s16(reluish, -reluish_mult_exp);
        }

        /* Convert from [-1,1] to [0,1] */
        reluish = (reluish + (1 << 15)) >> 1;

        /* Multiply: output = reluish * input_on_output_scale */
        const int16_t pre_out = sat_dbl_high_mul_s16(reluish, in_on_out_scale);

        /* Final shift and offset */
        int16_t out_val = rounding_div_pot_s16(pre_out, -output_mult_exp);
        out_val += output_zero_point;
        if (out_val > 127) out_val = 127;
        if (out_val < -128) out_val = -128;
        output[i] = (int8_t)out_val;
    }
}
