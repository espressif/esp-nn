/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-P4 optimized HardSwish with:
 * 1. Branch hoisting (borrowed from S3): dispatch on reluish_mult_exp ONCE
 * 2. 2x loop unrolling for better ILP on RISC-V pipeline
 * 3. All int16 arithmetic - no 64-bit multiply bottleneck
 */

#include <stdint.h>

static inline __attribute__((always_inline))
int16_t sat_rnd_dbl_hi_mul(int16_t a, int16_t b) {
    if (__builtin_expect(a == b && a == -32768, 0)) return 32767;
    return (int16_t)(((int32_t)a * (int32_t)b + (1 << 14)) >> 15);
}

static inline __attribute__((always_inline))
int16_t sat_dbl_hi_mul(int16_t a, int16_t b) {
    if (__builtin_expect(a == b && a == -32768, 0)) return 32767;
    return (int16_t)(((int32_t)a * (int32_t)b) >> 15);
}

static inline __attribute__((always_inline))
int16_t sat_left_shift_s16(int32_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

static inline __attribute__((always_inline))
int16_t rounding_div_pot_s16(int16_t val, int exp) {
    int32_t mask = (1 << exp) - 1;
    int32_t remainder = val & mask;
    int32_t threshold = (mask >> 1) + (val < 0 ? 1 : 0);
    return (int16_t)((val >> exp) + (remainder > threshold ? 1 : 0));
}

/* Core output computation shared by all paths */
static inline __attribute__((always_inline))
int8_t hard_swish_output(int16_t reluish, int16_t in_on_out_scale,
                          int neg_out_exp, int16_t output_zero_point) {
    int16_t pre = sat_dbl_hi_mul(reluish, in_on_out_scale);
    int16_t ov = rounding_div_pot_s16(pre, neg_out_exp);
    int32_t result = ov + output_zero_point;
    if (result > 127) result = 127;
    if (result < -128) result = -128;
    return (int8_t)result;
}

void esp_nn_hard_swish_s8_esp32p4(const int8_t *input,
                                   int8_t *output,
                                   const int32_t size,
                                   const int16_t input_zero_point,
                                   const int16_t output_mult_fxp,
                                   const int16_t reluish_mult_fxp,
                                   const int32_t reluish_mult_exp,
                                   const int32_t output_mult_exp,
                                   const int16_t output_zero_point)
{
    const int neg_out_exp = -output_mult_exp;
    int i = 0;

    /* Branch on reluish_mult_exp ONCE - 3 specialized loops */
    if (reluish_mult_exp > 0) {
        const int ls1 = reluish_mult_exp - 1;

        for (; i <= size - 2; i += 2) {
            int16_t iv0 = input[i] - input_zero_point;
            int16_t iv1 = input[i+1] - input_zero_point;
            int16_t hi0 = iv0 * 128, hi1 = iv1 * 128;

            int16_t on0 = sat_rnd_dbl_hi_mul(hi0, output_mult_fxp);
            int16_t on1 = sat_rnd_dbl_hi_mul(hi1, output_mult_fxp);

            int16_t rv0 = sat_left_shift_s16((int32_t)hi0 << ls1);
            int16_t rv1 = sat_left_shift_s16((int32_t)hi1 << ls1);
            rv0 = sat_rnd_dbl_hi_mul(rv0, reluish_mult_fxp);
            rv1 = sat_rnd_dbl_hi_mul(rv1, reluish_mult_fxp);
            rv0 = sat_left_shift_s16((int32_t)rv0 * 2);
            rv1 = sat_left_shift_s16((int32_t)rv1 * 2);

            rv0 = (int16_t)(((int32_t)rv0 + 32768) >> 1);
            rv1 = (int16_t)(((int32_t)rv1 + 32768) >> 1);

            output[i]   = hard_swish_output(rv0, on0, neg_out_exp, output_zero_point);
            output[i+1] = hard_swish_output(rv1, on1, neg_out_exp, output_zero_point);
        }
    } else if (reluish_mult_exp < 0) {
        const int neg_relu_exp = -reluish_mult_exp;

        for (; i <= size - 2; i += 2) {
            int16_t iv0 = input[i] - input_zero_point;
            int16_t iv1 = input[i+1] - input_zero_point;
            int16_t hi0 = iv0 * 128, hi1 = iv1 * 128;

            int16_t on0 = sat_rnd_dbl_hi_mul(hi0, output_mult_fxp);
            int16_t on1 = sat_rnd_dbl_hi_mul(hi1, output_mult_fxp);

            int16_t rv0 = sat_rnd_dbl_hi_mul(hi0, reluish_mult_fxp);
            int16_t rv1 = sat_rnd_dbl_hi_mul(hi1, reluish_mult_fxp);
            rv0 = rounding_div_pot_s16(rv0, neg_relu_exp);
            rv1 = rounding_div_pot_s16(rv1, neg_relu_exp);

            rv0 = (int16_t)(((int32_t)rv0 + 32768) >> 1);
            rv1 = (int16_t)(((int32_t)rv1 + 32768) >> 1);

            output[i]   = hard_swish_output(rv0, on0, neg_out_exp, output_zero_point);
            output[i+1] = hard_swish_output(rv1, on1, neg_out_exp, output_zero_point);
        }
    } else {
        for (; i <= size - 2; i += 2) {
            int16_t iv0 = input[i] - input_zero_point;
            int16_t iv1 = input[i+1] - input_zero_point;
            int16_t hi0 = iv0 * 128, hi1 = iv1 * 128;

            int16_t on0 = sat_rnd_dbl_hi_mul(hi0, output_mult_fxp);
            int16_t on1 = sat_rnd_dbl_hi_mul(hi1, output_mult_fxp);
            int16_t rv0 = sat_rnd_dbl_hi_mul(hi0, reluish_mult_fxp);
            int16_t rv1 = sat_rnd_dbl_hi_mul(hi1, reluish_mult_fxp);

            rv0 = (int16_t)(((int32_t)rv0 + 32768) >> 1);
            rv1 = (int16_t)(((int32_t)rv1 + 32768) >> 1);

            output[i]   = hard_swish_output(rv0, on0, neg_out_exp, output_zero_point);
            output[i+1] = hard_swish_output(rv1, on1, neg_out_exp, output_zero_point);
        }
    }

    /* Scalar remainder */
    for (; i < size; i++) {
        int16_t iv = input[i] - input_zero_point;
        int16_t hi = iv * 128;
        int16_t on_out = sat_rnd_dbl_hi_mul(hi, output_mult_fxp);

        int16_t rv = hi;
        if (reluish_mult_exp > 0)
            rv = sat_left_shift_s16((int32_t)rv << (reluish_mult_exp - 1));
        rv = sat_rnd_dbl_hi_mul(rv, reluish_mult_fxp);
        if (reluish_mult_exp > 0)
            rv = sat_left_shift_s16((int32_t)rv * 2);
        if (reluish_mult_exp < 0)
            rv = rounding_div_pot_s16(rv, -reluish_mult_exp);

        rv = (int16_t)(((int32_t)rv + 32768) >> 1);
        output[i] = hard_swish_output(rv, on_out, neg_out_exp, output_zero_point);
    }
}
