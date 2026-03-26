/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized HardSwish with hoisted branch conditions.
 * The reluish_mult_exp is constant per-layer, so we dispatch to
 * specialized inner loops that avoid per-element branching.
 */

#include <stdint.h>

/* Inlined fixed-point math for tight inner loop */
static inline int16_t sat_rnd_dbl_hi_mul(int16_t a, int16_t b) {
    if (a == b && a == -32768) return 32767;
    return (int16_t)(((int32_t)a * (int32_t)b + (1 << 14)) >> 15);
}

static inline int16_t sat_dbl_hi_mul(int16_t a, int16_t b) {
    if (a == b && a == -32768) return 32767;
    return (int16_t)(((int32_t)a * (int32_t)b) >> 15);
}

void esp_nn_hard_swish_s8_esp32s3(const int8_t *input,
                                   int8_t *output,
                                   const int32_t size,
                                   const int16_t input_zero_point,
                                   const int16_t output_mult_fxp,
                                   const int16_t reluish_mult_fxp,
                                   const int32_t reluish_mult_exp,
                                   const int32_t output_mult_exp,
                                   const int16_t output_zero_point)
{
    /* Branch on reluish_mult_exp ONCE, not per-element */
    if (reluish_mult_exp > 0) {
        /* Left-shift path (common for MobileNetV3) */
        const int left_shift_1 = reluish_mult_exp - 1;
        const int neg_out_exp = -output_mult_exp;

        for (int i = 0; i < size; i++) {
            const int16_t iv = input[i] - input_zero_point;
            const int16_t hi = iv * 128;

            const int16_t on_out = sat_rnd_dbl_hi_mul(hi, output_mult_fxp);

            /* Reluish with left shift */
            int32_t tmp = (int32_t)hi << left_shift_1;
            int16_t rv = (int16_t)(tmp > 32767 ? 32767 : (tmp < -32768 ? -32768 : tmp));
            rv = sat_rnd_dbl_hi_mul(rv, reluish_mult_fxp);
            tmp = (int32_t)rv * 2;
            rv = (int16_t)(tmp > 32767 ? 32767 : (tmp < -32768 ? -32768 : tmp));

            /* Convert [-1,1] to [0,1] */
            rv = (int16_t)(((int32_t)rv + 32768) >> 1);

            /* Output = reluish * input_on_output */
            int16_t pre = sat_dbl_hi_mul(rv, on_out);

            /* Final shift */
            int32_t mask = (1 << neg_out_exp) - 1;
            int32_t remainder = pre & mask;
            int32_t threshold = (mask >> 1) + (pre < 0 ? 1 : 0);
            int16_t ov = (int16_t)((pre >> neg_out_exp) + (remainder > threshold ? 1 : 0));

            ov += output_zero_point;
            if (ov > 127) ov = 127;
            if (ov < -128) ov = -128;
            output[i] = (int8_t)ov;
        }
    } else if (reluish_mult_exp < 0) {
        /* Right-shift path */
        const int neg_relu_exp = -reluish_mult_exp;
        const int neg_out_exp = -output_mult_exp;

        for (int i = 0; i < size; i++) {
            const int16_t iv = input[i] - input_zero_point;
            const int16_t hi = iv * 128;

            const int16_t on_out = sat_rnd_dbl_hi_mul(hi, output_mult_fxp);

            int16_t rv = sat_rnd_dbl_hi_mul(hi, reluish_mult_fxp);
            /* Right shift */
            {
                int32_t mask = (1 << neg_relu_exp) - 1;
                int32_t remainder = rv & mask;
                int32_t threshold = (mask >> 1) + (rv < 0 ? 1 : 0);
                rv = (int16_t)((rv >> neg_relu_exp) + (remainder > threshold ? 1 : 0));
            }

            rv = (int16_t)(((int32_t)rv + 32768) >> 1);
            int16_t pre = sat_dbl_hi_mul(rv, on_out);

            int32_t mask = (1 << neg_out_exp) - 1;
            int32_t remainder = pre & mask;
            int32_t threshold = (mask >> 1) + (pre < 0 ? 1 : 0);
            int16_t ov = (int16_t)((pre >> neg_out_exp) + (remainder > threshold ? 1 : 0));

            ov += output_zero_point;
            if (ov > 127) ov = 127;
            if (ov < -128) ov = -128;
            output[i] = (int8_t)ov;
        }
    } else {
        /* No shift path (reluish_mult_exp == 0) */
        const int neg_out_exp = -output_mult_exp;

        for (int i = 0; i < size; i++) {
            const int16_t iv = input[i] - input_zero_point;
            const int16_t hi = iv * 128;

            const int16_t on_out = sat_rnd_dbl_hi_mul(hi, output_mult_fxp);
            int16_t rv = sat_rnd_dbl_hi_mul(hi, reluish_mult_fxp);

            rv = (int16_t)(((int32_t)rv + 32768) >> 1);
            int16_t pre = sat_dbl_hi_mul(rv, on_out);

            int32_t mask = (1 << neg_out_exp) - 1;
            int32_t remainder = pre & mask;
            int32_t threshold = (mask >> 1) + (pre < 0 ? 1 : 0);
            int16_t ov = (int16_t)((pre >> neg_out_exp) + (remainder > threshold ? 1 : 0));

            ov += output_zero_point;
            if (ov > 127) ov = 127;
            if (ov < -128) ov = -128;
            output[i] = (int8_t)ov;
        }
    }
}
