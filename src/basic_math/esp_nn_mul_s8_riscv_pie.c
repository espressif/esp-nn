/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * s8 elementwise multiply for ESP32-P4 and ESP32-S31. It runs 16 elements
 * per step on PIE and gives the same bytes as the ANSI C kernel.
 *
 * Per element, with a and b the two inputs, M = out_mult >= 0 and
 * rs = -out_shift in 0..18:
 *
 *   x  = (a + off_a) * (b + off_b)           |x| <= 255 * 255 < 2^16
 *   x' = ((a + off_a) << 7) * ((b + off_b) << 7) = x << 14
 *
 * Both factors of x' fit in s16 and x' in s32, so esp.vmul.s32.s16xs16 gives
 * it exactly. A scalar mulh per lane (PIE has no 32x32 high multiply) gives
 *
 *   m = mulh(x', M) = floor(x * M / 2^18)    |m| < 2^29
 *
 * The rounding of SaturatingRoundingDoublingHighMul is round half up for any
 * M >= 0 (the sign-dependent nudge and truncation give the same result), and
 * the nudge 2^30 is a multiple of 2^18, so
 *
 *   SRDHM(x, M) = floor((x * M + 2^30) / 2^31) = (m + 2^12) >> 13
 *
 * The rounding divide by 2^rs (half away from zero) of that value v is
 * (v + 2^(rs - 1) - (v < 0)) >> rs. Here v < 0 can be replaced by x < 0: M >= 0,
 * so x < 0 gives v <= 0, and v = 0 gives 0 either way. Folding it in:
 *
 *   out = (m + K - (x < 0) * 2^13) >> (13 + rs)
 *   K   = 2^12 + 2^(rs - 1) * 2^13, or 2^12 when rs = 0 (no rounding term)
 *
 * m + K < 2^29 + 2^30 + 2^12 fits in s32, and 13 + rs <= 31.
 *
 * Offset and clamp: clamp(v + off, lo, hi) = clamp(v, lo - off, hi - off) + off.
 * The s32 clamp to [lo - off, hi - off] runs before the narrowing to s16, so a
 * large v (rs = 0) cannot wrap; the offset is then added on s16 lanes, and the
 * low byte of the result is the int8 cast of the reference.
 *
 * Parameters outside these ranges (left shift, rs > 18, M < 0, input offsets
 * outside -127..128, an output range that does not fit in s16) take the
 * scalar code, as do the elements left over after the 16-element steps.
 */

#include <stdbool.h>
#include <stdint.h>

#include <common_functions.h>

/* Constants of the vector step, in the order the step loads them: the step
 * walks the block with esp.vldbc.*.ip and steps back to its start at the end. */
typedef struct {
    int32_t input1_offset;  /* s16 */
    int32_t input2_offset;  /* s16 */
    int32_t scale;          /* s16, 1 << 7 */
    int32_t sign_lo;        /* s32, -(1 << 13), or 0 when rs = 0 */
    int32_t round_lo;       /* s32, K */
    int32_t sign_hi;
    int32_t round_hi;
    int32_t out_offset;     /* s16 */
} mul_s8_pie_consts_t;

typedef struct {
    mul_s8_pie_consts_t c __attribute__((aligned(16)));
    int32_t mult;
    int32_t shift;          /* 13 + rs */
    int32_t lo, hi;         /* activation range minus out_offset */
} mul_s8_pie_params_t;

/* Fill p and return true when the vector step gives the reference result. */
static bool mul_s8_pie_setup(mul_s8_pie_params_t *p,
                             const int32_t input1_offset,
                             const int32_t input2_offset,
                             const int32_t out_offset,
                             const int32_t out_mult,
                             const int32_t out_shift,
                             const int32_t activation_min,
                             const int32_t activation_max)
{
    if (out_mult < 0 || out_shift > 0 || out_shift < -18) {
        return false;
    }
    if (input1_offset < -127 || input1_offset > 128 ||
            input2_offset < -127 || input2_offset > 128) {
        return false;
    }
    /* The clamp bounds, the offset and the clamped value all stay within s16 */
    if (activation_min > activation_max ||
            activation_min < INT16_MIN || activation_max > INT16_MAX ||
            out_offset < INT16_MIN || out_offset > INT16_MAX ||
            activation_min - out_offset < INT16_MIN ||
            activation_max - out_offset > INT16_MAX) {
        return false;
    }
    const int32_t rs = -out_shift;
    const int32_t sign = rs > 0 ? -(1 << 13) : 0;
    const int32_t round = (1 << 12) + (rs > 0 ? (1 << (rs - 1)) << 13 : 0);

    p->c.input1_offset = input1_offset;
    p->c.input2_offset = input2_offset;
    p->c.scale = 1 << 7;
    p->c.sign_lo = sign;
    p->c.round_lo = round;
    p->c.sign_hi = sign;
    p->c.round_hi = round;
    p->c.out_offset = out_offset;
    p->mult = out_mult;
    p->shift = 13 + rs;
    p->lo = activation_min - out_offset;
    p->hi = activation_max - out_offset;
    return true;
}

/* One half step: 8 products from q4 (lanes 0..3) and q5 (4..7) to 8 s16
 * results in QD. QD and QB are free on entry and QB is scratch on exit. The
 * constant pointer t6 advances by 8. */
#define MUL_S8_PIE_HALF(QD, QB) \
    "esp.movi.32.a  q4, a2, 0           \n\t" \
    "esp.movi.32.a  q4, a3, 1           \n\t" \
    "esp.movi.32.a  q4, a4, 2           \n\t" \
    "esp.movi.32.a  q4, a5, 3           \n\t" \
    "esp.movi.32.a  q5, s8, 0           \n\t" \
    "esp.movi.32.a  q5, s9, 1           \n\t" \
    "esp.movi.32.a  q5, s10, 2          \n\t" \
    "esp.movi.32.a  q5, s11, 3          \n\t" \
    "esp.vcmp.lt.s32 q4, q4, q7         \n\t" \
    "esp.vcmp.lt.s32 q5, q5, q7         \n\t" \
    "mulh           a2, a2, %[mult]     \n\t" \
    "mulh           a3, a3, %[mult]     \n\t" \
    "mulh           a4, a4, %[mult]     \n\t" \
    "mulh           a5, a5, %[mult]     \n\t" \
    "mulh           s8, s8, %[mult]     \n\t" \
    "mulh           s9, s9, %[mult]     \n\t" \
    "mulh           s10, s10, %[mult]   \n\t" \
    "mulh           s11, s11, %[mult]   \n\t" \
    "esp.vldbc.32.ip q6, t6, 4          \n\t" /* sign */ \
    "esp.andq       q4, q4, q6          \n\t" \
    "esp.andq       q5, q5, q6          \n\t" \
    "esp.movi.32.q  " QD ", a2, 0       \n\t" \
    "esp.movi.32.q  " QD ", a3, 1       \n\t" \
    "esp.movi.32.q  " QD ", a4, 2       \n\t" \
    "esp.movi.32.q  " QD ", a5, 3       \n\t" \
    "esp.movi.32.q  " QB ", s8, 0       \n\t" \
    "esp.movi.32.q  " QB ", s9, 1       \n\t" \
    "esp.movi.32.q  " QB ", s10, 2      \n\t" \
    "esp.movi.32.q  " QB ", s11, 3      \n\t" \
    "esp.vadd.s32   " QD ", " QD ", q4  \n\t" \
    "esp.vadd.s32   " QB ", " QB ", q5  \n\t" \
    "esp.vldbc.32.ip q6, t6, 4          \n\t" /* K */ \
    "esp.vadd.s32   " QD ", " QD ", q6  \n\t" \
    "esp.vadd.s32   " QB ", " QB ", q6  \n\t" \
    "esp.movx.w.sar s1                  \n\t" \
    "esp.vsr.s32    " QD ", " QD "      \n\t" \
    "esp.vsr.s32    " QB ", " QB "      \n\t" \
    "esp.vsat.s32   " QD ", " QD ", a0, a1 \n\t" \
    "esp.vsat.s32   " QB ", " QB ", a0, a1 \n\t" \
    "esp.vunzip.16  " QD ", " QB "      \n\t"

/* Multiply `blocks` (>= 1) runs of 16 elements. in2 advances with in1 and out.
 * Any alignment: ESP_NN_PIE_ENABLE() turns on unaligned 128-bit access. */
static inline __attribute__((always_inline))
void mul_s8_pie_blocks(const int8_t *in1, const int8_t *in2, int8_t *out,
                       int32_t blocks, const mul_s8_pie_params_t *p)
{
    register const int8_t *r_in1 asm("x28") = in1;
    register const int8_t *r_in2 asm("x29") = in2;
    register int8_t *r_out asm("x30") = out;
    register const mul_s8_pie_consts_t *r_c asm("x31") = &p->c;
    register int32_t r_lo asm("x10") = p->lo;
    register int32_t r_hi asm("x11") = p->hi;
    register int32_t r_shift asm("x9") = p->shift;

    asm volatile (
        "esp.zero.q     q7                  \n\t"
        "1:                                 \n\t"
        "li             a2, 0               \n\t"
        "esp.movx.w.sar a2                  \n\t"
        "esp.vldext.s8.ip q0, q1, t3, 16    \n\t"
        "esp.vldext.s8.ip q2, q3, t4, 16    \n\t"
        "esp.vldbc.16.ip q6, t6, 4          \n\t" /* input1_offset */
        "esp.vadd.s16   q0, q0, q6          \n\t"
        "esp.vadd.s16   q1, q1, q6          \n\t"
        "esp.vldbc.16.ip q6, t6, 4          \n\t" /* input2_offset */
        "esp.vadd.s16   q2, q2, q6          \n\t"
        "esp.vadd.s16   q3, q3, q6          \n\t"
        "esp.vldbc.16.ip q6, t6, 4          \n\t" /* 1 << 7 */
        "esp.vmul.s16   q0, q0, q6          \n\t"
        "esp.vmul.s16   q1, q1, q6          \n\t"
        "esp.vmul.s16   q2, q2, q6          \n\t"
        "esp.vmul.s16   q3, q3, q6          \n\t"
        "esp.vmul.s32.s16xs16 q4, q5, q0, q2 \n\t"
        MUL_S8_PIE_HALF("q0", "q2")
        "li             a2, 0               \n\t"
        "esp.movx.w.sar a2                  \n\t"
        "esp.vmul.s32.s16xs16 q4, q5, q1, q3 \n\t"
        MUL_S8_PIE_HALF("q1", "q3")
        "esp.vldbc.16.ip q6, t6, -28        \n\t" /* out_offset, back to the start */
        "esp.vadd.s16   q0, q0, q6          \n\t"
        "esp.vadd.s16   q1, q1, q6          \n\t"
        "esp.vunzip.8   q0, q1              \n\t"
        "esp.vst.128.ip q0, t5, 16          \n\t"
        "addi           %[blocks], %[blocks], -1 \n\t"
        "bnez           %[blocks], 1b       \n\t"
        : [blocks] "+r"(blocks), "+r"(r_in1), "+r"(r_in2), "+r"(r_out), "+r"(r_c)
        : [mult] "r"(p->mult), "r"(r_lo), "r"(r_hi), "r"(r_shift)
        : "x12", "x13", "x14", "x15", "x24", "x25", "x26", "x27", "memory"
    );
}

/* Scalar elementwise multiply, for leftovers and unsupported parameters. */
static void mul_elementwise_s8_scalar(const int8_t *input1_data,
                                      const int8_t *input2_data,
                                      const int32_t input1_offset,
                                      const int32_t input2_offset,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t activation_min,
                                      const int32_t activation_max,
                                      const int32_t size)
{
    const int32_t left_shift = out_shift > 0 ? out_shift : 0;
    const int32_t right_shift = left_shift - out_shift;
    const int64_t nudge = (int64_t)1 << 30;

    int i = 0;
    for (; i <= size - 4; i += 4) {
        int32_t prod0 = (input1_data[i+0] + input1_offset) * (input2_data[i+0] + input2_offset);
        int32_t prod1 = (input1_data[i+1] + input1_offset) * (input2_data[i+1] + input2_offset);
        int32_t prod2 = (input1_data[i+2] + input1_offset) * (input2_data[i+2] + input2_offset);
        int32_t prod3 = (input1_data[i+3] + input1_offset) * (input2_data[i+3] + input2_offset);

        int32_t s0 = prod0 << left_shift;
        int32_t s1 = prod1 << left_shift;
        int32_t s2 = prod2 << left_shift;
        int32_t s3 = prod3 << left_shift;

        int32_t r0 = (int32_t)(((int64_t)s0 * out_mult + nudge) >> 31);
        int32_t r1 = (int32_t)(((int64_t)s1 * out_mult + nudge) >> 31);
        int32_t r2 = (int32_t)(((int64_t)s2 * out_mult + nudge) >> 31);
        int32_t r3 = (int32_t)(((int64_t)s3 * out_mult + nudge) >> 31);

        if (right_shift > 0) {
            int32_t rnd = (1 << (right_shift - 1));
            r0 = (r0 + rnd - (r0 < 0)) >> right_shift;
            r1 = (r1 + rnd - (r1 < 0)) >> right_shift;
            r2 = (r2 + rnd - (r2 < 0)) >> right_shift;
            r3 = (r3 + rnd - (r3 < 0)) >> right_shift;
        }

        r0 = max(activation_min, min(r0 + out_offset, activation_max));
        r1 = max(activation_min, min(r1 + out_offset, activation_max));
        r2 = max(activation_min, min(r2 + out_offset, activation_max));
        r3 = max(activation_min, min(r3 + out_offset, activation_max));

        output[i+0] = (int8_t) r0;
        output[i+1] = (int8_t) r1;
        output[i+2] = (int8_t) r2;
        output[i+3] = (int8_t) r3;
    }

    for (; i < size; i++) {
        int32_t prod = (input1_data[i] + input1_offset) * (input2_data[i] + input2_offset);
        int32_t out = esp_nn_requantize(prod, out_mult, out_shift);
        out = max(activation_min, min(out + out_offset, activation_max));
        output[i] = (int8_t) out;
    }
}

void esp_nn_mul_elementwise_s8_riscv_pie(const int8_t *input1_data,
                                        const int8_t *input2_data,
                                        const int32_t input1_offset,
                                        const int32_t input2_offset,
                                        int8_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const int32_t size)
{
    mul_s8_pie_params_t p;
    const int32_t blocks = size / 16;
    int32_t done = 0;

    if (blocks > 0 && mul_s8_pie_setup(&p, input1_offset, input2_offset, out_offset,
                                       out_mult, out_shift, activation_min, activation_max)) {
        ESP_NN_PIE_ENABLE();
        mul_s8_pie_blocks(input1_data, input2_data, output, blocks, &p);
        done = blocks * 16;
    }
    if (done < size) {
        mul_elementwise_s8_scalar(input1_data + done, input2_data + done,
                                  input1_offset, input2_offset, output + done,
                                  out_offset, out_mult, out_shift,
                                  activation_min, activation_max, size - done);
    }
}
