/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <common_functions.h>
#include "soc/soc.h"
#include "esp_nn_ansi_headers.h"
#include "esp_nn_riscv_pie.h"

/*
 * Requantization with one high multiply per element.
 *
 * The product p = (a + off_a) * (b + off_b) fits in 17 bits, so 4 * p * mult
 * fits in 64 bits and t = mulh(4 * p, mult) = floor(p * mult / 2^30).
 * TFLite's doubling high-mul with nudge is then r = (t + 1) >> 1.
 *
 * The rounding shift by rs >= 1 is (r + 2^(rs-1) - (r < 0)) >> rs. Nested
 * floors merge, so the whole step plus the output offset becomes
 *     out = (t + k - 2 * (r < 0)) >> (rs + 1)
 *     k   = 1 + 2^rs + (out_offset << (rs + 1))
 * r is non-zero only with the sign of p, and for r == 0 the sign term does
 * not change the result, so (r < 0) can be taken from p. For rs == 0 there
 * is no sign term and k = 1 + 2 * out_offset.
 *
 * k fits in 32 bits for rs <= MUL_FAST_MAX_RS. Larger shifts and positive
 * shifts use the reference code. Real models use shift in [-15, -1].
 */
#define MUL_FAST_MAX_RS 20

typedef struct {
    int32_t mult;
    int32_t k;
    int32_t sign_mask;
    int32_t sh;
    int32_t act_min;
    int32_t act_max;
} mul_rq_t;

static inline void mul_rq_init(mul_rq_t *q, int32_t out_offset, int32_t out_mult,
                               int32_t out_shift, int32_t act_min, int32_t act_max)
{
    const int32_t rs = -out_shift;
    q->mult = out_mult;
    q->k = 1 + (rs ? (1 << rs) : 0) + (int32_t)((uint32_t)out_offset << (rs + 1));
    q->sign_mask = rs ? -2 : 0;
    q->sh = rs + 1;
    q->act_min = act_min;
    q->act_max = act_max;
}

/* p4 is the product shifted left by 2. */
__NN_FORCE_INLINE__ int32_t mul_rq(int32_t p4, const mul_rq_t *q)
{
    int32_t t = (int32_t)(((int64_t)p4 * q->mult) >> 32);
    t = (t + q->k + ((p4 >> 31) & q->sign_mask)) >> q->sh;
    t = max(t, q->act_min);
    t = min(t, q->act_max);
    return t;
}

/* The fast path also needs:
 * - input offsets in [-127, 128], so |a + off| <= 255 and p fits in 17 bits;
 * - out_offset in [-128, 127], so k fits in 32 bits;
 * - out_mult >= 0, for mulhsu;
 * - activation limits in [-128, 127], since the vector clamp is s16 for
 *   rs >= 1.
 * TFLite's parameters always meet these. Others take the reference code. */
static inline int mul_fast_ok(int32_t off1, int32_t off2, int32_t out_offset,
                              int32_t out_mult, int32_t out_shift,
                              int32_t act_min, int32_t act_max)
{
    return out_shift <= 0 && out_shift >= -MUL_FAST_MAX_RS && out_mult >= 0 &&
           (uint32_t)(off1 + 127) <= 255 && (uint32_t)(off2 + 127) <= 255 &&
           (uint32_t)(out_offset + 128) <= 255 &&
           act_min >= -128 && act_max <= 127;
}

/*
 * PIE part: 16 elements per step. The vector unit loads both inputs at any
 * alignment, since the CFG unaligned bit is set. It adds the offsets in s16
 * and forms the 32-bit products P = 2 * p. PIE has no 32x32 high multiply, so only the
 * high multiply per lane is scalar. The lane is moved out, multiplied and
 * moved back. mulhsu(2 * p, 2 * mult) equals mulh(4 * p, mult), and 2 * mult
 * still fits in 32 unsigned bits. This saves one doubling of b.
 * The vector unit then adds k and the sign term, shifts by rs + 1, clamps,
 * narrows to s8 and stores 16 bytes.
 *
 * The sign term is -2 where t < 0. t has the sign of p wherever the term
 * matters. It is found as (t + k) < k with vcmp, which needs no SAR change.
 * |t| < 2^17 and |k| < 2^29, so the sum does not overflow.
 * The sign term only applies for rs >= 1. For rs == 0 the step skips it.
 *
 * For rs >= 1 the shifted value fits in s16: |t| < 2^17 and the shift is
 * at least 2. So the clamp runs on 8 s16 lanes after the first narrowing
 * instead of on 4 s32 lanes. For rs == 0 it can exceed s16 and the clamp
 * stays on s32.
 */

/*
 * Products of a[0..15] and b[0..15], 4 lanes each, into q1, q2, q4 and q3
 * in that order. Needs off1 in q6 and 2 * off2 in q7, as s16 lanes.
 * Sets SAR to 0. q5 is not touched, so k stays loaded across steps.
 * Each vmul writes one of its own sources. This works on P4 and frees the
 * register that keeps k.
 */
#define MUL_PIE_LOAD_AB_ASM                                      \
        "mv                    x28, %[a]          \n\t"          \
        "esp.vld.128.ip        q0, x28, 0         \n\t"          \
        "mv                    x28, %[b]          \n\t"          \
        "esp.vld.128.ip        q1, x28, 0         \n\t"
#define MUL_PIE_PRODUCTS_ASM                                     \
        "esp.vext.s8           q2, q3, q0         \n\t"          \
        "esp.vadd.s16          q2, q2, q6         \n\t"          \
        "esp.vadd.s16          q3, q3, q6         \n\t"          \
        "esp.vext.s8           q4, q0, q1         \n\t"          \
        "esp.vadd.s16          q4, q4, q4         \n\t"          \
        "esp.vadd.s16          q0, q0, q0         \n\t"          \
        "esp.vadd.s16          q4, q4, q7         \n\t"          \
        "esp.vadd.s16          q0, q0, q7         \n\t"          \
        "esp.vmul.s32.s16xs16  q1, q2, q2, q4     \n\t"          \
        "esp.vmul.s32.s16xs16  q4, q3, q3, q0     \n\t"

#define MUL_PIE_PRODUCTS(pa, pb)                                 \
    asm volatile (MUL_PIE_LOAD_AB_ASM                            \
        "li                    x28, 0             \n\t"          \
        "esp.movx.w.sar        x28                \n\t"          \
        MUL_PIE_PRODUCTS_ASM                                     \
        :: [a] "r"(pa), [b] "r"(pb)                              \
        : "x28", "memory")

/* Sign term for rs >= 1: qr = t + k on entry, qr += 2 * ((t + k) < k). */
#define MUL_PIE_SGN_ON(qr)                                       \
        "esp.vcmp.lt.s32       q0, " qr ", q5     \n\t"          \
        "esp.vadd.s32          " qr ", " qr ", q0 \n\t"          \
        "esp.vadd.s32          " qr ", " qr ", q0 \n\t"
#define MUL_PIE_SGN_OFF(qr) ""
/* Clamp on s32 lanes (rs == 0) or on s16 lanes after narrowing (rs >= 1). */
#define MUL_PIE_SAT32(qr)                                        \
        "esp.vsat.s32          " qr ", " qr ", x30, x31 \n\t"
#define MUL_PIE_SAT16(qr)                                        \
        "esp.vsat.s16          " qr ", " qr ", x30, x31 \n\t"
#define MUL_PIE_NONE(qr) ""

/*
 * Requantize the 4 lanes of qr in place, two lanes at a time through x28
 * and x29. Needs SAR = rs + 1, x30 = act_min, x31 = act_max and q5 = k.
 * sgn is MUL_PIE_SGN_ON or MUL_PIE_SGN_OFF. sat is MUL_PIE_SAT32 or
 * MUL_PIE_NONE.
 */
#define MUL_PIE_RQ_ASM(qr, sgn, sat)                             \
        "esp.movi.32.a         " qr ", x28, 0     \n\t"          \
        "esp.movi.32.a         " qr ", x29, 1     \n\t"          \
        "mulhsu                x28, x28, %[m]     \n\t"          \
        "mulhsu                x29, x29, %[m]     \n\t"          \
        "esp.movi.32.q         " qr ", x28, 0     \n\t"          \
        "esp.movi.32.q         " qr ", x29, 1     \n\t"          \
        "esp.movi.32.a         " qr ", x28, 2     \n\t"          \
        "esp.movi.32.a         " qr ", x29, 3     \n\t"          \
        "mulhsu                x28, x28, %[m]     \n\t"          \
        "mulhsu                x29, x29, %[m]     \n\t"          \
        "esp.movi.32.q         " qr ", x28, 2     \n\t"          \
        "esp.movi.32.q         " qr ", x29, 3     \n\t"          \
        "esp.vadd.s32          " qr ", " qr ", q5 \n\t"          \
        sgn(qr)                                                  \
        "esp.vsr.s32           " qr ", " qr "     \n\t"          \
        sat(qr)

/*
 * Prefetch: a byte load MUL_PF_DIST bytes ahead of a and of b, into a
 * register nothing reads. The core does not wait for it, so a cache miss
 * on PSRAM inputs overlaps with the step's compute. On PSRAM-sized inputs
 * this gives 8-11%. It only runs on external memory. On internal RAM it
 * would cost 2 instructions per step (1-2%) for nothing. 384 bytes
 * measured best among 256 to 512.
 */
#define MUL_PF_DIST 384
#define MUL_STR_(x) #x
#define MUL_STR(x) MUL_STR_(x)
/*
 * True for external flash and PSRAM, both behind the cache. On P4 and S31
 * they span SOC_DROM_LOW to SOC_EXTRAM_HIGH. Internal RAM lies outside that
 * range: above it on P4, below it on S31. A simpler check than
 * esp_ptr_external_ram(), which is not inline.
 */
static inline int mul_in_ext_mem(const void *p)
{
#if defined(SOC_DROM_LOW) && defined(SOC_EXTRAM_HIGH)
    return (uintptr_t) p - SOC_DROM_LOW < SOC_EXTRAM_HIGH - SOC_DROM_LOW;
#else
    return 1;
#endif
}
#define MUL_PF_ON(r, base, dist) "lb " r ", " MUL_STR(dist) "(" base ") \n\t"
#define MUL_PF_OFF(r, base, dist) ""

/*
 * n_steps full steps of 16 elements in a hardware loop. pa, pb and op
 * advance by 16 with their load or store. pf is MUL_PF_ON or MUL_PF_OFF.
 * The store is the last body instruction. The body is longer than 24
 * bytes, so ESP_NN_HWLOOP_LAST needs no pad: on rev < v3.0 builds it adds
 * only the loop end nop.
 *
 * The GPR operands of these PIE instructions take x8-x15 and x24-x31, not
 * only x26-x31. So SAR is set straight from z (= 0) and sh, and b has its
 * own pointer. This keeps the step free of register moves and adds.
 */
#define MUL_PIE_LOOP(sgn, sat32, sat16, pf, n_steps)                           \
    asm volatile (                                                              \
        ESP_NN_HWLOOP_BEGIN                                                     \
        "esp.lp.setup          0, %[cnt], 1f      \n\t"                         \
        pf("%[t0]", "%[a]", MUL_PF_DIST)                                        \
        "esp.vld.128.ip        q0, %[a], 16       \n\t"                         \
        "esp.vld.128.ip        q1, %[b], 16       \n\t"                         \
        /* pb has already advanced by 16. */                                    \
        pf("%[t1]", "%[b]", MUL_PF_DIST - 16)                                   \
        "esp.movx.w.sar        %[z]               \n\t"                         \
        MUL_PIE_PRODUCTS_ASM                                                    \
        "esp.movx.w.sar        %[sh]              \n\t"                         \
        MUL_PIE_RQ_ASM("q1", sgn, sat32)                                        \
        MUL_PIE_RQ_ASM("q2", sgn, sat32)                                        \
        MUL_PIE_RQ_ASM("q4", sgn, sat32)                                        \
        MUL_PIE_RQ_ASM("q3", sgn, sat32)                                        \
        "esp.vunzip.16         q1, q2             \n\t"                         \
        "esp.vunzip.16         q4, q3             \n\t"                         \
        sat16("q1")                                                             \
        sat16("q4")                                                             \
        "esp.vunzip.8          q1, q4             \n\t"                         \
        ESP_NN_HWLOOP_LAST("esp.vst.128.ip        q1, %[o], 16", 0)             \
        ESP_NN_HWLOOP_END                                                       \
        : [o] "+r"(op), [a] "+r"(pa), [b] "+r"(pb),                             \
          [t0] "=&r"(t0), [t1] "=&r"(t1)                                        \
        : [z] "r"(z), [cnt] "r"(n_steps),                                       \
          [m] "r"(m2), [sh] "r"(shr), "r"(lo), "r"(hi)                          \
        : "x28", "x29", "memory")

/* MUL_PIE_LOOP for any n_steps >= 1: in chunks of ESP_NN_HWLOOP_MAX_COUNT
 * steps where the hardware loop count is limited (ESP32-P4 rev < v3). */
#if ESP_NN_HWLOOP_MAX_COUNT < INT32_MAX
#define MUL_PIE_LOOP_N(sgn, sat32, sat16, pf, n_steps)                         \
    for (int32_t left = (n_steps); left > 0; ) {                              \
        const int32_t k = left < ESP_NN_HWLOOP_MAX_COUNT ?                    \
                          left : ESP_NN_HWLOOP_MAX_COUNT;                     \
        MUL_PIE_LOOP(sgn, sat32, sat16, pf, k);                               \
        left -= k;                                                            \
    }
#else
#define MUL_PIE_LOOP_N(sgn, sat32, sat16, pf, n_steps)                         \
    MUL_PIE_LOOP(sgn, sat32, sat16, pf, n_steps)
#endif

/* Requantize lane idx (0..15) of the products and store it to out[idx]. */
#define MUL_LANE(qr, sel, idx) do {                                   \
        register int32_t v_ asm("x29");                               \
        asm volatile ("esp.movi.32.a " qr ", %0, " #sel : "=r"(v_)); \
        out[idx] = (int8_t) mul_rq(v_ * 2, q);                        \
    } while (0)

/*
 * For each of rows rows, multiply n elements of a and b. a and out advance by
 * n per row, b is the same for every row (broadcast; rows is 1 for
 * elementwise). A tail of r < 16 elements, with n >= 16,
 * reuses one vector step on the last 16 elements and keeps only the last r
 * lanes. The loads stay inside the buffers. They happen before any tail
 * store, so the kept lanes are right even when out aliases a.
 */
static __attribute__((noinline)) void mul_rows(const int8_t *a, const int8_t *b, int8_t *out,
                                              int32_t rows, int32_t n,
                                              int32_t off1, int32_t off2,
                                              int32_t out_offset, int32_t out_mult,
                                              int32_t out_shift, int32_t act_min,
                                              int32_t act_max)
{
    /*
     * q is set up here, not by the callers, so their small-size path
     * stays a leaf without a stack frame.
     */
    mul_rq_t lq;
    mul_rq_init(&lq, out_offset, out_mult, out_shift, act_min, act_max);
    const mul_rq_t *q = &lq;
    const int32_t off2_x2 = off2 * 2;
    const uint32_t m2 = (uint32_t) lq.mult * 2;
    const int32_t sign = lq.sign_mask;
    const int32_t sh = lq.sh;
    register int32_t lo asm("x30") = lq.act_min;
    register int32_t hi asm("x31") = lq.act_max;
    const int16_t offs[2] = { (int16_t) off1, (int16_t) off2_x2 };

    asm volatile (
        "mv              x28, %[p]        \n\t"
        /* Offset 0 and an addi: for the rev < v3 ISA (xesppie), GCC 13.2
         * (IDF v5.3) is known to encode the esp.vldbc.16.ip offset in
         * 4-byte units, so 2 does not assemble there. The compiler cannot
         * tell toolchains apart, so every build uses this form. */
        "esp.vldbc.16.ip q6, x28, 0       \n\t"
        "addi            x28, x28, 2      \n\t"
        "esp.vldbc.16.ip q7, x28, 0       \n\t"
        "mv              x28, %[kp]       \n\t"
        "esp.vldbc.32.ip q5, x28, 0       \n\t"
        :: [p] "r"(offs), "m"(offs[0]), "m"(offs[1]), [kp] "r"(&lq.k), "m"(lq.k)
        : "x28");
    const int32_t cnt = n >> 4;
    /*
     * Prefetch steps read MUL_PF_DIST bytes ahead of a and b, so they stop
     * that far before the end of the row. They run only when input a is in
     * external memory. b is the same size as a or a small per-channel vector.
     */
    const int32_t c1 = (cnt > MUL_PF_DIST / 16 && mul_in_ext_mem(a)) ?
                       cnt - MUL_PF_DIST / 16 : 0;
    register int32_t z asm("x12") = 0;
    register int32_t shr asm("x13") = sh;
    for (; rows > 0; rows--, a += n, out += n) {
        register int8_t *op asm("x26") = out;
        register const int8_t *pa asm("x27") = a;
        register const int8_t *pb asm("x14") = b;
        int32_t t0, t1;
        if (sign) {
            if (c1) {
                MUL_PIE_LOOP_N(MUL_PIE_SGN_ON, MUL_PIE_NONE, MUL_PIE_SAT16, MUL_PF_ON, c1);
            }
            MUL_PIE_LOOP_N(MUL_PIE_SGN_ON, MUL_PIE_NONE, MUL_PIE_SAT16, MUL_PF_OFF, cnt - c1);
        } else {
            MUL_PIE_LOOP_N(MUL_PIE_SGN_OFF, MUL_PIE_SAT32, MUL_PIE_NONE, MUL_PF_OFF, cnt);
        }
        const int32_t r = n & 15;
        if (r == 0) {
            continue;
        }
        const int32_t i = n - 16;
        MUL_PIE_PRODUCTS(a + i, b + i);
        switch (r) {
        case 15: MUL_LANE("q1", 1, i + 1);  /* fall through */
        case 14: MUL_LANE("q1", 2, i + 2);  /* fall through */
        case 13: MUL_LANE("q1", 3, i + 3);  /* fall through */
        case 12: MUL_LANE("q2", 0, i + 4);  /* fall through */
        case 11: MUL_LANE("q2", 1, i + 5);  /* fall through */
        case 10: MUL_LANE("q2", 2, i + 6);  /* fall through */
        case 9:  MUL_LANE("q2", 3, i + 7);  /* fall through */
        case 8:  MUL_LANE("q4", 0, i + 8);  /* fall through */
        case 7:  MUL_LANE("q4", 1, i + 9);  /* fall through */
        case 6:  MUL_LANE("q4", 2, i + 10); /* fall through */
        case 5:  MUL_LANE("q4", 3, i + 11); /* fall through */
        case 4:  MUL_LANE("q3", 0, i + 12); /* fall through */
        case 3:  MUL_LANE("q3", 1, i + 13); /* fall through */
        case 2:  MUL_LANE("q3", 2, i + 14); /* fall through */
        default: MUL_LANE("q3", 3, i + 15);
        }
    }
    /* Other PIE kernels use SAR-dependent instructions and expect SAR = 0. */
    asm volatile ("li x28, 0\n\tesp.movx.w.sar x28" ::: "x28");
}

/* Scalar path for fewer than 16 elements. */
static void mul_row_scalar(const int8_t *a, const int8_t *b, int8_t *out,
                           int32_t n, const mul_rq_t *q, int32_t off1, int32_t off2)
{
    const int32_t off2_x4 = off2 * 4;
    for (int32_t i = 0; i < n; i++) {
        int32_t p = (a[i] + off1) * (b[i] * 4 + off2_x4);
        out[i] = (int8_t) mul_rq(p, q);
    }
}

/*
 * Same signature as the entry point, so the entry can tail-call it and its
 * small-size path stays a leaf without a stack frame.
 */
static __attribute__((noinline)) void mul_elementwise_pie(const int8_t *input1_data,
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
    ESP_NN_PIE_ENABLE();
    mul_rows(input1_data, input2_data, output, 1, size, input1_offset, input2_offset,
             out_offset, out_mult, out_shift, activation_min, activation_max);
}

/**
 * Elementwise multiply for s8, for ESP32-P4 and ESP32-S31.
 */
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
    if (!mul_fast_ok(input1_offset, input2_offset, out_offset, out_mult, out_shift,
                     activation_min, activation_max)) {
        esp_nn_mul_elementwise_s8_ansi(input1_data, input2_data, input1_offset,
                                       input2_offset, output, out_offset, out_mult,
                                       out_shift, activation_min, activation_max, size);
        return;
    }

    if (size < 16) {
        mul_rq_t q;
        mul_rq_init(&q, out_offset, out_mult, out_shift, activation_min, activation_max);
        mul_row_scalar(input1_data, input2_data, output, size, &q,
                       input1_offset, input2_offset);
        return;
    }
    mul_elementwise_pie(input1_data, input2_data, input1_offset, input2_offset, output,
                        out_offset, out_mult, out_shift, activation_min, activation_max, size);
}

/**
 * Multiply [positions, channels] by a per-channel vector, for ESP32-P4 and
 * ESP32-S31.
 */
void esp_nn_mul_broadcast_channel_s8_riscv_pie(const int8_t *input1,
                                               const int8_t *input2_per_ch,
                                               const int32_t input1_offset,
                                               const int32_t input2_offset,
                                               int8_t *output,
                                               const int32_t output_offset,
                                               const int32_t output_mult,
                                               const int32_t output_shift,
                                               const int32_t activation_min,
                                               const int32_t activation_max,
                                               const int32_t total_spatial,
                                               const int32_t channels)
{
    if (!mul_fast_ok(input1_offset, input2_offset, output_offset, output_mult,
                     output_shift, activation_min, activation_max)) {
        esp_nn_mul_broadcast_channel_s8_ansi(input1, input2_per_ch, input1_offset,
                                             input2_offset, output, output_offset,
                                             output_mult, output_shift, activation_min,
                                             activation_max, total_spatial, channels);
        return;
    }

    if (channels < 16) {
        mul_rq_t q;
        mul_rq_init(&q, output_offset, output_mult, output_shift, activation_min, activation_max);
        for (int32_t s = 0; s < total_spatial; s++) {
            mul_row_scalar(input1 + s * channels, input2_per_ch, output + s * channels,
                           channels, &q, input1_offset, input2_offset);
        }
        return;
    }
    ESP_NN_PIE_ENABLE();
    mul_rows(input1, input2_per_ch, output, total_spatial, channels, input1_offset,
             input2_offset, output_offset, output_mult, output_shift, activation_min,
             activation_max);
}
