/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <string.h>
#include <common_functions.h>
#include "soc/soc.h"
#include "esp_nn_ansi_headers.h"
#include "esp_nn_riscv_pie.h"
#include "esp_nn_vldbc_riscv_pie.h"

/*
 * Elementwise s8 add for ESP32-P4 and ESP32-S31, bit-exact with
 * esp_nn_add_elementwise_s8_ansi.
 *
 * Input stage, for v = in + offset (|v| <= 255), left shift L and input
 * multiplier m with right shift e:
 *     d = rdbpot(srdhm(v << L, m), e)
 * srdhm(x, m) is floor((x * m + 2^30) / 2^31), so with R = 30 - L
 *     t = floor(v * m / 2^R)
 *     d = (t + k - 2 * (t < 0)) >> (e + 1),  k = 1 + 2^e    (e >= 1)
 *     d = (t + 1) >> 1                                       (e == 0)
 * Nested floors merge, as in the multiply kernel. With m = mh * 2^16 + ml,
 * ml the sign-extended low 16 bits:
 *     t = ((v * mh) << (16 - R)) + ((v * ml) >> R)
 * Both products are 16 x 16 bit, so the whole input stage runs on PIE:
 * esp.vmul.s32.s16xs16 shifts its products right by SAR, which gives the
 * floor of v * ml / 2^R directly.
 *
 * Output stage, for s = d1 + d2 (|s| < 2^29), output multiplier mo with
 * right shift rs and output offset:
 *     t = mulhsu(2 * s, 2 * mo) = floor(s * mo / 2^30)
 *     out = (t + k - 2 * (t < 0)) >> (rs + 1),
 *     k = 1 + 2^rs + (out_offset << (rs + 1))                (rs >= 1)
 *     out = (t + 1 + 2 * out_offset) >> 1                    (rs == 0)
 * PIE has no 32 x 32 bit high multiply, so only this multiply is scalar,
 * one per element. The clamp runs on s32 lanes before narrowing, so any
 * activation limits give the reference result.
 *
 * A host check of these formulas against the reference covered L 14..20,
 * shifts 0..-20, every offset in range and int32 activation limits.
 *
 * The fast path needs:
 * - left_shift in [14, 20], so R in [10, 16] and |t| < 2^29;
 * - input offsets in [-127, 128], so |v| <= 255;
 * - multipliers in [0, 0x7fff8000), so mh fits in s16;
 * - shifts in [-20, 0] and out_offset in [-128, 127], so k fits in s32.
 * TFLite's parameters meet these. Others take the scalar code.
 */
#define ADD_FAST_MAX_RS 20
#define ADD_MULT_LIMIT 0x7fff8000

static inline int add_fast_ok(int32_t off1, int32_t off2, int32_t m1, int32_t m2,
                              int32_t s1, int32_t s2, int32_t left_shift,
                              int32_t out_offset, int32_t mo, int32_t so)
{
    return left_shift >= 14 && left_shift <= 20 &&
           (uint32_t)(off1 + 127) <= 255 && (uint32_t)(off2 + 127) <= 255 &&
           (uint32_t) m1 < ADD_MULT_LIMIT && (uint32_t) m2 < ADD_MULT_LIMIT &&
           (uint32_t) mo < ADD_MULT_LIMIT &&
           s1 <= 0 && s1 >= -ADD_FAST_MAX_RS && s2 <= 0 && s2 >= -ADD_FAST_MAX_RS &&
           so <= 0 && so >= -ADD_FAST_MAX_RS &&
           (uint32_t)(out_offset + 128) <= 255;
}

/*
 * Broadcast constants, in the order the step reads them. Each slot is 4
 * bytes. The step walks the table with post-increment loads and rewinds the
 * pointer once. An s16 constant is read with ADD_LD16. Under
 * ESP_NN_VLDBC_IP_BROKEN, ADD_S16 stores it in both halves of its slot and
 * ADD_LD16 is esp.vldbc.32.ip, which gives the same 8 lanes.
 */
#if ESP_NN_VLDBC_IP_BROKEN
#define ADD_LD16 "esp.vldbc.32.ip       q7, %[c], 4         \n\t"
#define ADD_S16(v) ((int32_t)((uint32_t)(uint16_t)(v) * 0x10001u))
#else
#define ADD_LD16 "esp.vldbc.16.ip       q7, %[c], 4         \n\t"
#define ADD_S16(v) (v)
#endif

typedef struct {
    int32_t off1, off2;          /* s16 */
    int32_t p1, mh1, ml1, k1, sm1;   /* p = 2^(16 - R); p, mh, ml s16; */
    int32_t p2, mh2, ml2, k2, sm2;   /* sm = -1 for e >= 1, else 0 */
    int32_t ko, smo;
} add_consts_t;
#define ADD_CONSTS_BYTES 56
_Static_assert(sizeof(add_consts_t) == ADD_CONSTS_BYTES, "constant table size");

static inline void add_in_consts(int32_t m, int32_t shift, int32_t *mh, int32_t *ml,
                                 int32_t *k, int32_t *sm)
{
    const int32_t e = -shift;
    const int32_t lo = (int16_t)(m & 0xffff);
    *ml = ADD_S16(lo);
    *mh = ADD_S16((m - lo) >> 16);
    *k = 1 + (e ? (1 << e) : 0);
    *sm = e ? -1 : 0;
}

/*
 * Prefetch: a byte load ADD_PF_DIST bytes ahead of each input, into a
 * register nothing reads, so a cache miss on external memory overlaps the
 * step's compute. Only for inputs in flash or PSRAM.
 */
#define ADD_PF_DIST 384
#define ADD_STR_(x) #x
#define ADD_STR(x) ADD_STR_(x)
static inline int add_in_ext_mem(const void *p)
{
#if defined(SOC_DROM_LOW) && defined(SOC_EXTRAM_HIGH)
    return (uintptr_t) p - SOC_DROM_LOW < SOC_EXTRAM_HIGH - SOC_DROM_LOW;
#else
    return 1;
#endif
}
#define ADD_PF_ON(r, base, dist) "lb " r ", " ADD_STR(dist) "(" base ") \n\t"
#define ADD_PF_OFF(r, base, dist) ""

/* Input stage for v (8 s16 lanes) into qa (lanes 0-3) and qb (lanes 4-7).
 * Uses qt0, qt1 and q7. Reads p, mh, ml, k, sm from %[c].
 * v * mh is shifted left by 16 - R through v: v * 2^(16 - R) fits s16
 * (|v| <= 255, 16 - R <= 6). esp.vsl.s32 would do the same on the product,
 * but the ESP32-P4 rev < v3 instruction set does not have it. */
#define ADD_IN_STAGE(v, qa, qb, qt0, qt1, se)                                 \
        "esp.movx.w.sar        %[z]                \n\t"                      \
        ADD_LD16                                                              \
        "esp.vmul.s16          " qt0 ", " v ", q7  \n\t"                      \
        ADD_LD16                                                              \
        "esp.vmul.s32.s16xs16  " qa ", " qb ", " qt0 ", q7 \n\t"              \
        ADD_LD16                                                              \
        "esp.movx.w.sar        %[slo]              \n\t"                      \
        "esp.vmul.s32.s16xs16  " qt0 ", " qt1 ", " v ", q7 \n\t"              \
        "esp.vadd.s32          " qa ", " qa ", " qt0 " \n\t"                  \
        "esp.vadd.s32          " qb ", " qb ", " qt1 " \n\t"                  \
        ADD_KSGN_SHIFT(qa, qb, qt0, qt1, se)

/* qa, qb += k; += 2 * ((qa < k) & sm); >>= SAR from se. */
#define ADD_KSGN_SHIFT(qa, qb, qt0, qt1, se)                                  \
        "esp.vldbc.32.ip       q7, %[c], 4         \n\t"                      \
        "esp.vadd.s32          " qa ", " qa ", q7  \n\t"                      \
        "esp.vadd.s32          " qb ", " qb ", q7  \n\t"                      \
        "esp.vcmp.lt.s32       " qt0 ", " qa ", q7 \n\t"                      \
        "esp.vcmp.lt.s32       " qt1 ", " qb ", q7 \n\t"                      \
        "esp.vldbc.32.ip       q7, %[c], 4         \n\t"                      \
        "esp.andq              " qt0 ", " qt0 ", q7 \n\t"                     \
        "esp.andq              " qt1 ", " qt1 ", q7 \n\t"                     \
        "esp.vadd.s32          " qa ", " qa ", " qt0 " \n\t"                  \
        "esp.vadd.s32          " qa ", " qa ", " qt0 " \n\t"                  \
        "esp.vadd.s32          " qb ", " qb ", " qt1 " \n\t"                  \
        "esp.vadd.s32          " qb ", " qb ", " qt1 " \n\t"                  \
        "esp.movx.w.sar        " se "              \n\t"                      \
        "esp.vsr.s32           " qa ", " qa "      \n\t"                      \
        "esp.vsr.s32           " qb ", " qb "      \n\t"

/* High multiply of the 4 lanes of qr by %[m], through x28 and x29. */
#define ADD_MULH4(qr)                                                         \
        "esp.movi.32.a         " qr ", x28, 0      \n\t"                      \
        "esp.movi.32.a         " qr ", x29, 1      \n\t"                      \
        "mulhsu                x28, x28, %[m]      \n\t"                      \
        "mulhsu                x29, x29, %[m]      \n\t"                      \
        "esp.movi.32.q         " qr ", x28, 0      \n\t"                      \
        "esp.movi.32.q         " qr ", x29, 1      \n\t"                      \
        "esp.movi.32.a         " qr ", x28, 2      \n\t"                      \
        "esp.movi.32.a         " qr ", x29, 3      \n\t"                      \
        "mulhsu                x28, x28, %[m]      \n\t"                      \
        "mulhsu                x29, x29, %[m]      \n\t"                      \
        "esp.movi.32.q         " qr ", x28, 2      \n\t"                      \
        "esp.movi.32.q         " qr ", x29, 3      \n\t"

/*
 * One step: 8 elements of a and b into 8 bytes of out in q0. Loads both
 * inputs before anything is stored. Ends with %[c] back at the table start.
 */
#define ADD_STEP_BODY(pf)                                                     \
        pf("%[t0]", "%[a]", ADD_PF_DIST)                                      \
        "esp.vld.l.64.ip       q0, %[a], 8         \n\t"                      \
        "esp.vld.l.64.ip       q1, %[b], 8         \n\t"                      \
        pf("%[t1]", "%[b]", ADD_PF_DIST - 8)                                  \
        "esp.vext.s8           q2, q7, q0          \n\t"                      \
        "esp.vext.s8           q3, q7, q1          \n\t"                      \
        ADD_LD16                                                              \
        "esp.vadd.s16          q2, q2, q7          \n\t"                      \
        ADD_LD16                                                              \
        "esp.vadd.s16          q3, q3, q7          \n\t"                      \
        ADD_IN_STAGE("q2", "q0", "q1", "q4", "q5", "%[se1]")                  \
        ADD_IN_STAGE("q3", "q2", "q6", "q4", "q5", "%[se2]")                  \
        "esp.vadd.s32          q0, q0, q2          \n\t"                      \
        "esp.vadd.s32          q1, q1, q6          \n\t"                      \
        "esp.vadd.s32          q0, q0, q0          \n\t"                      \
        "esp.vadd.s32          q1, q1, q1          \n\t"                      \
        ADD_MULH4("q0")                                                       \
        ADD_MULH4("q1")                                                       \
        ADD_KSGN_SHIFT("q0", "q1", "q4", "q5", "%[so]")                       \
        "addi                  %[c], %[c], -" ADD_STR(ADD_CONSTS_BYTES) " \n\t" \
        "esp.vsat.s32          q0, q0, %[lo], %[hi] \n\t"                     \
        "esp.vsat.s32          q1, q1, %[lo], %[hi] \n\t"                     \
        "esp.vunzip.16         q0, q1              \n\t"                      \
        "esp.vunzip.8          q0, q1              \n\t"

#define ADD_PIE_LOOP(pf, n_steps)                                             \
    asm volatile (                                                            \
        ESP_NN_HWLOOP_BEGIN                                                   \
        "esp.lp.setup          0, %[cnt], 1f       \n\t"                      \
        ADD_STEP_BODY(pf)                                                     \
        ESP_NN_HWLOOP_LAST("esp.vst.l.64.ip       q0, %[o], 8", 0)            \
        ESP_NN_HWLOOP_END                                                     \
        : [o] "+r"(po), [a] "+r"(pa), [b] "+r"(pb), [c] "+r"(pc),             \
          [t0] "=&r"(t0), [t1] "=&r"(t1)                                      \
        : [cnt] "r"(n_steps), [m] "r"(m2),                                    \
          [z] "r"(z), [slo] "r"(slo),                         \
          [se1] "r"(se1), [se2] "r"(se2), [so] "r"(so),                       \
          [lo] "r"(lo), [hi] "r"(hi), "m"(*cst)                               \
        : "x28", "x29", "memory")

static __attribute__((noinline)) void add_pie(const int8_t *a, const int8_t *b,
                                             int8_t *out, int32_t size,
                                             int32_t off1, int32_t off2,
                                             int32_t m1, int32_t m2_in,
                                             int32_t s1, int32_t s2, int32_t left_shift,
                                             int32_t out_offset, int32_t mo, int32_t out_shift,
                                             int32_t act_min, int32_t act_max)
{
    add_consts_t cs __attribute__((aligned(16)));
    const add_consts_t *cst = &cs;
    const int32_t rso = -out_shift;
    cs.off1 = ADD_S16(off1);
    cs.off2 = ADD_S16(off2);
    add_in_consts(m1, s1, &cs.mh1, &cs.ml1, &cs.k1, &cs.sm1);
    add_in_consts(m2_in, s2, &cs.mh2, &cs.ml2, &cs.k2, &cs.sm2);
    cs.ko = 1 + (rso ? (1 << rso) : 0) + (int32_t)((uint32_t) out_offset << (rso + 1));
    cs.smo = rso ? -1 : 0;

    const int32_t R = 30 - left_shift;
    cs.p1 = cs.p2 = ADD_S16(1 << (16 - R));
    const uint32_t m2 = (uint32_t) mo * 2;
    register const add_consts_t *pc asm("x15") = cst;
    register int32_t z asm("x12") = 0;
    register int32_t slo asm("x24") = R;
    register int32_t se1 asm("x25") = 1 - s1;
    register int32_t se2 asm("x9") = 1 - s2;
    register int32_t so asm("x11") = rso + 1;
    register int32_t lo asm("x30") = act_min;
    register int32_t hi asm("x31") = act_max;

    const int32_t cnt = size >> 3;
    /* Prefetch steps stop ADD_PF_DIST bytes before the end. */
    const int32_t pf_steps = ADD_PF_DIST / 8;
    const int32_t c1 = (cnt > pf_steps && (add_in_ext_mem(a) || add_in_ext_mem(b))) ?
                       cnt - pf_steps : 0;
    register int8_t *po asm("x26") = out;
    register const int8_t *pa asm("x27") = a;
    register const int8_t *pb asm("x14") = b;
    int32_t t0, t1;
    if (c1) {
        ADD_PIE_LOOP(ADD_PF_ON, c1);
    }
    ADD_PIE_LOOP(ADD_PF_OFF, cnt - c1);

    const int32_t r = size & 7;
    if (r) {
        /*
         * One more step on the last 8 elements, into a buffer. Only the
         * last r bytes are kept: the other lanes may read inputs that out
         * has already overwritten when out aliases an input.
         */
        int8_t tail[8] __attribute__((aligned(8)));
        po = tail;
        pa = a + size - 8;
        pb = b + size - 8;
        ADD_PIE_LOOP(ADD_PF_OFF, 1);
        memcpy(out + size - r, tail + 8 - r, r);
    }
    /* Other PIE kernels use SAR-dependent instructions and expect SAR = 0. */
    asm volatile ("li x28, 0\n\tesp.movx.w.sar x28" ::: "x28");
}

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

/* Scalar path, for fewer than 8 elements and for parameters outside the
 * PIE fast path. Out of line, so its loop keeps the register allocation it
 * has as a function of its own: inlined into the entry, it spilled a
 * register in the loop and was 4% slower. */
static __attribute__((noinline)) void add_scalar(const int8_t *input1_data,
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

/**
 * Elementwise add for s8, for ESP32-P4 and ESP32-S31.
 */
void esp_nn_add_elementwise_s8_riscv_pie(const int8_t *input1_data,
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
    if (size < 8 ||
        !add_fast_ok(input1_offset, input2_offset, input1_mult, input2_mult,
                     input1_shift, input2_shift, left_shift, out_offset,
                     out_mult, out_shift)) {
        add_scalar(input1_data, input2_data, input1_offset, input2_offset,
                   input1_mult, input2_mult, input1_shift, input2_shift, left_shift,
                   output, out_offset, out_mult, out_shift, activation_min,
                   activation_max, size);
        return;
    }
    ESP_NN_PIE_ENABLE();
    add_pie(input1_data, input2_data, output, size, input1_offset, input2_offset,
            input1_mult, input2_mult, input1_shift, input2_shift, left_shift,
            out_offset, out_mult, out_shift, activation_min, activation_max);
}
