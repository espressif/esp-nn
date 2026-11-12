/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 1x1 stride-1 convolution with s8 MACs into ACCX.
 *
 * For in_channels a multiple of 16, up to 96, the filter row of one output
 * channel fits in q0-q5. The positions are contiguous in NHWC, so the input
 * streams through q7 with `ee.vmulas.s8.accx.ld.ip`: 16 MACs per cycle, into
 * the 40-bit ACCX, which cannot overflow. No transpose, no sign-extend.
 *
 * The input offset is folded into the bias: bias + offset * sum(row). The
 * row sum is one more ACCX pass against a vector of ones.
 *
 * The requant is the fast path of the mult8 1x1 asm, per output in scalar:
 *   u = mulsh(4 * acc, mult)
 *   out = (u + 1 + 2^rs - 2 * (u < -1) + (out_offset << (rs + 1))) >> (rs + 1)
 * It is exact when |acc| < 2^29, ls == 0 and 1 <= rs <= 21. Calls with any
 * channel outside that return 0, and the caller runs the general path.
 *
 * Positions run in blocks of about 16 KB of input and output, all output
 * channels per block, so both blocks stay in the cache when in PSRAM.
 *
 * The file also holds the MAC loop of the small-map 3x3 kernel, which
 * shares the chunked ACCX loops below.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "esp_nn_conv_inline_esp32s3.h"
#include "esp_nn_conv_dispatch_esp32s3.h"

#define ACCX_MAX_BLOCKS 6
/* Input + output bytes of one position block, kept within the D-cache.
 * Blocks stay at 64 positions or more: shorter blocks cost more in per-OC
 * setup than the cache saves. */
#define ACCX_BLOCK_BYTES (16 * 1024)

/* The fast requant needs |acc| < 2^29, with |acc| < (len << 15) + |bias|.
 * |bias| is taken as unsigned, so INT32_MIN has a defined value. */
static inline int accx_fast_bound_ok(int32_t len, int32_t bias)
{
    const uint32_t abs_bias = bias < 0 ? -(uint32_t)bias : (uint32_t)bias;
    return ((((uint32_t)len << 15) + abs_bias) >> 29) == 0;
}

/* The fast requant's rounding term, plus out_offset << (rs + 1). */
static inline int32_t accx_round_term(int32_t rs, int32_t out_offset)
{
    return 1 + (1 << rs) + (int32_t)((uint32_t)out_offset << (rs + 1));
}

/* The fast requant of one position in C, the same math as the asm pairs. */
static inline int8_t accx_requant_one(int64_t part, int32_t b4, int32_t mult,
                                      int32_t r, int32_t rs,
                                      int32_t act_min, int32_t act_max)
{
    int32_t t = (int32_t)((uint32_t)(int32_t)part * 4 + (uint32_t)b4);
    int32_t u = (int32_t)(((int64_t)t * mult) >> 32);
    t = (int32_t)((uint32_t)u + (uint32_t)r + 2 * (uint32_t)((u + 1) >> 31));
    t >>= rs + 1;
    t = t < act_min ? act_min : t > act_max ? act_max : t;
    return (int8_t)t;
}

/* One ACCX MAC of filter block j, loading the next input block into q7. */
#define MACLD(j) "ee.vmulas.s8.accx.ld.ip q7, %[in], 16, q7, q" #j "\n"
#define MAC(j)   "ee.vmulas.s8.accx q7, q" #j "\n"

#define REQUANT_STORE                       \
    "rur.accx_0   %[t]\n"                    \
    "addx4  %[t], %[t], %[b4]\n"            \
    "mulsh  %[t], %[t], %[mult]\n"          \
    "addi   %[m], %[t], 1\n"                \
    "srai   %[m], %[m], 31\n"               \
    "add    %[t], %[t], %[r]\n"             \
    "addx2  %[t], %[m], %[t]\n"             \
    "sra    %[t], %[t]\n"                   \
    "min    %[t], %[t], %[amax]\n"          \
    "max    %[t], %[t], %[amin]\n"          \
    "s8i    %[t], %[out], 0\n"              \
    "add    %[out], %[out], %[stride]\n"

/* Body for nblk filter blocks. The last position does not load past the
 * input: its last MAC has no load. */
#define ACCX_LOOP(MACS, MACS_LAST)                                          \
    __asm__ volatile(                                                       \
        "wsr.sar %[m]\n"                                                    \
        "ee.vld.128.ip q7, %[in], 16\n"                                     \
        "loopgtz %[cnt], 1f\n"                                              \
        "ee.zero.accx\n"                                                    \
        MACS                                                                \
        REQUANT_STORE                                                       \
        "1:\n"                                                              \
        "ee.zero.accx\n"                                                    \
        MACS_LAST                                                           \
        REQUANT_STORE                                                       \
        : [in] "+r"(in), [out] "+r"(out), [cnt] "+r"(cnt),                  \
          [t] "=&r"(t), [m] "+&r"(m)                                        \
        : [b4] "r"(b4), [mult] "r"(mult), [r] "r"(r), [amin] "r"(amin),     \
          [amax] "r"(amax), [stride] "r"(stride)                          \
        : "memory")

static inline void accx_run(int nblk, const int8_t *in, int8_t *out,
                            int npos, int32_t stride, int32_t b4,
                            int32_t mult, int32_t r, int32_t sar,
                            int32_t amin, int32_t amax)
{
    int32_t cnt = npos - 1;
    int32_t t, m = sar;
    switch (nblk) {
    case 1:
        ACCX_LOOP(MACLD(0), MAC(0));
        break;
    case 2:
        ACCX_LOOP(MACLD(0) MACLD(1), MACLD(0) MAC(1));
        break;
    case 3:
        ACCX_LOOP(MACLD(0) MACLD(1) MACLD(2), MACLD(0) MACLD(1) MAC(2));
        break;
    case 4:
        ACCX_LOOP(MACLD(0) MACLD(1) MACLD(2) MACLD(3),
                  MACLD(0) MACLD(1) MACLD(2) MAC(3));
        break;
    case 5:
        ACCX_LOOP(MACLD(0) MACLD(1) MACLD(2) MACLD(3) MACLD(4),
                  MACLD(0) MACLD(1) MACLD(2) MACLD(3) MAC(4));
        break;
    default:
        ACCX_LOOP(MACLD(0) MACLD(1) MACLD(2) MACLD(3) MACLD(4) MACLD(5),
                  MACLD(0) MACLD(1) MACLD(2) MACLD(3) MACLD(4) MAC(5));
        break;
    }
}

/* Two positions per step. The input streams two blocks ahead through q6
 * and q7, so no MAC waits on the load just before it. The caller makes sure
 * the two blocks after the last position are still input. */
#define M(s, f) "ee.vmulas.s8.accx.ld.ip q" #s ", %[in], 16, q" #s ", q" #f "\n"
#define ACCX_PAIR_LOOP(MACS_A, MACS_B)                                      \
    __asm__ volatile(                                                       \
        "wsr.sar %[m]\n"                                                    \
        "ee.vld.128.ip q6, %[in], 16\n"                                     \
        "ee.vld.128.ip q7, %[in], 16\n"                                     \
        "loopgtz %[cnt], 1f\n"                                              \
        "ee.zero.accx\n"                                                    \
        MACS_A                                                              \
        "rur.accx_0 %[ta]\n"                                                \
        "ee.zero.accx\n"                                                    \
        MACS_B                                                              \
        "addx4  %[ta], %[ta], %[b4]\n"                                      \
        "mulsh  %[ta], %[ta], %[mult]\n"                                    \
        "rur.accx_0 %[tb]\n"                                                \
        "addi   %[m], %[ta], 1\n"                                           \
        "addx4  %[tb], %[tb], %[b4]\n"                                      \
        "srai   %[m], %[m], 31\n"                                           \
        "mulsh  %[tb], %[tb], %[mult]\n"                                    \
        "add    %[ta], %[ta], %[r]\n"                                       \
        "addx2  %[ta], %[m], %[ta]\n"                                       \
        "addi   %[m], %[tb], 1\n"                                           \
        "sra    %[ta], %[ta]\n"                                             \
        "srai   %[m], %[m], 31\n"                                           \
        "add    %[tb], %[tb], %[r]\n"                                       \
        "min    %[ta], %[ta], %[amax]\n"                                    \
        "addx2  %[tb], %[m], %[tb]\n"                                       \
        "max    %[ta], %[ta], %[amin]\n"                                    \
        "sra    %[tb], %[tb]\n"                                             \
        "s8i    %[ta], %[out], 0\n"                                         \
        "min    %[tb], %[tb], %[amax]\n"                                    \
        "add    %[out], %[out], %[stride]\n"                                \
        "max    %[tb], %[tb], %[amin]\n"                                    \
        "s8i    %[tb], %[out], 0\n"                                         \
        "add    %[out], %[out], %[stride]\n"                                \
        "1:\n"                                                              \
        : [in] "+r"(in), [out] "+r"(out), [cnt] "+r"(cnt),                  \
          [ta] "=&r"(ta), [tb] "=&r"(tb), [m] "+&r"(m)                      \
        : [b4] "r"(b4), [mult] "r"(mult), [r] "r"(r), [amin] "r"(amin),     \
          [amax] "r"(amax), [stride] "r"(stride)                            \
        : "memory")

/* npos is even. */
static inline void accx_run_pairs(int nblk, const int8_t *in, int8_t *out,
                                  int npos, int32_t stride, int32_t b4,
                                  int32_t mult, int32_t r, int32_t sar,
                                  int32_t amin, int32_t amax)
{
    int32_t cnt = npos / 2;
    int32_t ta, tb, m = sar;
    switch (nblk) {
    case 1:
        ACCX_PAIR_LOOP(M(6, 0), M(7, 0));
        break;
    case 2:
        ACCX_PAIR_LOOP(M(6, 0) M(7, 1), M(6, 0) M(7, 1));
        break;
    case 3:
        ACCX_PAIR_LOOP(M(6, 0) M(7, 1) M(6, 2), M(7, 0) M(6, 1) M(7, 2));
        break;
    case 4:
        ACCX_PAIR_LOOP(M(6, 0) M(7, 1) M(6, 2) M(7, 3),
                       M(6, 0) M(7, 1) M(6, 2) M(7, 3));
        break;
    case 5:
        ACCX_PAIR_LOOP(M(6, 0) M(7, 1) M(6, 2) M(7, 3) M(6, 4),
                       M(7, 0) M(6, 1) M(7, 2) M(6, 3) M(7, 4));
        break;
    default:
        ACCX_PAIR_LOOP(M(6, 0) M(7, 1) M(6, 2) M(7, 3) M(6, 4) M(7, 5),
                       M(6, 0) M(7, 1) M(6, 2) M(7, 3) M(6, 4) M(7, 5));
        break;
    }
}

/* Loads nblk 16-byte filter blocks into q0.. and returns their sum.
 * The rows stay in q0-q5 for the loops that follow, which are separate asm
 * statements. That is safe because GCC for the S3 never allocates q
 * registers, and the RTOS saves them on a context switch. */
static inline int32_t accx_load_row(const int8_t *row, int nblk,
                                    const int8_t *one)
{
    int32_t sum, zero = 0;
    __asm__ volatile(
        "ee.zero.accx\n"
        "ee.vldbc.8 q7, %[one]\n"
        "ee.vld.128.ip q0, %[row], 16\n"
        "ee.vmulas.s8.accx q0, q7\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q1, %[row], 16\n"
        "ee.vmulas.s8.accx q1, q7\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q2, %[row], 16\n"
        "ee.vmulas.s8.accx q2, q7\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q3, %[row], 16\n"
        "ee.vmulas.s8.accx q3, q7\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q4, %[row], 16\n"
        "ee.vmulas.s8.accx q4, q7\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q5, %[row], 16\n"
        "ee.vmulas.s8.accx q5, q7\n"
        "2:\n"
        "ee.srs.accx %[sum], %[zero], 0\n"
        : [row] "+r"(row), [n] "+r"(nblk), [sum] "=r"(sum)
        : [one] "r"(one), [zero] "r"(zero)
        : "memory");
    return sum;
}

/*
 * in_channels over 96: the row does not fit in q0-q5. Each OC runs the row
 * in chunks of 4 to 6 blocks, as even as the block count allows. Per chunk the filter blocks stay in
 * q0..q(nb-1) and the positions stream. The ACCX partial sum of each
 * position is saved to part[] with `ee.st.accx.ip` and restored with
 * `ee.ld.accx.ip`. A requant pass over part[] then writes the outputs.
 *
 * The input streams two blocks ahead, as in the pair loop above. The load
 * of the last block of a position steps to the next position with
 * `.ld.xp` (skip = in_ch - 16 * (nb - 1)).
 */
#define ACCX_LARGE_BLOCK 64
#define ACCX_LARGE_MIN_POS 32
#define X(s, f) "ee.vmulas.s8.accx.ld.xp q" #s ", %[in], %[skip], q" #s ", q" #f "\n"

#define N(s, f) "ee.vmulas.s8.accx q" #s ", q" #f "\n"

/* The last pair runs out of the loop, with MACS_END in place of MACS_B:
 * its last two MACs load nothing, so no load goes past the input. */
#define ACCX_PART_LOOP(PRE, MACS_A, MACS_B, MACS_END)                       \
    __asm__ volatile(                                                       \
        "ee.vld.128.ip q6, %[in], 16\n"                                     \
        "ee.vld.128.ip q7, %[in], 16\n"                                     \
        "loopgtz %[cnt], 1f\n"                                              \
        PRE                                                                 \
        MACS_A                                                              \
        "ee.st.accx.ip %[p], 8\n"                                           \
        PRE                                                                 \
        MACS_B                                                              \
        "ee.st.accx.ip %[p], 8\n"                                           \
        "1:\n"                                                              \
        PRE                                                                 \
        MACS_A                                                              \
        "ee.st.accx.ip %[p], 8\n"                                           \
        PRE                                                                 \
        MACS_END                                                            \
        "ee.st.accx.ip %[p], 8\n"                                           \
        : [in] "+r"(in), [cnt] "+r"(cnt), [p] "+r"(p)                       \
        : [skip] "r"(skip)                                                  \
        : "memory")

/* The pair loop for 6-block chunks: the last pair loads nothing past its
 * own blocks. */
#define ACCX_PART_LOOP_6(PRE)                                               \
    ACCX_PART_LOOP(PRE, M(6, 0) M(7, 1) M(6, 2) X(7, 3) M(6, 4) M(7, 5),   \
                   M(6, 0) M(7, 1) M(6, 2) X(7, 3) M(6, 4) M(7, 5),        \
                   M(6, 0) M(7, 1) M(6, 2) X(7, 3) N(6, 4) N(7, 5))

#define ACCX_PART_NB(PRE)                                                   \
    switch (nb) {                                                           \
    case 4:                                                                 \
        ACCX_PART_LOOP(PRE, M(6, 0) X(7, 1) M(6, 2) M(7, 3),                \
                       M(6, 0) X(7, 1) M(6, 2) M(7, 3),                     \
                       M(6, 0) X(7, 1) N(6, 2) N(7, 3));                    \
        break;                                                              \
    case 5:                                                                 \
        ACCX_PART_LOOP(PRE, M(6, 0) M(7, 1) X(6, 2) M(7, 3) M(6, 4),        \
                       M(7, 0) M(6, 1) X(7, 2) M(6, 3) M(7, 4),             \
                       M(7, 0) M(6, 1) X(7, 2) N(6, 3) N(7, 4));            \
        break;                                                              \
    default:                                                                \
        ACCX_PART_LOOP_6(PRE);                                              \
        break;                                                              \
    }

/* npos is even and at least 2. The first chunk starts from zero, the others
 * from part[]. */
static void accx_part_pairs(int nb, int first, const int8_t *in, int npos,
                            int32_t skip, int64_t *p)
{
    int32_t cnt = npos / 2 - 1;
    if (first) {
        ACCX_PART_NB("ee.zero.accx\n");
    } else {
        ACCX_PART_NB("ee.ld.accx.ip %[p], 0\n");
    }
}

/* One position, no load past its nb (4 to 6) blocks. Inlined, so its
 * second caller (the small-map kernel) leaves accx_large unchanged. */
static inline __attribute__((always_inline))
void accx_part_one(int nb, const int8_t *in, int64_t *p)
{
    __asm__ volatile(
        "ee.ld.accx.ip %[p], 0\n"
        "ee.vld.128.ip q6, %[in], 16\n"
        "ee.vld.128.ip q7, %[in], 16\n"
        "ee.vmulas.s8.accx q6, q0\n"
        "ee.vld.128.ip q6, %[in], 16\n"
        "ee.vmulas.s8.accx q7, q1\n"
        "ee.vld.128.ip q7, %[in], 16\n"
        "ee.vmulas.s8.accx q6, q2\n"
        "ee.vmulas.s8.accx q7, q3\n"
        "addi %[n], %[n], -4\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q6, %[in], 16\n"
        "ee.vmulas.s8.accx q6, q4\n"
        "addi %[n], %[n], -1\n"
        "beqz %[n], 2f\n"
        "ee.vld.128.ip q7, %[in], 16\n"
        "ee.vmulas.s8.accx q7, q5\n"
        "2:\n"
        "ee.st.accx.ip %[p], 0\n"
        : [in] "+r"(in), [n] "+r"(nb), [p] "+r"(p)
        :
        : "memory");
}

/* Requant of an even number of partial sums, two per step. Inlined into
 * both callers, as accx_part_one. */
static inline __attribute__((always_inline))
void accx_requant_pairs(const int64_t *p, int8_t *out, int npos,
                        int32_t stride, int32_t b4, int32_t mult,
                        int32_t r, int32_t sar, int32_t amin, int32_t amax)
{
    int32_t cnt = npos / 2;
    int32_t ta, tb, m = sar;
    __asm__ volatile(
        "wsr.sar %[m]\n"
        "loopgtz %[cnt], 1f\n"
        "l32i   %[ta], %[p], 0\n"
        "l32i   %[tb], %[p], 8\n"
        "addi   %[p], %[p], 16\n"
        "addx4  %[ta], %[ta], %[b4]\n"
        "addx4  %[tb], %[tb], %[b4]\n"
        "mulsh  %[ta], %[ta], %[mult]\n"
        "mulsh  %[tb], %[tb], %[mult]\n"
        "addi   %[m], %[ta], 1\n"
        "srai   %[m], %[m], 31\n"
        "add    %[ta], %[ta], %[r]\n"
        "addx2  %[ta], %[m], %[ta]\n"
        "addi   %[m], %[tb], 1\n"
        "sra    %[ta], %[ta]\n"
        "srai   %[m], %[m], 31\n"
        "add    %[tb], %[tb], %[r]\n"
        "min    %[ta], %[ta], %[amax]\n"
        "addx2  %[tb], %[m], %[tb]\n"
        "max    %[ta], %[ta], %[amin]\n"
        "sra    %[tb], %[tb]\n"
        "s8i    %[ta], %[out], 0\n"
        "min    %[tb], %[tb], %[amax]\n"
        "add    %[out], %[out], %[stride]\n"
        "max    %[tb], %[tb], %[amin]\n"
        "s8i    %[tb], %[out], 0\n"
        "add    %[out], %[out], %[stride]\n"
        "1:\n"
        : [p] "+r"(p), [out] "+r"(out), [cnt] "+r"(cnt),
          [ta] "=&r"(ta), [tb] "=&r"(tb), [m] "+&r"(m)
        : [b4] "r"(b4), [mult] "r"(mult), [r] "r"(r), [amin] "r"(amin),
          [amax] "r"(amax), [stride] "r"(stride)
        : "memory");
}

/* noinline keeps part[] and row_copy out of the frame of the path up to
 * 96 channels. */
static __attribute__((noinline))
void accx_large(const int8_t *input, int spatial, int in_ch,
                       int32_t in_offset, const int8_t *filter,
                       const int32_t *bias, int8_t *out_data, int out_ch,
                       int32_t out_offset, const int32_t *out_mult,
                       const int32_t *out_shift, int32_t act_min,
                       int32_t act_max)
{
    int64_t part[ACCX_LARGE_BLOCK] __attribute__((aligned(16)));
    int8_t row_copy[16 * ACCX_MAX_BLOCKS] __attribute__((aligned(16)));
    static const int8_t one = 1;
    const int aligned = ((uintptr_t)filter & 15) == 0;
    /* ceil(nblk / 6) chunks. With nblk >= 8 each gets 4 to 6 blocks. */
    const int nblk = in_ch / 16;
    const int nchunk = (nblk + 5) / 6;

    for (int p0 = 0; p0 < spatial; p0 += ACCX_LARGE_BLOCK) {
        int npos = spatial - p0;
        if (npos > ACCX_LARGE_BLOCK) {
            npos = ACCX_LARGE_BLOCK;
        }
        const int fast = npos & ~1;
        const int8_t *in = input + p0 * in_ch;
        int8_t *out = out_data + p0 * out_ch;
        const int8_t *row = filter;
        for (int oc = 0; oc < out_ch; oc++, row += in_ch) {
            int32_t sum = 0;
            int off = 0;
            for (int c = 0; c < nchunk; c++) {
                const int nb = nblk / nchunk + (c < nblk % nchunk);
                const int32_t skip = in_ch - 16 * (nb - 1);
                const int8_t *src = row + off;
                if (!aligned) {
                    memcpy(row_copy, src, 16 * nb);
                    src = row_copy;
                }
                sum += accx_load_row(src, nb, &one);
                const int8_t *cin = in + off;
                if (fast > 0) {
                    accx_part_pairs(nb, c == 0, cin, fast, skip, part);
                }
                for (int i = fast; i < npos; i++) {
                    if (c == 0) {
                        part[i] = 0;
                    }
                    accx_part_one(nb, cin + i * in_ch, &part[i]);
                }
                off += 16 * nb;
            }
            int32_t b = (bias ? bias[oc] : 0) + in_offset * sum;
            int32_t rs = -out_shift[oc];
            int32_t r = accx_round_term(rs, out_offset);
            int32_t b4 = (int32_t)((uint32_t)b << 2);
            int even = npos & ~1;
            accx_requant_pairs(part, out + oc, even, out_ch, b4, out_mult[oc],
                               r, rs + 1, act_min, act_max);
            for (int i = even; i < npos; i++) {
                out[oc + i * out_ch] = accx_requant_one(part[i], b4, out_mult[oc],
                                                        r, rs, act_min, act_max);
            }
        }
    }
}


/*
 * 3x3 conv on a padded map that fits the cache, filter in PSRAM (see
 * esp_nn_conv_s8_3x3_small_map_esp32s3.c). in_ch % 32 == 0, so each window
 * row (3 * in_ch bytes, contiguous in the padded map) splits into chunks of
 * exactly 6 blocks. Output channels run outermost, so each filter window is
 * read from memory once. Per chunk the filter blocks sit in q0-q5 while the
 * positions of each output row stream at stride step_x, two blocks ahead,
 * with partial sums through part[] as in accx_large. It writes out_ch
 * channels; output positions are out_stride bytes apart.
 */
void esp_nn_conv_s8_3x3_small_map_accx(const int8_t *padded,
                                       int32_t padded_row, int in_ch,
                                       int32_t step_x, int32_t step_y,
                                       int out_wd, int out_ht,
                                       const int8_t *filter,
                                       const int32_t *bias, int32_t in_offset,
                                       int8_t *out_data, int out_ch,
                                       int out_stride, int32_t out_offset,
                                       const int32_t *out_shift,
                                       const int32_t *out_mult,
                                       int32_t act_min, int32_t act_max)
{
    int64_t part[ACCX_LARGE_BLOCK] __attribute__((aligned(16)));
    int8_t row_copy[96] __attribute__((aligned(16)));
    static const int8_t one = 1;
    const int aligned = ((uintptr_t)filter & 15) == 0;
    const int window = 9 * in_ch;
    const int nchunk = window / 96;
    const int row_chunks = nchunk / 3;
    const int32_t skip = step_x - 80;

    for (int oc = 0; oc < out_ch; oc++) {
        const int8_t *frow = filter + (int32_t)oc * window;
        const int32_t b0 = bias ? bias[oc] : 0;
        const int32_t mult = out_mult[oc];
        const int32_t shift = out_shift[oc];
        /* Same bound as the 1x1 fast requant: |acc| < (K << 15) + |bias|. */
        const int fast = shift <= -1 && shift >= -21 &&
                         accx_fast_bound_ok(window, b0);
        /* Blocks of whole output rows, at most 64 positions (one row if
         * the row is longer: then the row runs in pieces). */
        for (int oy = 0; oy < out_ht; ) {
            int rows = ACCX_LARGE_BLOCK / out_wd;
            rows = rows < 1 ? 1 : rows;
            rows = rows > out_ht - oy ? out_ht - oy : rows;
            for (int ox0 = 0; ox0 < out_wd; ) {
                int seg = out_wd - ox0;
                seg = seg > ACCX_LARGE_BLOCK ? ACCX_LARGE_BLOCK : seg;
                const int8_t *base = padded + oy * step_y + ox0 * step_x;
                const int8_t *src = frow;
                int32_t sum = 0;
                for (int c = 0; c < nchunk; c++, src += 96) {
                    const int8_t *row = src;
                    if (!aligned) {
                        memcpy(row_copy, src, 96);
                        row = row_copy;
                    }
                    sum += accx_load_row(row, 6, &one);
                    const int ky = c / row_chunks;
                    const int8_t *rowin = base + ky * padded_row
                                          + (c - ky * row_chunks) * 96;
                    int64_t *p = part;
                    for (int r = 0; r < rows; r++, rowin += step_y) {
                        if (seg >= 2) {
                            const int8_t *in = rowin;
                            int32_t cnt = (seg >> 1) - 1;
                            if (c == 0) {
                                ACCX_PART_LOOP_6("ee.zero.accx\n");
                            } else {
                                ACCX_PART_LOOP_6("ee.ld.accx.ip %[p], 0\n");
                            }
                        }
                        if (seg & 1) {
                            if (c == 0) {
                                *p = 0;
                            }
                            accx_part_one(6, rowin + (seg - 1) * step_x, p);
                            p++;
                        }
                    }
                }
                const int32_t corr = b0 + in_offset * sum;
                for (int r = 0; r < rows; r++) {
                    int8_t *out = out_data + oc
                                  + ((int32_t)(oy + r) * out_wd + ox0) * out_stride;
                    const int64_t *pr = part + r * seg;
                    if (fast) {
                        const int32_t rs = -shift;
                        const int32_t rr = accx_round_term(rs, out_offset);
                        const int32_t b4 = (int32_t)((uint32_t)corr << 2);
                        const int even = seg & ~1;
                        accx_requant_pairs(pr, out, even, out_stride, b4, mult, rr,
                                           rs + 1, act_min, act_max);
                        if (seg > even) {
                            out[even * out_stride] = accx_requant_one(pr[even], b4, mult, rr,
                                                                  rs, act_min, act_max);
                        }
                        continue;
                    }
                    for (int i = 0; i < seg; i++) {
                        int32_t acc = esp_nn_requant_s3((int32_t)pr[i] + corr,
                                                        mult, shift);
                        acc += out_offset;
                        acc = acc < act_min ? act_min
                              : acc > act_max ? act_max : acc;
                        out[i * out_stride] = (int8_t)acc;
                    }
                }
                ox0 += seg;
            }
            oy += rows;
        }
    }
}

int esp_nn_conv_s8_1x1_accx_esp32s3(const int8_t *input, int spatial,
                                    uint16_t in_ch, int32_t in_offset,
                                    const int8_t *filter, const int32_t *bias,
                                    int8_t *out_data, uint16_t out_ch,
                                    int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    int32_t act_min, int32_t act_max)
{
    if (in_ch < 16 || in_ch % 16 != 0 || spatial < 1 ||
            ((uintptr_t)input & 15) != 0) {
        return 0;
    }
    /* Over 96 channels the row runs in ceil(blocks / 6) chunks. Each chunk
     * needs at least 4 blocks: below that the partial-sum save and restore
     * cost more than they gain, so the layer stays on the general path. */
    if (in_ch > 16 * ACCX_MAX_BLOCKS) {
        const int nblk = in_ch / 16;
        const int nchunk = (nblk + ACCX_MAX_BLOCKS - 1) / ACCX_MAX_BLOCKS;
        if (nblk / nchunk < 4) {
            return 0;
        }
        /* The per-chunk setup needs enough positions to pay off. The
         * caller must rebuild the asm table after this decline, because
         * an earlier chunk of the same layer may have run here. */
        if (spatial < ACCX_LARGE_MIN_POS) {
            return -1;
        }
    }
    /* Same bound as the asm fast path: |acc| < (in_ch << 15) + |bias|. */
    for (int oc = 0; oc < out_ch; oc++) {
        int32_t b = bias ? bias[oc] : 0;
        int32_t sh = out_shift[oc];
        if (sh > -1 || sh < -21) {
            return 0;
        }
        if (!accx_fast_bound_ok(in_ch, b)) {
            return 0;
        }
    }

    if (in_ch > 16 * ACCX_MAX_BLOCKS) {
        accx_large(input, spatial, in_ch, in_offset, filter, bias,
                   out_data, out_ch, out_offset, out_mult, out_shift,
                   act_min, act_max);
        return 1;
    }

    const int nblk = in_ch / 16;
    const int aligned = ((uintptr_t)filter & 15) == 0;
    int8_t row_copy[16 * ACCX_MAX_BLOCKS] __attribute__((aligned(16)));
    static const int8_t one = 1;

    int block = ACCX_BLOCK_BYTES / (in_ch + out_ch);
    block = block < 64 ? 64 : block > 256 ? 256 : block & ~1;
    for (int p0 = 0; p0 < spatial; p0 += block) {
        int npos = spatial - p0;
        if (npos > block) {
            npos = block;
        }
        /* The pair loop loads two blocks past its last position. Those
         * must still be input, else the 1-ahead loop runs the rest. */
        int fast = spatial - p0 - (nblk == 1 ? 2 : 1);
        if (fast > npos) {
            fast = npos;
        }
        fast = fast < 0 ? 0 : fast & ~1;
        const int8_t *in = input + p0 * in_ch;
        int8_t *out = out_data + p0 * out_ch;
        const int8_t *row = filter;
        for (int oc = 0; oc < out_ch; oc++, row += in_ch) {
            const int8_t *src = row;
            if (!aligned) {
                memcpy(row_copy, row, in_ch);
                src = row_copy;
            }
            int32_t sum = accx_load_row(src, nblk, &one);
            int32_t b = (bias ? bias[oc] : 0) + in_offset * sum;
            int32_t rs = -out_shift[oc];
            int32_t r = accx_round_term(rs, out_offset);
            int32_t b4 = (int32_t)((uint32_t)b << 2);
            if (fast > 0) {
                accx_run_pairs(nblk, in, out + oc, fast, out_ch, b4,
                               out_mult[oc], r, rs + 1, act_min, act_max);
            }
            if (npos > fast) {
                accx_run(nblk, in + fast * in_ch, out + oc + fast * out_ch,
                         npos - fast, out_ch, b4, out_mult[oc], r, rs + 1,
                         act_min, act_max);
            }
        }
    }
    return 1;
}
