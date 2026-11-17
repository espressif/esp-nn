/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Conv for short window rows (filter_wd * in_ch < 32), out_ch a multiple
 * of 16: 3x3 first layers on RGB input, 3x3 on 8 channels.
 *
 * QACC lanes are 16 output channels. Each window byte is broadcast from
 * the input and multiplied with a 16-channel filter vector, so interior
 * pixels need no staging at all. The filter is transposed once per call.
 * Integer sums in another order: bit-exact.
 */

#include <common_functions.h>
#include "esp_nn_conv_oc16_riscv_pie.h"
#include "esp_nn_conv_tile_requant_riscv_pie.h"

#define OC16_PX 16  /* pixels per batch: one QACC tile per 16 channels */

/*
 * res[16 q + j] = sum over the window of pixel q of in[k] * flt[16 k + j],
 * for npx >= 1 pixels. Pixel q's window has `rows` rows of row_size bytes
 * (2 to 31), rstr[q] apart, starting at bases[q].
 *
 * Each row runs the last row_size slots of a 31-slot sequence, entered by
 * a computed jump. A slot is a fused MAC + next-filter load and a broadcast
 * load of the input byte two slots ahead. Input bytes alternate q0/q2 by
 * slot parity, so no load is used by the next instruction. The last two
 * slots load no input, so only the row's bytes are read. The filter is
 * read 16 bytes past its end.
 */
#define OC16_SLOT_PAIR \
        "esp.vmulas.s8.qacc.ld.ip q1, x31, 16, q0, q1 \n\t" \
        "esp.vldbc.8.xp  q0, x30, x28        \n\t" \
        "esp.vmulas.s8.qacc.ld.ip q1, x31, 16, q2, q1 \n\t" \
        "esp.vldbc.8.xp  q2, x30, x28        \n\t"

static __attribute__((noinline))
void oc16_mac(const int8_t *const *bases, const int32_t *rstr, int32_t npx,
              int32_t rows, const int8_t *flt, int32_t row_size,
              int32_t *res)
{
    /* Entry slot e = 31 - row_size. Slots 0 to 28 are 8 bytes. */
    const int32_t entry = 31 - row_size;
    asm volatile (
        /* 4-byte aligned 32-bit slots: a slot that straddles a fetch
         * boundary runs slower. */
        ".balign 4                           \n\t"
        ".option push                        \n\t"
        ".option norvc                       \n\t"
        "li      x28, 1                      \n\t"
        "la      t4, 3f                      \n\t"
        "add     t4, t4, %[eoff]             \n\t"
        "1:                                  \n\t"
        "lw      t0, 0(%[b])                 \n\t"
        "lw      t1, 0(%[r])                 \n\t"
        "mv      t2, %[rows]                 \n\t"
        "mv      x31, %[flt]                 \n\t"
        "esp.zero.qacc                       \n\t"
        "esp.vld.128.ip  q1, x31, 16         \n\t"
        "2:                                  \n\t"
        "mv      x30, t0                     \n\t"
        "bnez    %[odd], 4f                  \n\t"
        "esp.vldbc.8.xp  q0, x30, x28        \n\t"
        "esp.vldbc.8.xp  q2, x30, x28        \n\t"
        "jr      t4                          \n\t"
        "4:                                  \n\t"
        "esp.vldbc.8.xp  q2, x30, x28        \n\t"
        "esp.vldbc.8.xp  q0, x30, x28        \n\t"
        "jr      t4                          \n\t"
        "3:                                  \n\t"
        ".rept 14                            \n\t"
        OC16_SLOT_PAIR
        ".endr                               \n\t"
        "esp.vmulas.s8.qacc.ld.ip q1, x31, 16, q0, q1 \n\t"
        "esp.vldbc.8.xp  q0, x30, x28        \n\t"
        /* Slots 29 and 30: no input load */
        "esp.vmulas.s8.qacc.ld.ip q1, x31, 16, q2, q1 \n\t"
        "esp.vmulas.s8.qacc.ld.ip q1, x31, 16, q0, q1 \n\t"
        "add     t0, t0, t1                  \n\t"
        "addi    t2, t2, -1                  \n\t"
        "bnez    t2, 2b                      \n\t"
        "mv      x30, %[res]                 \n\t"
        "esp.st.qacc.l.l.128.ip  x30, 16     \n\t"
        "esp.st.qacc.l.h.128.ip  x30, 16     \n\t"
        "esp.st.qacc.h.l.128.ip  x30, 16     \n\t"
        "esp.st.qacc.h.h.128.ip  x30, 16     \n\t"
        "mv      %[res], x30                 \n\t"
        "addi    %[b], %[b], 4               \n\t"
        "addi    %[r], %[r], 4               \n\t"
        "addi    %[npx], %[npx], -1          \n\t"
        "bnez    %[npx], 1b                  \n\t"
        ".option pop                         \n\t"
        : [b] "+r"(bases), [r] "+r"(rstr), [npx] "+r"(npx), [res] "+r"(res)
        : [rows] "r"(rows), [flt] "r"(flt), [eoff] "r"(entry * 8),
          [odd] "r"(entry & 1)
        : "t0", "t1", "t2", "t4", "x28", "x30", "x31", "memory"
    );
}

/*
 * Requant of 16 pixels x 16 channels. res[16 q + j] is pixel q, channel j;
 * prm[4 j ..] = {2 comb, 2 mult, rnd, e1} of channel j. Channel j goes
 * through the PIE tile requant (lanes = pixels, so one shift per row) into
 * tile[16 j + q]: about 7 instructions per output against 10 for a scalar
 * requant with per-lane shifts.
 */
static inline __attribute__((always_inline))
void oc16_requant16(const int32_t *res, const int32_t *prm, int32_t *buf,
                    int8_t *tile, int32_t act_min, int32_t act_max)
{
    for (int32_t j = 0; j < 16; j++, prm += 4) {
        TILE_RQ_SCALAR(res + j, buf, prm[0], prm[1], 64);
        esp_nn_conv_requant_tile_vec(buf, prm + 2, prm[3], act_min, act_max,
                                     tile + 16 * j);
    }
}

/* Stages the window of a border pixel at d, with pad_val outside the
 * input. Kept out of line so the interior loop stays in registers. */
__attribute__((noinline))
static void oc16_stage_border(const int8_t *input_data, int32_t input_wd,
                              int32_t input_ht, int32_t in_ch,
                              int32_t filter_wd, int32_t filter_ht,
                              int32_t y0, int32_t x0, int8_t pad_val,
                              int8_t *d)
{
    for (int32_t fy = 0; fy < filter_ht; fy++) {
        const int32_t iy = y0 + fy;
        const bool row_in = iy >= 0 && iy < input_ht;
        for (int32_t fx = 0; fx < filter_wd; fx++) {
            const int32_t ix = x0 + fx;
            if (row_in && ix >= 0 && ix < input_wd) {
                const int8_t *src = input_data + (iy * input_wd + ix) * in_ch;
                for (int32_t c = 0; c < in_ch; c++) {
                    *d++ = src[c];
                }
            } else {
                for (int32_t c = 0; c < in_ch; c++) {
                    *d++ = pad_val;
                }
            }
        }
    }
}

void esp_nn_conv_s8_oc16_pie(const data_dims_t *input_dims,
                             const int8_t *input_data,
                             const data_dims_t *filter_dims,
                             const int8_t *filter_data,
                             const data_dims_t *output_dims,
                             int8_t *out_data,
                             const conv_params_t *conv_params,
                             const int32_t *tab, int8_t *work)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t in_ch = input_dims->channels;
    const int32_t filter_wd = filter_dims->width;
    const int32_t filter_ht = filter_dims->height;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ht = output_dims->height;
    const int32_t out_ch = output_dims->channels;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;
    const int8_t pad_val = (int8_t)(-conv_params->in_offset);
    const int32_t row_size = filter_wd * in_ch;
    const int32_t window_len = row_size * filter_ht;
    const int32_t in_rstride = input_wd * in_ch;

    /* Unaligned output stores: CFG is per core and not saved per task. */
    ESP_NN_PIE_ENABLE();

    /* Kept in the work area, not on the stack: this kernel can run on
     * the dual-core worker, whose stack is small. */
    int32_t *results = (int32_t *)work;
    int32_t *buf = results + OC16_PX * 16;
    int8_t *tile = (int8_t *)(buf + 16);
    int8_t *ftrans = work + ESP_NN_CONV_OC16_BUF_BYTES;
    int8_t *win = ftrans + out_ch * window_len + 16;

    /* ftrans[(b * window_len + k) * 16 + j] = filter[(16 b + j) * wl + k] */
    for (int32_t oc = 0; oc < out_ch; oc++) {
        const int8_t *src = filter_data + oc * window_len;
        int8_t *dst = ftrans + (oc >> 4) * window_len * 16 + (oc & 15);
        for (int32_t k = 0; k < window_len; k++) {
            dst[k * 16] = src[k];
        }
    }

    /* Output columns and rows whose window lies inside the input. */
    const int32_t ox_lo = (pad_wd + stride_wd - 1) / stride_wd;
    const int32_t oy_lo = (pad_ht + stride_ht - 1) / stride_ht;
    const int32_t ox_num = input_wd - filter_wd + pad_wd;
    const int32_t oy_num = input_ht - filter_ht + pad_ht;
    const int32_t ox_end = ox_num >= 0 ? ox_num / stride_wd + 1 : 0;
    const int32_t oy_end = oy_num >= 0 ? oy_num / stride_ht + 1 : 0;
    const int32_t px_step = stride_wd * in_ch;

    const int8_t *bases[OC16_PX];
    int32_t rstr[OC16_PX];
    const int32_t total = out_wd * out_ht;
    int32_t oy = 0, ox = 0;
    bool row_in = oy >= oy_lo && oy < oy_end;
    const int8_t *row_ptr = input_data
            + ((oy * stride_ht - pad_ht) * input_wd - pad_wd) * in_ch;
    for (int32_t pix = 0; pix < total; pix += OC16_PX) {
        const int32_t npx = (total - pix < OC16_PX) ? total - pix : OC16_PX;
        for (int32_t q = 0; q < npx; q++) {
            if (row_in && ox >= ox_lo && ox < ox_end) {
                bases[q] = row_ptr + ox * px_step;
                rstr[q] = in_rstride;
            } else {
                int8_t *d = win + q * window_len;
                oc16_stage_border(input_data, input_wd, input_ht, in_ch,
                                  filter_wd, filter_ht,
                                  oy * stride_ht - pad_ht,
                                  ox * stride_wd - pad_wd, pad_val, d);
                bases[q] = d;
                rstr[q] = row_size;
            }
            if (++ox == out_wd) {
                ox = 0;
                oy++;
                row_in = oy >= oy_lo && oy < oy_end;
                row_ptr += stride_ht * in_rstride;
            }
        }
        /* Lanes past the last pixel repeat it; they are never stored. */
        for (int32_t q = npx; q < OC16_PX; q++) {
            bases[q] = bases[npx - 1];
            rstr[q] = rstr[npx - 1];
        }
        int8_t *out = out_data + pix * out_ch;
        for (int32_t b = 0; b < out_ch; b += 16, out += 16) {
            oc16_mac(bases, rstr, OC16_PX, filter_ht, ftrans + b * window_len,
                     row_size, results);
            oc16_requant16(results, tab + 4 * b, buf, tile, act_min,
                           act_max);
            if (npx == OC16_PX) {
                esp_nn_conv_tile8x16_store(tile, out, out_ch);
                esp_nn_conv_tile8x16_store(tile + 8 * 16, out + 8, out_ch);
            } else {
                for (int32_t q = 0; q < npx; q++) {
                    for (int32_t j = 0; j < 16; j++) {
                        out[q * out_ch + j] = tile[16 * j + q];
                    }
                }
            }
        }
    }
}
