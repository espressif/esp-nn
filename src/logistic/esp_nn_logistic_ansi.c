/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <math.h>

/*
 * LUT-based int8 logistic (sigmoid) for quantized inference.
 *
 * For int8, there are only 256 possible input values. We precompute sigmoid
 * for all of them during Prepare() and store as a 256-byte LUT.
 * Eval() then becomes a trivial table lookup — O(1) per element.
 *
 * Output quantization is fixed: scale = 1/256, zero_point = -128.
 * This matches TFLite's convention for int8 logistic output.
 */

int32_t esp_nn_get_logistic_s8_scratch_size_ansi(void)
{
    return 256; /* LUT: one int8 output per possible int8 input */
}

void esp_nn_logistic_s8_prepare_ansi(int8_t *lut,
                                      int32_t input_zero_point,
                                      float input_scale)
{
    /* Build LUT: for each possible int8 input value (-128..127),
     * compute sigmoid and quantize to output int8.
     *
     * Output quant: scale=1/256, zero_point=-128
     * So output_int8 = clamp(round(sigmoid * 256) - 128, -128, 127)
     * Which simplifies to: output_int8 = clamp(round(sigmoid * 256) - 128, -128, 127)
     */
    for (int i = 0; i < 256; i++) {
        /* Index matches (uint8_t) cast of int8: i=0→int8(0), i=128→int8(-128) */
        int8_t input_val = (int8_t)i;
        float dequant = (input_val - input_zero_point) * input_scale;
        float sigmoid = 1.0f / (1.0f + expf(-dequant));

        /* Quantize to output: scale=1/256, zp=-128 */
        int32_t out_q = (int32_t)roundf(sigmoid * 256.0f) - 128;
        if (out_q < -128) out_q = -128;
        if (out_q > 127) out_q = 127;
        lut[i] = (int8_t)out_q;
    }
}

void esp_nn_logistic_s8_ansi(const int8_t *input, int8_t *output,
                              int32_t size, const int8_t *lut)
{
    for (int i = 0; i < size; i++) {
        output[i] = lut[(uint8_t)input[i]];
    }
}
