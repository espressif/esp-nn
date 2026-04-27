/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 optimized HardSwish using 256-byte lookup table.
 *
 * Key insight: HardSwish maps int8 -> int8 with fixed quantization parameters
 * per layer. Only 256 possible input values exist. We precompute the full
 * mapping once using the ANSI reference (bit-exact), then the inner loop
 * is a single byte load per element.
 *
 * Scratch buffer: 256 bytes (set via esp_nn_set_hard_swish_scratch_buf).
 */

#include <stdint.h>
#include <stddef.h>

/* Use ANSI C reference to build LUT — guarantees bit-exact match */
extern void esp_nn_hard_swish_s8_ansi(const int8_t *input,
                                       int8_t *output,
                                       const int32_t size,
                                       const int16_t input_zero_point,
                                       const int16_t output_mult_fxp,
                                       const int16_t reluish_mult_fxp,
                                       const int32_t reluish_mult_exp,
                                       const int32_t output_mult_exp,
                                       const int16_t output_zero_point);

static int8_t *hard_swish_scratch = NULL;

int32_t esp_nn_get_hard_swish_scratch_size_esp32s3(void)
{
    return 512; /* 256 for lut_input + 256 for lut output */
}

void esp_nn_set_hard_swish_scratch_buf_esp32s3(void *buf)
{
    hard_swish_scratch = (int8_t *)buf;
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
    if (!hard_swish_scratch) {
        /* No scratch — fall through to ANSI */
        esp_nn_hard_swish_s8_ansi(input, output, size,
                                   input_zero_point, output_mult_fxp,
                                   reluish_mult_fxp, reluish_mult_exp,
                                   output_mult_exp, output_zero_point);
        return;
    }

    /* Build 256-byte LUT using ANSI reference (bit-exact).
     * lut[i] = hardswish((int8_t)i) for the given quant params.
     * Indexed by (uint8_t)input_val for direct lookup. */
    int8_t *lut_input = hard_swish_scratch;
    int8_t *lut = hard_swish_scratch + 256;

    for (int i = 0; i < 256; i++) {
        lut_input[i] = (int8_t)i;
    }
    esp_nn_hard_swish_s8_ansi(lut_input, lut, 256,
                               input_zero_point, output_mult_fxp,
                               reluish_mult_fxp, reluish_mult_exp,
                               output_mult_exp, output_zero_point);

    /* Apply LUT — one byte load per element */
    for (int i = 0; i < size; i++) {
        output[i] = lut[(uint8_t)input[i]];
    }
}
