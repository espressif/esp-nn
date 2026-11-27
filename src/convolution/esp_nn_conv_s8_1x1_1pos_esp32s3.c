/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 1x1 convolution of a single position: a matrix-vector product.
 *
 * The mult8 1x1 asm runs a lone position through its scalar leftover path,
 * 8 channels per step with s16 MACs. Here each filter row is one s8 dot
 * product (16 MACs per instruction) against an aligned copy of the input.
 * The input offset is folded in as offset * sum(row). The sum re-reads the
 * row while it is still in the cache, so the filter is fetched once.
 */

#include <stdint.h>
#include <string.h>
#include <common_functions.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_conv_dispatch_esp32s3.h"

/* `work` is 16-byte aligned and holds in_channels bytes. */
void esp_nn_conv_s8_1x1_1pos_esp32s3(const int8_t *input, uint16_t in_channels,
                                     int32_t input_offset,
                                     const int8_t *filter_data,
                                     const int32_t *bias, int8_t *out_data,
                                     uint16_t out_channels, int32_t out_offset,
                                     const int32_t *out_shift,
                                     const int32_t *out_mult,
                                     int32_t activation_min,
                                     int32_t activation_max, int8_t *work)
{
    const int n16 = in_channels >> 4;
    const int simd_len = n16 << 4;

    memcpy(work, input, in_channels);
    for (int ch = 0; ch < out_channels; ch++) {
        const int8_t *row = filter_data + ch * in_channels;
        int32_t acc = 0;
        if (n16 > 0) {
            acc = esp_nn_dot_s8_unaligned_esp32s3(work, row, n16);
        }
        for (int i = simd_len; i < in_channels; i++) {
            acc += (int32_t)work[i] * row[i];
        }
        if (input_offset != 0) {
            acc += input_offset * esp_nn_filter_sum_s8_esp32s3(row, in_channels);
        }
        if (bias) {
            acc += bias[ch];
        }
        acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[ch], out_shift[ch]);
        acc += out_offset;
        acc = max(acc, activation_min);
        acc = min(acc, activation_max);
        out_data[ch] = (int8_t)acc;
    }
}
