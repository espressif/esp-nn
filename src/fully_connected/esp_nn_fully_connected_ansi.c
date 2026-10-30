// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

#include <common_functions.h>
#include <esp_nn.h>

void esp_nn_fully_connected_s8_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t out_shift,
                                    const int32_t out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{
    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        int32_t result = 0;
        for (int32_t data_idx = 0; data_idx < row_len; data_idx++) {
            int32_t filter_index = row_len * out_c + data_idx;
            int32_t input_val = input_data[data_idx];
            int32_t filter_val = filter_data[filter_index];
            result += (filter_val + filter_offset) * (input_val + input_offset);
        }
        if (bias) {
            result += bias[out_c];
        }
        result = esp_nn_multiply_by_quantized_mult(result, out_mult, out_shift);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}

void esp_nn_fully_connected_per_ch_s8_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t* out_shift,
                                    const int32_t* out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{
    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        int32_t result = 0;
        for (int32_t data_idx = 0; data_idx < row_len; data_idx++) {
            int32_t filter_index = row_len * out_c + data_idx;
            int32_t input_val = input_data[data_idx];
            int32_t filter_val = filter_data[filter_index];
            result += (filter_val + filter_offset) * (input_val + input_offset);
        }
        if (bias) {
            result += bias[out_c];
        }
        result = esp_nn_multiply_by_quantized_mult(result, out_mult[out_c], out_shift[out_c]);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}

/* A per-channel batch with weight zero-point 0 is a 1x1 convolution:
 * height = batches, width = 1, in_ch = row_len, out_ch = out_channels.
 * The conv kernel already splits that work across both cores when
 * esp_nn_dual_core_enable() has been called. Small shapes, a nonzero weight
 * zero-point, an unaligned input or filter, and a scratch buffer whose
 * recorded size is below esp_nn_get_conv_scratch_size for that 1x1 shape stay
 * on the fully connected kernel. The input zero-point may be nonzero. A
 * buffer installed without a length has no recorded size. */
int esp_nn_fc_per_ch_s8_batch_via_conv(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches)
{
    if (filter_offset != 0 || batches < 4 || out_channels < 16 ||
            row_len < 64 || (row_len & 7) != 0 ||
            (((uintptr_t) input_data) & 7) != 0 ||
            (((uintptr_t) filter_data) & 7) != 0) {
        return 0;
    }

    const data_dims_t input_dims = {
        .width = 1, .height = batches, .channels = row_len,
    };
    const data_dims_t filter_dims = {
        .width = 1, .height = 1, .channels = row_len,
    };
    const data_dims_t output_dims = {
        .width = 1, .height = batches, .channels = out_channels,
    };
    const conv_params_t conv_params = {
        .in_offset = input_offset,
        .out_offset = out_offset,
        .stride = { .width = 1, .height = 1 },
        .padding = { .width = 0, .height = 0 },
        .dilation = { .width = 1, .height = 1 },
        .activation = { .min = activation_min, .max = activation_max },
    };
    const int scratch_need = esp_nn_get_conv_scratch_size(
            &input_dims, &filter_dims, &output_dims, &conv_params);
    if (!esp_nn_conv_scratch_fits(scratch_need)) {
        return 0;
    }
    quant_data_t quant = {
        .shift = (int32_t *) out_shift,
        .mult = (int32_t *) out_mult,
    };
    esp_nn_conv_s8(&input_dims, input_data, &filter_dims, filter_data, bias,
                   &output_dims, out_data, &conv_params, &quant);
    return 1;
}

/* Portable reference for the batched per-channel FC API: loop the per-row
 * kernel. Optimized targets override the macro with a weight-stationary impl.
 * A qualifying zero-point-0 batch is the 1x1 convolution above. */
void esp_nn_fully_connected_per_ch_s8_batch_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t* out_shift,
                                    const int32_t* out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches)
{
    if (esp_nn_fc_per_ch_s8_batch_via_conv(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult, activation_min,
                                           activation_max, batches)) {
        return;
    }
    for (int32_t b = 0; b < batches; b++) {
        esp_nn_fully_connected_per_ch_s8_ansi(input_data + b * row_len, input_offset,
                                    row_len, filter_data, filter_offset, bias,
                                    out_data + b * out_channels, out_channels,
                                    out_offset, out_shift, out_mult,
                                    activation_min, activation_max);
    }
}
