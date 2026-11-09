/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 scratch-size getters (conv, depthwise, softmax). Pure C with no target code, so
 * this file also builds on a host: a code generator can call the same getters
 * the kernels are written against instead of mirroring their arithmetic.
 *
 * The conv and depthwise sizing lives in esp_nn_conv_dispatch_esp32s3.h,
 * which the kernels include too, so a dispatch change updates both.
 */

#include <stdint.h>
#include <stddef.h>
#include <esp_nn_defs.h>
#include <esp_nn_esp32s3.h>

#include "esp_nn_conv_dispatch_esp32s3.h"

int esp_nn_get_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    return esp_nn_conv_scratch_size_esp32s3_impl(input_dims, filter_dims,
                                                 output_dims, conv_params);
}

int esp_nn_get_depthwise_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                                   const data_dims_t *filter_dims,
                                                   const data_dims_t *output_dims,
                                                   const dw_conv_params_t *conv_params)
{
    return esp_nn_depthwise_conv_scratch_size_esp32s3_impl(input_dims, filter_dims,
                                                           output_dims, conv_params);
}

int32_t esp_nn_get_softmax_scratch_size_esp32s3(const int32_t width, const int32_t height)
{
    (void) width;
    (void) height;
    /* Two 256-entry per-layer LUTs (raw exp + accumulation-scaled exp). */
    return 2 * 256 * 4 + 16;
}
