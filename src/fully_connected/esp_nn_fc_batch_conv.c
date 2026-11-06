/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Batched per-channel fully connected as a 1x1 convolution. Built for the
 * S3 and PIE targets, whose 1x1 kernels win over the fully connected one. */

#include <stddef.h>
#include <stdint.h>

#include "sdkconfig.h"
#include <common_functions.h>
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "esp_nn_fc_internal.h"

/* The ANSI build (CONFIG_NN_ANSI_C) still compiles this file. */
#if ESP_NN_FC_BATCH_VIA_CONV

/* Measured ESP32-S3 floors. One core beats the S3 fully connected kernel
 * from batch 3, input 8, and output 8. Batch 1 and 2 are faster only when
 * the shared 1x1 kernel splits across both cores: the row length is a
 * multiple of 8, filter bytes are above 24 KiB, and there are at least 32
 * outputs. The filter and output numbers match the split in
 * esp_nn_conv_esp32s3.c. esp_nn_dual_core_active() means the worker exists.
 * That kernel still runs the same shape on one core when its worker scratch
 * does not fit, or when esp_nn_dual_core_begin() refuses a nested call or a
 * call already on the worker's core. Those one-core shapes lose to fully
 * connected. This gate cannot see that refusal. */
enum {
    ESP_NN_FC_S3_MIN_BATCH = 3,
    ESP_NN_FC_S3_MIN_ROW = 8,
    ESP_NN_FC_S3_MIN_OUT = 8,
    ESP_NN_FC_S3_DUAL_MIN_OUT = 32,
    ESP_NN_FC_S3_DUAL_MIN_FILTER_BYTES = 24 * 1024,
};

/* Measured ESP32-P4 floors. ESP32-S31 uses the same PIE kernels, so it
 * uses these floors too. Batch 1 already wins. Widths under 8 were not
 * measured. Output 4 wins. The shared convolution still starts its second
 * core at batch 4; this gate does not change that. */
enum {
    ESP_NN_FC_P4_MIN_BATCH = 1,
    ESP_NN_FC_P4_MIN_ROW = 8,
    ESP_NN_FC_P4_MIN_OUT = 4,
};

/* Performance only. The S3 1x1 fast path and its dual-core split store the
 * output 8 bytes at a time. Other outputs take the slower C path there,
 * which loses to the fully connected kernel. */
static int fc_out_aligned(const int8_t *out_data)
{
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
    return (((uintptr_t) out_data) & 7) == 0;
#else
    (void) out_data;
    return 1;
#endif
}

/* Performance only. Correctness checks stay in the caller. */
static int fc_conv_shape_wins(int32_t batches, uint16_t row_len,
                              uint16_t out_channels)
{
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
    if (row_len < ESP_NN_FC_S3_MIN_ROW) {
        return 0;
    }
    if (batches >= ESP_NN_FC_S3_MIN_BATCH &&
            out_channels >= ESP_NN_FC_S3_MIN_OUT) {
        return 1;
    }
    if (batches >= 1 && batches < ESP_NN_FC_S3_MIN_BATCH &&
            (row_len & 7) == 0 &&
            out_channels >= ESP_NN_FC_S3_DUAL_MIN_OUT &&
            (uint32_t) row_len * (uint32_t) out_channels >
                (uint32_t) ESP_NN_FC_S3_DUAL_MIN_FILTER_BYTES &&
            esp_nn_dual_core_active()) {
        return 1;
    }
    return 0;
#elif defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
    return batches >= ESP_NN_FC_P4_MIN_BATCH &&
            row_len >= ESP_NN_FC_P4_MIN_ROW &&
            out_channels >= ESP_NN_FC_P4_MIN_OUT;
#else
#error "esp_nn_fc_batch_conv.c is built only for the optimized S3 and PIE targets"
#endif
}

/* The buffer is usable when this 1x1 shape needs workspace, the pointer is
 * 16-byte aligned, and the length covers that workspace. */
static int fc_scratch_fits(int bytes, const void *scratch, size_t scratch_size)
{
    if (bytes <= 0 || scratch == NULL) {
        return 0;
    }
    return (((uintptr_t) scratch) & 15) == 0 &&
            scratch_size >= (size_t) bytes;
}

/* A per-channel batch with weight zero-point 0 is a 1x1 convolution:
 * height = batches, width = 1, in_ch = row_len, out_ch = out_channels.
 * Convolution supplies the second core when its own split applies, so
 * esp_nn_dual_core_enable() still has to have been called. The input
 * zero-point may be nonzero.
 *
 * Correctness: a nonzero weight zero-point stays on fully connected because
 * convolution has no weight-zero-point term. Batches above UINT16_MAX stay
 * there too: the S3 and PIE kernels store height in uint16_t, and this path
 * maps batches to that height. The input and filter pointers must be 8-byte
 * aligned. A scratch buffer that is missing, not 16-byte aligned, or
 * shorter than esp_nn_get_conv_scratch_size for this 1x1 shape also stays
 * on fully connected. The convolution scratch pointer is not read. For the
 * duration of the convolution call, that pointer is set to scratch and then
 * restored.
 *
 * Performance: fc_conv_shape_wins is the per-chip floor from the board
 * sweeps. On S3 the output must also be 8-byte aligned (fc_out_aligned).
 * The S3 fast assembly still requires channels % 8 inside the convolution
 * kernel and uses its existing fallback for other widths.
 * Batch 1 and 2 require that multiple of 8 here as well. A live worker
 * is also required, and the convolution can still decline the split and
 * finish on one core. */
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
                                    const int32_t batches,
                                    void *scratch,
                                    size_t scratch_size)
{
    if (batches > UINT16_MAX ||
            filter_offset != 0 ||
            (((uintptr_t) input_data) & 7) != 0 ||
            (((uintptr_t) filter_data) & 7) != 0 ||
            !fc_out_aligned(out_data) ||
            !fc_conv_shape_wins(batches, row_len, out_channels)) {
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
    if (!fc_scratch_fits(scratch_need, scratch, scratch_size)) {
        return 0;
    }
    quant_data_t quant = {
        .shift = (int32_t *) out_shift,
        .mult = (int32_t *) out_mult,
    };
    void *saved_scratch = NULL;
    esp_nn_conv_scratch_borrow(scratch, &saved_scratch);
    esp_nn_conv_s8(&input_dims, input_data, &filter_dims, filter_data, bias,
                   &output_dims, out_data, &conv_params, &quant);
    esp_nn_conv_scratch_restore(saved_scratch);
    return 1;
}

#endif /* ESP_NN_FC_BATCH_VIA_CONV */
