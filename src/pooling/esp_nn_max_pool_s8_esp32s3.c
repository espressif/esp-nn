/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESP32-S3 max pool wrapper.
 *
 * The assembly has two vector paths with different requirements:
 *   channels % 8 == 0: ee.vld.l.64 / ee.vst.l.64, 8-byte buffers
 *   channels % 8 == 4: ee.vldbc.32 / s32i,       4-byte buffers
 * Neither has USAR/QUP compensation, so a buffer below that alignment would be
 * processed lane-shifted: the load reads the bytes before the window and the
 * store writes the previous bytes over the next pixel. The assembly also
 * writes nothing at all for channels % 4 != 0. Use the reference in all of
 * those cases; tensor data is normally 16-byte aligned, so the vector paths
 * stay the common ones.
 */

#include <stdint.h>
#include <esp_nn_ansi_headers.h>

extern void esp_nn_max_pool_s8_esp32s3_asm(const int8_t *input,
                                           const uint16_t input_wd,
                                           const uint16_t input_ht,
                                           int8_t *output,
                                           const uint16_t output_wd,
                                           const uint16_t output_ht,
                                           const uint16_t stride_wd,
                                           const uint16_t stride_ht,
                                           const uint16_t filter_wd,
                                           const uint16_t filter_ht,
                                           const uint16_t pad_wd,
                                           const uint16_t pad_ht,
                                           const int32_t activation_min,
                                           const int32_t activation_max,
                                           const uint16_t channels);

void esp_nn_max_pool_s8_esp32s3(const int8_t *input,
                                const uint16_t input_wd,
                                const uint16_t input_ht,
                                int8_t *output,
                                const uint16_t output_wd,
                                const uint16_t output_ht,
                                const uint16_t stride_wd,
                                const uint16_t stride_ht,
                                const uint16_t filter_wd,
                                const uint16_t filter_ht,
                                const uint16_t pad_wd,
                                const uint16_t pad_ht,
                                const int32_t activation_min,
                                const int32_t activation_max,
                                const uint16_t channels)
{
    /* Same split as avg_pool: the % 8 == 0 path uses ee.vld.l.64 /
     * ee.vst.l.64 and needs 8-byte buffers, the % 8 == 4 path uses
     * ee.vldbc.32 / s32i and needs 4-byte ones. The assembly writes nothing
     * for channels % 4 != 0, so the reference handles that as well as any
     * buffer below the alignment its path needs. */
    const uintptr_t bufs = (uintptr_t) input | (uintptr_t) output;
    const uintptr_t need = (channels % 8 == 0) ? 7u : 3u;
    if (channels % 4 == 0 && (bufs & need) == 0) {
        esp_nn_max_pool_s8_esp32s3_asm(input, input_wd, input_ht, output,
                                       output_wd, output_ht, stride_wd, stride_ht,
                                       filter_wd, filter_ht, pad_wd, pad_ht,
                                       activation_min, activation_max, channels);
        return;
    }

    esp_nn_max_pool_s8_ansi(input, input_wd, input_ht, output, output_wd,
                            output_ht, stride_wd, stride_ht, filter_wd, filter_ht,
                            pad_wd, pad_ht, activation_min, activation_max,
                            channels);
}
