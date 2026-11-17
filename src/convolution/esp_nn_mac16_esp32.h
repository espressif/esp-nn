/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * The ESP32 MAC16 asm routines (esp_nn_conv_mac16_esp32.S,
 * esp_nn_depthwise_conv_mac16_esp32.S) used by the conv and depthwise
 * kernels. Each .S file documents its routines.
 */
#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <sdkconfig.h>
#include <esp_memory_utils.h>

/*
 * Whether the MAC16 kernels may run on these buffers.
 * - CONFIG_NN_SKIP_NUDGE: the asm requant always rounds exactly, so the
 *   generic kernels run and the results match the reference build.
 * - CONFIG_SPIRAM_CACHE_WORKAROUND: the hand-written asm has none of the
 *   compiler's PSRAM workaround, so buffers in PSRAM take the generic
 *   kernels on those builds. The asm and the C around it also store to the
 *   caller's stack, so a task stack in PSRAM takes them too.
 */
static inline bool esp32_mac16_data_ok(const void *in, const void *filter,
                                       const void *out, const void *scratch)
{
#if defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE)
    return false;
#elif defined(CONFIG_SPIRAM_CACHE_WORKAROUND)
    int stack_probe;
    return !esp_ptr_external_ram(in) && !esp_ptr_external_ram(filter) &&
           !esp_ptr_external_ram(out) && !esp_ptr_external_ram(scratch) &&
           !esp_ptr_external_ram(&stack_probe);
#else
    (void) in; (void) filter; (void) out; (void) scratch;
    return true;
#endif
}

/*
 * The conv asm (esp_nn_conv_rq256s_oc_esp32) also loads the bias, mult and
 * shift arrays itself, so on CONFIG_SPIRAM_CACHE_WORKAROUND builds those
 * must not be in PSRAM either. The depthwise reads them in C.
 */
static inline bool esp32_mac16_quant_ok(const int32_t *bias, const int32_t *mult,
                                        const int32_t *shift)
{
#if defined(CONFIG_SPIRAM_CACHE_WORKAROUND) && \
        !(defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE))
    return (bias == NULL || !esp_ptr_external_ram(bias)) &&
           !esp_ptr_external_ram(mult) && !esp_ptr_external_ram(shift);
#else
    (void) bias; (void) mult; (void) shift;
    return true;
#endif
}

/* esp_nn_conv_mac16_esp32.S */
void esp_nn_conv_dot_mac16_esp32(const int16_t *filter, const int16_t *input,
                                 int32_t *out, int32_t n_pix, int32_t len4,
                                 int32_t bias);
void esp_nn_conv_dot_rq_mac16_esp32(const int16_t *filter, const int16_t *input,
                                    int8_t *out, int32_t n_pix, int32_t len4,
                                    const int32_t *p);
void esp_nn_conv_rq256s_oc_esp32(int16_t *filter, const int16_t *input, int8_t *out,
                                 int32_t n_pix, int32_t len4, const int32_t *q);
void esp_nn_conv_dot_rq256w_mac16_esp32(const int16_t *filter, const int16_t *input,
                                        int8_t *out, int32_t n_pix, int32_t len4,
                                        const int32_t *p);
void esp_nn_widen_lanes_esp32(int16_t *dst, const int8_t *src, int32_t n4);
void esp_nn_widen_perm_esp32(int16_t *dst, const int8_t *src, int32_t n, int32_t offset);
void esp_nn_widen_u8_esp32(int16_t *dst, const int8_t *src, int32_t n, int32_t offset);
void esp_nn_widen_s8_s16_esp32(int16_t *dst, const int8_t *src, int32_t n,
                               int32_t offset);
void esp_nn_conv_requant_store_esp32(const int32_t *sums, int32_t n, int8_t *out,
                                     int32_t out_step, const int32_t *params);

/* esp_nn_depthwise_conv_mac16_esp32.S */
void esp_nn_dw_zero_esp32(int16_t *dst, int32_t n_cols, int32_t half);
void esp_nn_dw_widen_t_esp32(int16_t *dst, const int8_t *src, int32_t n_x,
                             int32_t n_c, const int32_t *p);
void esp_nn_dw_widen_u8_esp32(int16_t *dst, const int8_t *src, int32_t n_x,
                              int32_t n_c, const int32_t *p);
void esp_nn_dw3x3_dot_esp32(const int16_t *plane, int32_t *sums, int32_t out_wd,
                            int32_t n_rows, const int32_t *fp);
void esp_nn_dw3x3_dot_rq_esp32(const int16_t *plane, int8_t *out, int32_t out_wd,
                               int32_t n_rows, const int32_t *fp);
