/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Byte offsets of the s3mp_args_t fields that the max pool assembly reads.
 * esp_nn_max_pool_s8_esp32s3.c checks them against the struct. Only
 * #defines here: the .S file includes this header too.
 */
#pragma once

#define S3MP_N32            0
#define S3MP_N16            4
#define S3MP_N8             8
#define S3MP_N4             12
#define S3MP_CHANNELS       16
#define S3MP_IN_STEP        20
#define S3MP_ROW_BYTES      24
#define S3MP_ACT_MIN        28  /* int8 */
#define S3MP_ACT_MAX        29  /* int8 */
#define S3MP_N16_FULL       32
#define S3MP_IN_ROW_STEP    36
#define S3MP_OUT_ROW_STEP   40
#define S3MP_ROWS           44
#define S3MP_ROWS_LEFT      48
