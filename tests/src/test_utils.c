/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

uint32_t esp_nn_test_passed;
uint32_t esp_nn_test_failed;
uint32_t esp_nn_test_skipped;

void esp_nn_test_pass(void)
{
    esp_nn_test_passed++;
}

void esp_nn_test_fail(void)
{
    esp_nn_test_failed++;
}

void esp_nn_test_skip(void)
{
    esp_nn_test_skipped++;
}
