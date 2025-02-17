/*
 * SPDX-FileCopyrightText: 2020-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <common_functions.h>
#include <stdio.h>

/* mult value range */
#define MULT_MAX    INT32_MAX
#define MULT_MIN    0

/* shift value range */
#define SHIFT_MIN   -31
#define SHIFT_MAX   30

/**
 * @brief callback function to run before C function
 */
void profile_c_start();

/**
 * @brief callback function to run after C function
 *
 * @return uint32_t cycles consumed running C function
 */
uint32_t profile_c_end();

/**
 * @brief callback function to run before optimized function
 */
void profile_opt_start();

/**
 * @brief callback function to run after optimized function
 *
 * @return uint32_t cycles consumed running optimized function
 */
uint32_t profile_opt_end();

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define CHECK_EQUAL(ARRAY1, ARRAY2, size) ({    \
    bool res = true;                            \
    for (int _i = 0; _i < size; _i++) {         \
        if (ARRAY1[_i] != ARRAY2[_i]) {         \
            res = false;                        \
            break;                              \
        }                                       \
    }                                           \
    res;                                        \
})

#define PRINT_ARRAY_INT(ARRAY, width, height) ({        \
    int *_array = (int *) ARRAY;                        \
    for (int _j = 0; _j < height; _j++) {               \
        for (int _i = 0; _i < width; _i++) {            \
            printf("%d\t", _array[width * _j + _i]);    \
        }                                               \
        printf("\n");                                   \
    }                                                   \
    printf("\n");                                       \
})

#define PRINT_ARRAY_HEX(ARRAY, width, height) ({        \
    uint8_t *_array = (uint8_t *) ARRAY;                \
    for (int _j = 0; _j < height; _j++) {               \
        for (int _i = 0; _i < width; _i++) {            \
            printf("%02x\t", _array[width * _j + _i]);  \
        }                                               \
        printf("\n");                                   \
    }                                                   \
    printf("\n");                                       \
})

#define PRINT_ARRAY_INT8(ARRAY, width, height) ({        \
    int8_t *_array = (int8_t *) ARRAY;                \
    for (int _j = 0; _j < height; _j++) {               \
        for (int _i = 0; _i < width; _i++) {            \
            printf("%4d ", _array[width * _j + _i]);  \
        }                                               \
        printf("\n");                                   \
    }                                                   \
    printf("\n");                                       \
})

#if CONFIG_IDF_CMAKE
#if ((CONFIG_SPIRAM || CONFIG_SPIRAM_SUPPORT || CONFIG_ESP32S3_SPIRAM_SUPPORT) && \
        (CONFIG_SPIRAM_USE_CAPS_ALLOC || CONFIG_SPIRAM_USE_MALLOC))
#define IDF_HEAP_CAPS 1
#endif
#endif

#if IDF_HEAP_CAPS
#include "esp_heap_caps.h"
#define ESP_NN_TEST_ALLOC(SIZE) heap_caps_malloc(SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#else
#include <malloc.h>
#define ESP_NN_TEST_ALLOC(SIZE) malloc(SIZE)
#endif
