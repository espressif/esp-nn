# ESP-NN

The library contains optimised NN (Neural Network) functions for various Espressif chips.

* Supported platforms:
   * TensorFlow Lite Micro (TFLite Micro). Repo can be found [here](https://github.com/espressif/tflite-micro-esp-examples)

* Supported ESP chips include:
   * ESP32-S3 (Assembly versions optimised to benefit from vector instructions of ESP32-S3)
   * ESP32-P4 (Optimised using PIE/QACC SIMD instructions)
   * ESP32-S31 (Optimised using PIE/QACC SIMD instructions, shares ESP32-P4 kernels)
   * ESP32 (Generic optimisations)
   * ESP32-C3 (Generic optimisations)

## Performance

### Kernel performance

Speedup of the optimized s8 kernels over the portable ANSI C ones (cycles, ANSI C / optimized):

| Kernel | Shape | ESP32-S3 | ESP32-P4 | ESP32-S31 |
| ------ | ----- | -------: | -------: | --------: |
| Convolution 1x1 | input 10x10, filter 64x1x1x64 | 14.24 | 7.00 | 6.24 |
| Convolution 1x1 | input 8x8, filter 16x1x1x16 | 8.01 | 3.51 | 2.94 |
| Depthwise conv 5x5 | input 12x12, filter 8x5x5x4 | 4.49 | 1.94 | 1.75 |
| Fully connected | length 271, 3 outputs | 5.47 | 8.36 | 5.11 |
| Elementwise multiply | 1615 elements | 3.51 | 1.60 | 1.82 |
| Max pooling 3x3 | input 16x16x16 | 7.83 | 19.94 | 12.65 |
| Average pooling 3x3 | input 16x16x16 | 3.62 | 3.59 | 3.95 |

All kernels and shapes, with cycle counts and the setup per chip, are in
[docs/performance.md](docs/performance.md).

### Model-level performance:

  * **Person Detection** (Visual Wake Words, INT8 quantized — from [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro))
    * Numbers are time (ms) for `invoke()` call, using internal memory

    | Chip     | CPU Freq | without ESP-NN | with ESP-NN |
    | -------- | -------- | -------------- | ----------- |
    | ESP32-P4 | 360MHz   | 1395ms         | 73ms        |
    | ESP32-S31 | 320MHz  | 1669ms         | 72ms        |
    | ESP32-S3 | 240MHz   | 2300ms         | 54ms        |
    | ESP32    | 240MHz   | 4084ms         | 380ms       |
    | ESP32-C3 | 160MHz   | 3355ms         | 426ms       |

  * **MobileNetV3 Small** (INT8 quantized, 224x224x3, 1000 classes)

    | Chip     | CPU Freq | without ESP-NN | with ESP-NN | with ESP-NN, dual core |
    | -------- | -------- | -------------- | ----------- | ---------------------- |
    | ESP32-P4 | 400MHz   | 10980ms        | 388ms       | 253ms                  |
    | ESP32-S3 | 240MHz   | 26679ms        | 729ms       | 690ms                  |

  * **YOLO11n** (INT8 quantized, 192x192x3, 80 classes, object detection)

    | Chip     | CPU Freq | without ESP-NN | with ESP-NN | with ESP-NN, dual core |
    | -------- | -------- | -------------- | ----------- | ---------------------- |
    | ESP32-P4 | 400MHz   | 52240ms        | 620ms       | 417ms                  |
    | ESP32-S3 | 240MHz   | 130753ms       | 1253ms      | 1053ms                 |

> **Note**:
  - The above is time taken for execution of the `invoke()` call
  - SPIRAM used for TensorArena.
  - Person detection on ESP32-S3 with internal RAM: 47ms
  - MobileNetV3 and YOLO11n were measured with the `mobilenet_v3` and
    `yolo_test_app` examples in
    [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro), with no
    profiler attached; ESP32-P4 with 200MHz HEX PSRAM, ESP32-S3 with 80MHz OCT
    PSRAM and a 64KB data cache
  - The `dual core` column is with `esp_nn_dual_core_enable()` called once at
    startup, which lets the heavier kernels split their work across both cores
  - `Without ESP-NN` is esp-tflite-micro with esp-nn turned off, so it runs the
    TFLite Micro reference kernels. Turn off "Use ESP-NN optimized kernels" in
    `idf.py menuconfig` (under `ESP TFLite-Micro`), or set
    `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=n` in `sdkconfig`.


## Configuration

  * To configure, please use `idf.py menuconfig` and under `ESP-NN` select `NN_OPTIMIZATIONS`
  * There are two options presented:
     * Optimized versions
     * ANSI C

  * Default selection is for `Optimized versions`. For ESP32-S3, ESP32-P4 and ESP32-S31, assembly versions are automatically selected, whereas for other chips (viz., ESP32, ESP32-C3), generic optimisations are selected.
  * For debugging purposes, you may want to select `ANSI C` reference versions.


## Contributing

If you encounter an issue with ESP-NN, or wish to submit a feature request, please use the Issues section on the Github.

For general questions related to this library, please use the esp32.com forum.

Please check [CONTRIBUTING.md](CONTRIBUTING.md) for further information if you'd like to contribute to ESP-NN.

## Copyrights and License

All original source code in this repository is Copyright (C) 2020-2026 Espressif Systems. This source code is licensed under the Apache License 2.0 as described in the file LICENSE.
