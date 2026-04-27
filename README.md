# ESP-NN

The library contains optimised NN (Neural Network) functions for various Espressif chips.

* Supported platforms:
   * TensorFlow Lite Micro (TFLite Micro). Repo can be found [here](https://github.com/espressif/tflite-micro-esp-examples)

* Supported ESP chips include:
   * ESP32-S3 (Assembly versions optimised to benefit from vector instructions of ESP32-S3)
   * ESP32-P4 (Optimised using PIE/QACC SIMD instructions)
   * ESP32 (Generic optimisations)
   * ESP32-C3 (Generic optimisations)

## Performance

### Kernelwise performance for s8 versions:

  * Kernelwise performance on ESP32-P4 chip
    * Numbers are ticks taken for kernel to execute
    * Chip config: 360MHz, SPI-RAM: HEX 200MHz, L2-Cache: 128KB

    | Function        | ANSI C  | Optimized | Opt Ratio | Data info   | Memory    |
    | ----------------| --------|---------|---------|-------------|-----------|
    | elementwise_add | 190786  | 88451   | 2.16    | size = 1615 | External  |
    | elementwise_mul | 76585   | 47601   | 1.60    | size = 1615 | External  |
    | convolution     | 4005512 | 572459  | 7.00    | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | External |
    | convolution     | 249700  | 71104   | 3.51    | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | External |
    | convolution     | 816975  | 533318  | 1.53    | input(10,10), filter(64x3x3x3), pad(0,0), stride(1,1) | External |
    | depthwise conv  | 962834  | 482389  | 2.00    | input (16, 16), pad(0,0), stride(1,1) filter: 1x3x3x16 | External |
    | depthwise conv  | 1365066 | 703989  | 1.94    | input (12, 12), pad(1,1), stride(1,1)  filter: 8x5x5x4 | External |
    | max pool        | 482184  | 24178   | 19.94   | input(16,16), filter (1x3x3x16) | Internal |
    | avg pool        | 303210  | 84401   | 3.59    | input(16,16), filter (1x3x3x16) | Internal |
    | fully connected | 7650    | 915     | 8.36    | len: 271, ch = 3 | Internal |
    | prelu (relu6)   | 1195    | 154     | 7.76    | size, 1615  | Internal  |
    | softmax         | 14260   | 8587    | 1.66    | width: 256  | Internal  |
    | hard_swish      | 703970  | 516582  | 1.36    | size: 12544 | External  |
    | mean            | 10113   | 4686    | 2.16    | 7x7x16     | Internal  |


  * Kernelwise performance on ESP32-S3 chip
    * Numbers are ticks taken for kernel to execute
    * Chip config: 240MHz, SPI: QPI 80MHz, Data cache: 64KB

    | Function        | ANSI C   | Optimized | Opt Ratio | Data info   | Memory    |
    | ----------------| ---------|-----------|-----------|-------------|-----------|
    | elementwise_add | 281337   | 74440     | 3.78      | size = 1615 | External  |
    | elementwise_mul | 122703   | 35002     | 3.51      | size = 1615 | External  |
    | convolution     | 4712500  | 331008    | 14.24     | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | External |
    | convolution     | 312754   | 39022     | 8.01      | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | External |
    | convolution     | 2193289  | 394842    | 5.55      | input(8,8), filter(64x3x3x3), pad(0,0), stride(1,1) | External |
    | depthwise conv  | 1159831  | 184176    | 6.30      | input(18,18), pad(0,0), stride(1,1), filter: 1x3x3x16 | External |
    | depthwise conv  | 1671363  | 372435    | 4.49      | input(12,12), pad(1,1), stride(1,1), filter: 8x5x5x4 | External |
    | max pool        | 376294   | 48069     | 7.83      | input(16,16), filter(1x3x3x16) | Internal |
    | avg pool        | 427293   | 118052    | 3.62      | input(16,16), filter(1x3x3x16) | Internal |
    | fully connected | 8443     | 1078      | 7.83      | len: 271, ch = 3 | Internal |
    | softmax         | 15209    | 11107     | 1.37      | h: 8, w: 32 | Internal  |
    | prelu (relu6)   | 1125     | 98        | 11.48     | size: 1615  | Internal  |


### Model-level performance:

  * **Person Detection** (Visual Wake Words, INT8 quantized — from [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro))
    * Numbers are time (ms) for `invoke()` call, using internal memory

    | Chip     | CPU Freq | without ESP-NN | with ESP-NN |
    | -------- | -------- | -------------- | ----------- |
    | ESP32-P4 | 360MHz   | 1395ms         | 73ms        |
    | ESP32-S3 | 240MHz   | 2300ms         | 54ms        |
    | ESP32    | 240MHz   | 4084ms         | 380ms       |
    | ESP32-C3 | 160MHz   | 3355ms         | 426ms       |

  * **MobileNetV3 Small** (INT8 quantized, 224x224x3, 1000 classes)

    | Chip     | CPU Freq | without ESP-NN | with ESP-NN |
    | -------- | -------- | -------------- | ----------- |
    | ESP32-S3 | 240MHz   | 26000ms        | 1434ms      |
    | ESP32-P4 | 360MHz   | 11600ms        | 1050ms      |

> **Note**:
  - The above is time taken for execution of the `invoke()` call
  - SPIRAM used for TensorArena.
  - Person detection on ESP32-S3 with internal RAM: 47ms
  - ESP32-P4 optimisation is work in progress
  - `Without ESP-NN` case is when `esp-nn` is completely disabled by removing below flag from [CMakeLists.txt](CMakeLists.txt):
    ```cmake
      # enable ESP-NN optimizations by Espressif
      target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
    ```


## Configuration

  * To configure, please use `idf.py menuconfig` and under `ESP-NN` select `NN_OPTIMIZATIONS`
  * There are two options presented:
     * Optimized versions
     * ANSI C

  * Default selection is for `Optimized versions`. For ESP32-S3 and ESP32-P4, assembly versions are automatically selected, whereas for other chips (viz., ESP32, ESP32-C3), generic optimisations are selected.
  * For debugging purposes, you may want to select `ANSI C` reference versions.


## Contributing

If you encounter an issue with ESP-NN, or wish to submit a feature request, please use the Issues section on the Github.

For general questions related to this library, please use the esp32.com forum.

Please check [CONTRIBUTING.md](CONTRIBUTING.md) for further information if you'd like to contribute to ESP-NN.

## Copyrights and License

All original source code in this repository is Copyright (C) 2020-2021 Espressif Systems. This source code is licensed under the Apache License 2.0 as described in the file LICENSE.
