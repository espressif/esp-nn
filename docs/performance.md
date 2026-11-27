# ESP-NN Kernel Performance

CPU cycles for one call of each s8 kernel, measured with the benchmark app in `bench_app/`.

- **ANSI C** is the portable reference kernel.
- **Optimized** is the target kernel.
- **Ratio** is ANSI C cycles over optimized cycles.

Setup per chip:

- **ESP32-S3:** 240 MHz, QPI PSRAM at 80 MHz, 64 KB data cache.
- **ESP32-P4:** 360 MHz, HEX PSRAM at 200 MHz, 128 KB L2 cache.
- **ESP32-S31:** 320 MHz, data in internal memory. PIE runs on core 1 only, and ESP-IDF moves
  tasks that use the optimized kernels to core 1.

The "Memory" column says where the tensors were.

## Convolution

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | 4712500 | 331008 | 14.24 | External |
| ESP32-S3 | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | 312754 | 39022 | 8.01 | External |
| ESP32-S3 | input(8,8), filter(64x3x3x3), pad(0,0), stride(1,1) | 2193289 | 394842 | 5.55 | External |
| ESP32-S3 | input(8,8), filter(16x3x3x16), pad(0,0), stride(2,2), SAME | 362880 | 46838 | 7.75 | Internal |
| ESP32-P4 | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | 4041317 | 360029 | 11.22 | External |
| ESP32-P4 | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | 258401 | 50215 | 5.15 | External |
| ESP32-P4 | input(10,10), filter(64x3x3x3), pad(0,0), stride(1,1) | 1803612 | 216972 | 8.31 | External |
| ESP32-S31 | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | 4113419 | 360281 | 11.42 | Internal |
| ESP32-S31 | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | 272046 | 50197 | 5.42 | Internal |
| ESP32-S31 | input(10,10), filter(64x3x3x3), pad(0,0), stride(1,1) | 1798040 | 219440 | 8.19 | Internal |

The ESP32-P4 and ESP32-S31 rows above come from one bench_app run:

- ESP32-P4 revision v3.2, ESP-IDF v6.1. bench_app asks for 360 MHz, which this revision runs at 400 MHz.
- ESP32-S31 revision v0.0 at 320 MHz, ESP-IDF master (v6.2-dev).

## Depthwise Convolution

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(18,18), pad(0,0), stride(1,1), filter: 1x3x3x16 | 1138116 | 86358 | 13.18 | External |
| ESP32-S3 | input(12,12), pad(1,1), stride(1,1), filter: 8x5x5x4 | 1280646 | 260656 | 4.91 | External |
| ESP32-S3 | input (1, 384), pad(0,1), stride(1,1), filter: 1x3x1x128 | 10740507 | 2135842 | 5.03 | Internal |
| ESP32-S3 | input (1, 192), pad(0,1), stride(1,2), filter: 1x3x1x256 | 5359220 | 1071014 | 5.00 | Internal |
| ESP32-P4 | input (16, 16), pad(0,0), stride(1,1) filter: 1x3x3x16 | 962834 | 482389 | 2.00 | External |
| ESP32-P4 | input (12, 12), pad(1,1), stride(1,1)  filter: 8x5x5x4 | 1365066 | 703989 | 1.94 | External |
| ESP32-S31 | out(10,10), pad(1,1), stride(1,1), filter: 1x3x3x16 | 315201 | 118944 | 2.65 | Internal |
| ESP32-S31 | out(12,12), pad(1,1), stride(1,1), filter: 8x5x5x4 | 1325229 | 755116 | 1.75 | Internal |

## Fully Connected

| Chip | Data info | ANSI C | Optimized | Ratio | Cores | Memory |
| ---- | --------- | -----: | --------: | ----: | ----: | ------ |
| ESP32-S3 | len: 271, ch = 3, non-zero input offset | 8448 | 1589 | 5.32 | 1 | Internal |
| ESP32-S3 | len: 384, ch = 512, non-zero input offset | 2007118 | 171671 | 11.69 | 1 | Internal |
| ESP32-S3 | len: 384, ch = 512, non-zero input offset | 2007118 | 106005 | 18.93 | 2 | Internal |
| ESP32-P4 | len: 271, ch = 3, non-zero input offset | 7572 | 1510 | 5.01 | 1 | Internal |
| ESP32-P4 | len: 384, ch = 512, non-zero input offset | 1823280 | 128593 | 14.18 | 1 | Internal |
| ESP32-P4 | len: 384, ch = 512, non-zero input offset | 1823374 | 71370 | 25.55 | 2 | Internal |
| ESP32-S31 | len: 271, ch = 3, non-zero input offset | 7547 | 1475 | 5.11 | 1 | Internal |

### Batched Fully Connected (Weight Zero-Point 0)

Optimized is the 1×1 scratch path, and Cores is how many cores that path uses after `esp_nn_dual_core_enable()`.

| Chip | Data info | ANSI C | Optimized | Ratio | Cores | Memory |
| ---- | --------- | -----: | --------: | ----: | ----: | ------ |
| ESP32-S3 | batch: 16, len: 64, ch: 64 | 753650 | 47009 | 16.03 | 1 | Internal |
| ESP32-S3 | batch: 32, len: 64, ch: 128 | 3014051 | 185091 | 16.28 | 1 | Internal |
| ESP32-S3 | batch: 4, len: 128, ch: 192 | 1056319 | 92467 | 11.42 | 1 | Internal |
| ESP32-S3 | batch: 4, len: 128, ch: 200 | 1100290 | 63290 | 17.38 | 2 | Internal |
| ESP32-S3 | batch: 48, len: 160, ch: 32 | 2607913 | 105845 | 24.64 | 1 | Internal |
| ESP32-S3 | batch: 64, len: 128, ch: 256 | 22544884 | 718330 | 31.39 | 2 | Internal |
| ESP32-P4 | batch: 16, len: 64, ch: 64 | 657470 | 62621 | 10.50 | 2 | Internal |
| ESP32-P4 | batch: 32, len: 64, ch: 128 | 2632680 | 131883 | 19.96 | 2 | Internal |
| ESP32-P4 | batch: 4, len: 128, ch: 192 | 936243 | 61539 | 15.21 | 2 | Internal |
| ESP32-P4 | batch: 4, len: 128, ch: 200 | 975182 | 63822 | 15.28 | 2 | Internal |
| ESP32-P4 | batch: 48, len: 160, ch: 32 | 2318600 | 91881 | 25.23 | 2 | Internal |
| ESP32-P4 | batch: 64, len: 128, ch: 256 | 19977571 | 516458 | 38.68 | 2 | Internal |

## Elementwise Add

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | size = 1615 | 281337 | 74440 | 3.78 | External |
| ESP32-P4 | size = 1615 | 190786 | 88451 | 2.16 | External |
| ESP32-S31 | size = 1615 | 196283 | 75032 | 2.62 | Internal |

## Elementwise Multiply

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | size = 1615 | 117961 | 6783 | 17.39 | External |
| ESP32-P4 | size = 1615 | 72756 | 9413 | 7.73 | External |
| ESP32-S31 | size = 1615 | 72748 | 9260 | 7.86 | Internal |

## Channel Broadcast Multiply

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | 49 positions x 576 channels | 2097974 | 148998 | 14.08 | External |
| ESP32-P4 | 49 positions x 576 channels | 1245229 | 160931 | 7.74 | External |
| ESP32-S31 | 49 positions x 576 channels | 1242538 | 155958 | 7.97 | Internal |

The multiply rows were measured with the ESP32-P4 at 400 MHz.

## Max Pooling

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(16,16), filter(1x3x3x16) | 489650 | 11955 | 40.96 | Internal |
| ESP32-P4 | input(16,16), filter (1x3x3x16) | 482184 | 24178 | 19.94 | Internal |
| ESP32-S31 | input(16,16), filter(1x3x3x16) | 274866 | 21724 | 12.65 | Internal |

## Average Pooling

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(16,16), filter(1x3x3x16) | 427293 | 118052 | 3.62 | Internal |
| ESP32-P4 | input(16,16), filter (1x3x3x16) | 303210 | 84401 | 3.59 | Internal |
| ESP32-S31 | input(16,16), filter(1x3x3x16) | 297200 | 75257 | 3.95 | Internal |

## ReLU6

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | size: 1615 | 1125 | 98 | 11.48 | Internal |
| ESP32-P4 | size, 1615 | 1195 | 154 | 7.76 | Internal |
| ESP32-S31 | size: 1615 | 625 | 123 | 5.08 | Internal |

## Hard Swish

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-P4 | size: 12544 | 703970 | 516582 | 1.36 | External |
| ESP32-S31 | size: 12544 | 620950 | 421618 | 1.47 | Internal |

## Softmax

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | h: 8, w: 32 | 15209 | 11107 | 1.37 | Internal |
| ESP32-P4 | width: 256 | 14260 | 8587 | 1.66 | Internal |
| ESP32-S31 | h: 8, w: 32 | 12090 | 8225 | 1.47 | Internal |

## Mean

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-P4 | 7x7x16 | 10113 | 4686 | 2.16 | Internal |
| ESP32-S31 | 7x7x16 | 10447 | 4399 | 2.37 | Internal |
