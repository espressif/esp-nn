# esp-nn kernel performance

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
| ESP32-P4 | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | 4005512 | 572459 | 7.00 | External |
| ESP32-P4 | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | 249700 | 71104 | 3.51 | External |
| ESP32-P4 | input(10,10), filter(64x3x3x3), pad(0,0), stride(1,1) | 816975 | 533318 | 1.53 | External |
| ESP32-S31 | input(10,10), filter(64x1x1x64), pad(0,0), stride(1,1) | 4107787 | 658148 | 6.24 | Internal |
| ESP32-S31 | input(8,8), filter(16x1x1x16), pad(0,0), stride(1,1) | 277008 | 94210 | 2.94 | Internal |
| ESP32-S31 | input(8,8), filter(64x3x3x3), pad(0,0), stride(1,1) | 1798390 | 765708 | 2.35 | Internal |

## Depthwise convolution

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(18,18), pad(0,0), stride(1,1), filter: 1x3x3x16 | 1159831 | 184176 | 6.30 | External |
| ESP32-S3 | input(12,12), pad(1,1), stride(1,1), filter: 8x5x5x4 | 1671363 | 372435 | 4.49 | External |
| ESP32-S3 | input (1, 384), pad(0,1), stride(1,1), filter: 1x3x1x128 | 10845578 | 2137390 | 5.07 | Internal |
| ESP32-S3 | input (1, 192), pad(0,1), stride(1,2), filter: 1x3x1x256 | 5411399 | 1070931 | 5.05 | Internal |
| ESP32-P4 | input (16, 16), pad(0,0), stride(1,1) filter: 1x3x3x16 | 962834 | 482389 | 2.00 | External |
| ESP32-P4 | input (12, 12), pad(1,1), stride(1,1)  filter: 8x5x5x4 | 1365066 | 703989 | 1.94 | External |
| ESP32-S31 | out(10,10), pad(1,1), stride(1,1), filter: 1x3x3x16 | 315201 | 118944 | 2.65 | Internal |
| ESP32-S31 | out(12,12), pad(1,1), stride(1,1), filter: 8x5x5x4 | 1325229 | 755116 | 1.75 | Internal |

## Fully connected

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | len: 271, ch = 3, non-zero input offset | 8439 | 1543 | 5.47 | Internal |
| ESP32-P4 | len: 271, ch = 3 | 7650 | 915 | 8.36 | Internal |
| ESP32-S31 | len: 271, ch = 3, non-zero input offset | 7547 | 1475 | 5.11 | Internal |

### Batched fully connected (weight zero-point 0)

Times are in microseconds (µs) for one batched call. Before uses the existing ESP-NN batched fully connected implementation: the optimized weight-stationary kernel on ESP32-S3 and the portable ANSI C implementation on ESP32-P4. The P4 also has an optimized single-row PIE kernel, but its existing batched API does not use it. After uses the optimized 1×1 convolution path for batched fully connected operations with weight zero-point 0. ESP32-S3 is revision v0.2 at 240 MHz. ESP32-P4 is revision v1.3 at 360 MHz.

The Cores column indicates the number of cores used by the new 1×1 convolution path with multicore enabled. On ESP32-S3, the second core requires more than 24 KiB of int8 weights and at least 32 output channels. On ESP32-P4, it requires a batch of at least 4 rows. Both chips fall back to single-core execution if the worker cannot be claimed or its scratch buffer is insufficient.

| Chip | Shape | Before (µs) | After (µs) | Gain | Cores |
| ---- | ----- | ----------: | ---------: | ---: | ----: |
| ESP32-S3 | 16×64×64 | 554 | 196 | 2.8× | 1 |
| ESP32-S3 | 32×64×128 | 2149 | 771 | 2.8× | 1 |
| ESP32-S3 | 4×128×192 | 563 | 385 | 1.5× | 1 |
| ESP32-S3 | 4×128×200 | 586 | 283 | 2.1× | 2 |
| ESP32-S3 | 48×160×32 | 993 | 441 | 2.3× | 1 |
| ESP32-S3 | 64×128×256 | 9848 | 2615 | 3.8× | 2 |
| ESP32-P4 | 16×64×64 | 1819 | 159 | 11.4× | 2 |
| ESP32-P4 | 32×64×128 | 7281 | 325 | 22.4× | 2 |
| ESP32-P4 | 4×128×192 | 2593 | 155 | 16.7× | 2 |
| ESP32-P4 | 4×128×200 | 2701 | 161 | 16.8× | 2 |
| ESP32-P4 | 48×160×32 | 6426 | 226 | 28.5× | 2 |
| ESP32-P4 | 64×128×256 | 55326 | 1382 | 40.0× | 2 |

## Elementwise add

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | size = 1615 | 281337 | 74440 | 3.78 | External |
| ESP32-P4 | size = 1615 | 190786 | 88451 | 2.16 | External |
| ESP32-S31 | size = 1615 | 196283 | 75032 | 2.62 | Internal |

## Elementwise multiply

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | size = 1615 | 122703 | 35002 | 3.51 | External |
| ESP32-P4 | size = 1615 | 76585 | 47601 | 1.60 | External |
| ESP32-S31 | size = 1615 | 83337 | 45765 | 1.82 | Internal |

## Max pooling

| Chip | Data info | ANSI C | Optimized | Ratio | Memory |
| ---- | --------- | -----: | --------: | ----: | ------ |
| ESP32-S3 | input(16,16), filter(1x3x3x16) | 376294 | 48069 | 7.83 | Internal |
| ESP32-P4 | input(16,16), filter (1x3x3x16) | 482184 | 24178 | 19.94 | Internal |
| ESP32-S31 | input(16,16), filter(1x3x3x16) | 274866 | 21724 | 12.65 | Internal |

## Average pooling

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

## Hard swish

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
