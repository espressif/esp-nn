# esp-nn kernel performance

CPU cycles for one call of each s8 kernel, measured with the test app in `test_app/`.

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
