# Scratch Buffers

Some kernels need a scratch buffer. The caller owns it, and each kernel has a pair of functions:

* `esp_nn_get_<op>_scratch_size(...)` returns the bytes one call needs for the given shapes and parameters.
* `esp_nn_set_<op>_scratch_buf(buf)` hands the buffer to the kernel. Call it before the kernel runs.

The pairs are `conv`, `depthwise_conv`, `softmax`, `mean` and `hard_swish`. The batched fully connected call takes its buffer as an argument instead, sized by `esp_nn_get_fc_batch_scratch_size()`.

A typical flow, as in esp-tflite-micro's kernels:

```c
/* At setup: size and allocate. */
int size = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims, &output_dims, &conv_params);
void *scratch = size > 0 ? heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT) : NULL;

/* Per call. */
esp_nn_set_conv_scratch_buf(scratch);
esp_nn_conv_s8(&input_dims, input, &filter_dims, filter, bias, &output_dims, output,
               &conv_params, &quant_data);
```

* The size depends on the target and on the ESP-NN version, as well as on the shapes. A size computed with one version is only valid for that version.
* The set call stores a pointer that the kernels share, so two tasks must not run kernels that use the same buffer at the same time.

## Scratch Sizes on Linux

Tools that compute scratch sizes ahead of time, such as code generators, can build the real getters on Linux instead of copying their arithmetic. The getters for each target are in plain C files with no target code:

| Target | Getter sources |
| --- | --- |
| ESP32-S3 | `src/convolution/esp_nn_scratch_sizes_esp32s3.c` |
| ESP32-P4, ESP32-S31 | `src/convolution/esp_nn_scratch_sizes_riscv_pie.c` |
| Other chips | `src/*/*_opt.c` |

Build them with the macros that `sdkconfig.h` sets on the device, `CONFIG_IDF_TARGET_<TARGET>=1` and `CONFIG_NN_OPTIMIZED=1`, plus the `src/*/*_ansi.c` files. `esp_nn.h` then maps `esp_nn_get_conv_scratch_size()` and the other getters to the target's functions, as on the device.

For example, `sizes.c`:

```c
#include <stdio.h>
#include <esp_nn.h>

int main(void)
{
    data_dims_t input = {.width = 56, .height = 56, .channels = 16};
    data_dims_t filter = {.width = 3, .height = 3, .channels = 16};
    data_dims_t output = {.width = 56, .height = 56, .channels = 72};
    conv_params_t params = {.stride = {1, 1}, .padding = {1, 1}, .dilation = {1, 1}};
    printf("conv scratch: %d bytes\n",
           esp_nn_get_conv_scratch_size(&input, &filter, &output, &params));
    return 0;
}
```

Built for ESP32-P4 from the root of the ESP-NN repository:

```sh
gcc -std=c11 -DCONFIG_IDF_TARGET_ESP32P4=1 -DCONFIG_NN_OPTIMIZED=1 \
    -Iinclude -Isrc/common -Isrc/convolution \
    sizes.c src/convolution/esp_nn_scratch_sizes_riscv_pie.c src/*/*_ansi.c -lm -o sizes
./sizes
```

For ESP32-S3, use `CONFIG_IDF_TARGET_ESP32S3=1` and `esp_nn_scratch_sizes_esp32s3.c`. For other chips, use their `CONFIG_IDF_TARGET_*` macro and `src/*/*_opt.c`.
