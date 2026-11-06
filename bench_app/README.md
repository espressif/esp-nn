# bench_app: esp-nn kernel cycles

Measures how many CPU cycles each esp-nn kernel takes, against its ANSI C reference, on real
hardware. `test_app` checks correctness only; the cycle numbers come from here.

## Build and run

```bash
cd bench_app
idf.py set-target esp32s3        # or esp32p4, any esp-nn target
idf.py -p PORT flash monitor
```

ESP32-S31 is a preview target on IDF master: `idf.py --preview set-target esp32s31`.

The run takes a few seconds. Per case and kernel variant it:
- computes the expected output of two input sets with the ANSI reference;
- times the reference and the variant, 9 calls each with the input sets alternating, and keeps
  the fastest call after the first;
- checks every output against the reference, and the guard bytes around the output.

Timing runs on one core, with interrupts off during each call and esp-nn's dual-core worker off.

## Output

```
BENCH_START target=esp32s3 idf=v5.5 cpu_mhz=240 psram=1 iters=9 seed=0xc0ffee42
BENCH op=conv variant=esp_nn mem=ext ansi=4740697 opt=302647 ratio=15.66 result=ok shape="input 10x10x64, filter 64x1x1x64, pad 0, stride 1"
...
BENCH_DONE cases=40 failures=0 skipped=0
```

- `ansi`, `opt`: cycles for one call of the reference and of the variant.
- `ratio`: `ansi / opt`.
- `mem`: where the tensors were: `ext` for PSRAM, `int` for internal RAM. A case asks for PSRAM
  and falls back to internal RAM on boards without it.
  Each case runs the same inputs several times and keeps the fastest, so tensors that fit the data
  cache are timed with a warm cache, even when they live in PSRAM.
- `result`: `ok`, or `MISMATCH` when an output or a guard byte differs. The run continues.
  `skip` means the case could not allocate its buffers. `BENCH_DONE` counts these as `skipped`.
- `BENCH_WARN`: a variant slower than its limit relative to another variant of the same case.
  - Fully connected: `esp_nn` against the same call with operands 8 bytes past a 16-byte
    boundary. It guards against esp-nn#36.
  - Batched fully connected: `esp_nn_scratch` (the 1x1 convolution path, when its gate accepts
    the shape) against `esp_nn`. The gate should pick the 1x1 path only where it wins.

Each case seeds its data from the run seed, its op and its shape. A case gets the same data
however many cases run before it.

The published kernel performance tables are made from these lines.

## Adding a case or a variant

Each op has its own file in `main/`, for example `bench_fully_connected.c`.

- **A new shape:** add a row to the op's shape table. Nothing else changes.
- **Another way to run the same op**, for example a contributed kernel: write a function with
  the `bench_fn` signature that makes one call, and add it to the op's variants table:

  ```c
  static void fc_batch_contrib(void *p, int v, int8_t *out)
  {
      const fc_batch_ctx_t *c = p;
      /* one call computing the same output as the reference */
  }

  static const bench_variant_t s_batch_variants[] = {
      { "esp_nn", fc_batch_opt },
      { "esp_nn_scratch", fc_batch_scratch },
      { "contrib", fc_batch_contrib },
  };
  ```

  Every variant is timed and checked against the same reference, and prints its own `BENCH`
  line next to the others.
- **A new op:** add a file with a shape table, a context struct, the reference call and a
  variants table, then call it from `bench_main.c`. `components/bench_harness/include/bench.h`
  describes `bench_run()`.
