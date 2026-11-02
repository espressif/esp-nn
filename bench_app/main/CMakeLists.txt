idf_component_register(SRCS "bench_main.c"
                            "bench_conv.c"
                            "bench_depthwise.c"
                            "bench_fully_connected.c"
                            "bench_basic_math.c"
                            "bench_pooling.c"
                            "bench_activations.c"
                            "bench_softmax_mean.c"
                       PRIV_REQUIRES esp-nn bench_harness esp_rom freertos)
