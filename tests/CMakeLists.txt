
set(COMPONENT_ADD_INCLUDEDIRS ./include/)
set(COMPONENT_SRCS "src/test_utils.c"
                   "src/alignment_test.c"
                   "src/basic_math_test.c"
                   "src/convolution_test.c"
                   "src/fully_connected_test.c"
                   "src/pooling_test.c"
                   "src/relu_test.c"
                   "src/softmax_test.c"
                   "src/hard_swish_test.c"
                   "src/mean_test.c"
                   "src/pie_context_test.c"
                   "src/pie_context_test_esp32p4.S")

set(COMPONENT_REQUIRES )
set(COMPONENT_PRIV_REQUIRES esp-nn)
if(IDF_TARGET STREQUAL "esp32p4")
    # pie_context_test.c: extra interrupt source on the core under test
    list(APPEND COMPONENT_PRIV_REQUIRES esp_driver_gptimer)
endif()

register_component()

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-unused-function)
