# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS "../" "../tests/")
set(IDF_EXCLUDE_COMPONENTS test test_app)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# -D ESP_NN_TEST_REV_LT_V3_PATHS=1 builds the ESP32-P4 rev < v3 code paths
# into a rev v3 image, which esp-emu runs:
# - hardware loops run in chunks of at most 7 iterations;
# - the kernels do not use the esp.vldbc.8/16.ip increments.
if(ESP_NN_TEST_REV_LT_V3_PATHS)
    idf_build_set_property(COMPILE_DEFINITIONS "ESP_NN_HWLOOP_MAX_COUNT=7" APPEND)
    idf_build_set_property(COMPILE_DEFINITIONS "ESP_NN_VLDBC_IP_BROKEN=1" APPEND)
endif()
project(test_app)
