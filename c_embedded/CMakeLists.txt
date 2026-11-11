cmake_minimum_required(VERSION 3.14)
project(F1_Embedded_ECU VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Werror -pedantic -O3)
endif()

include_directories(include)

# Core Shared / Static Library
add_library(f1_ecu SHARED
    src/f1_can_decoder.c
    src/f1_ekf_core.c
)

if(UNIX AND NOT APPLE)
    target_link_libraries(f1_ecu m)
endif()

# Embedded Test Executable
add_executable(test_f1_ecu
    tests/test_f1_ecu.c
)
target_link_libraries(test_f1_ecu f1_ecu)

if(UNIX AND NOT APPLE)
    target_link_libraries(test_f1_ecu m)
endif()

enable_testing()
add_test(NAME ECU_Unit_Tests COMMAND test_f1_ecu)
