cmake_minimum_required(VERSION 3.15)
project(FlashMoeUniversal VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ─── Options ─────────────────────────────────────────────────────────────────
option(USE_METAL   "Use Metal backend (macOS ARM only)"          OFF)
option(USE_OPENCL  "Use OpenCL backend (cross-platform GPU)"     OFF)
option(USE_VULKAN  "Use Vulkan compute backend (Linux/Windows)"  OFF)
option(USE_CUDA    "Use NVIDIA CUDA backend"                     OFF)
option(USE_CPU     "Use CPU backend (always enabled)"            ON)
option(ENABLE_GUI  "Build PyQt5 GUI launcher (Python required)"  ON)
option(ENABLE_SIMD "Enable SIMD acceleration (AVX2/NEON)"        ON)
option(USE_BLAS    "Use BLAS for large matrix operations"        ON)

# ─── Detect architecture & set SIMD flags ────────────────────────────────────
include(CheckCCompilerFlag)

if(ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        check_c_compiler_flag("-mavx2 -mfma" HAS_AVX2)
        if(HAS_AVX2)
            set(SIMD_FLAGS "-mavx2 -mfma")
            add_compile_definitions(HAS_AVX2=1)
            message(STATUS "SIMD: AVX2 + FMA enabled")
        else()
            check_c_compiler_flag("-msse4.1" HAS_SSE41)
            if(HAS_SSE41)
                set(SIMD_FLAGS "-msse4.1")
                add_compile_definitions(HAS_SSE41=1)
                message(STATUS "SIMD: SSE4.1 enabled")
            endif()
        endif()
        check_c_compiler_flag("-mavx512f" HAS_AVX512)
        if(HAS_AVX512)
            option(USE_AVX512 "Enable AVX-512 (Skylake-X and later)" OFF)
            if(USE_AVX512)
                set(SIMD_FLAGS "${SIMD_FLAGS} -mavx512f -mavx512bw -mavx512vl")
                add_compile_definitions(HAS_AVX512=1)
                message(STATUS "SIMD: AVX-512 enabled")
            endif()
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|ARM64|arm64)")
        set(SIMD_FLAGS "-march=armv8-a+simd")
        add_compile_definitions(HAS_NEON=1)
        message(STATUS "SIMD: ARM NEON enabled")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7|armv8)")
        set(SIMD_FLAGS "-mfpu=neon")
        add_compile_definitions(HAS_NEON=1)
        message(STATUS "SIMD: ARM NEON (32-bit) enabled")
    endif()
endif()

# ─── BLAS ─────────────────────────────────────────────────────────────────────
set(BLAS_LIBS "")
if(USE_BLAS)
    if(APPLE)
        set(BLAS_LIBS "-framework Accelerate")
        add_compile_definitions(HAS_BLAS=1 OS_MACOS=1)
        message(STATUS "BLAS: Apple Accelerate Framework")
    else()
        find_package(BLAS)
        if(BLAS_FOUND)
            set(BLAS_LIBS ${BLAS_LIBRARIES})
            add_compile_definitions(HAS_BLAS=1)
            message(STATUS "BLAS: ${BLAS_LIBRARIES}")
        else()
            # Try OpenBLAS manually
            find_library(OPENBLAS_LIB openblas PATHS /usr/lib /usr/local/lib)
            if(OPENBLAS_LIB)
                set(BLAS_LIBS ${OPENBLAS_LIB})
                add_compile_definitions(HAS_BLAS=1)
                message(STATUS "BLAS: OpenBLAS found at ${OPENBLAS_LIB}")
            else()
                message(STATUS "BLAS: not found, using scalar fallback")
            endif()
        endif()
    endif()
endif()

# ─── OS Detection ─────────────────────────────────────────────────────────────
if(APPLE)
    add_compile_definitions(OS_MACOS=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        add_compile_definitions(PLATFORM_APPLE_SILICON=1 ARCH_ARM64=1)
    else()
        add_compile_definitions(ARCH_X86_64=1)
    endif()
elseif(UNIX)
    add_compile_definitions(OS_LINUX=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64)")
        add_compile_definitions(ARCH_X86_64=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        add_compile_definitions(ARCH_ARM64=1)
    endif()
elseif(WIN32)
    add_compile_definitions(OS_WINDOWS=1 _CRT_SECURE_NO_WARNINGS=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        add_compile_definitions(ARCH_X86_64=1)
    endif()
endif()

# ─── Sources ──────────────────────────────────────────────────────────────────
set(CORE_SOURCES
    src/core/inference_engine.c
)

set(CLI_SOURCES
    src/cli/main.c
)

# ─── Compilation flags ────────────────────────────────────────────────────────
set(BASE_FLAGS "-O3 -Wall -Wextra -Wno-unused-parameter")
if(UNIX AND NOT APPLE)
    set(BASE_FLAGS "${BASE_FLAGS} -march=native")
endif()
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${BASE_FLAGS} ${SIMD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${SIMD_FLAGS}")

# ─── Main executable ──────────────────────────────────────────────────────────
add_executable(flash_moe ${CORE_SOURCES} ${CLI_SOURCES})

target_include_directories(flash_moe PRIVATE
    src/
    src/core/
    src/platform/
    src/backends/
)

# ─── Link libraries ───────────────────────────────────────────────────────────
set(LINK_LIBS ${BLAS_LIBS} m)

if(UNIX AND NOT APPLE)
    list(APPEND LINK_LIBS pthread)
endif()

if(WIN32)
    list(APPEND LINK_LIBS ws2_32)
endif()

# ─── Metal backend (macOS only) ───────────────────────────────────────────────
if(USE_METAL AND APPLE)
    target_sources(flash_moe PRIVATE src/backends/metal_backend.m)
    list(APPEND LINK_LIBS "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
    add_compile_definitions(USE_METAL_BACKEND=1)
    message(STATUS "Backend: Apple Metal enabled")
endif()

# ─── OpenCL backend ───────────────────────────────────────────────────────────
if(USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        target_sources(flash_moe PRIVATE src/backends/opencl_backend.c)
        list(APPEND LINK_LIBS ${OpenCL_LIBRARIES})
        target_include_directories(flash_moe PRIVATE ${OpenCL_INCLUDE_DIRS})
        add_compile_definitions(USE_OPENCL_BACKEND=1)
        message(STATUS "Backend: OpenCL enabled (${OpenCL_VERSION_STRING})")
    else()
        message(WARNING "OpenCL not found, skipping OpenCL backend")
    endif()
endif()

# ─── CUDA backend ─────────────────────────────────────────────────────────────
if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        target_sources(flash_moe PRIVATE src/backends/cuda_backend.cu)
        add_compile_definitions(USE_CUDA_BACKEND=1)
        message(STATUS "Backend: CUDA enabled (${CUDA_VERSION})")
    else()
        message(WARNING "CUDA not found, skipping CUDA backend")
    endif()
endif()

# ─── Vulkan backend ───────────────────────────────────────────────────────────
if(USE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        target_sources(flash_moe PRIVATE src/backends/vulkan_backend.c)
        list(APPEND LINK_LIBS ${Vulkan_LIBRARIES})
        target_include_directories(flash_moe PRIVATE ${Vulkan_INCLUDE_DIRS})
        add_compile_definitions(USE_VULKAN_BACKEND=1)
        message(STATUS "Backend: Vulkan enabled")
    else()
        message(WARNING "Vulkan not found, skipping Vulkan backend")
    endif()
endif()

target_link_libraries(flash_moe PRIVATE ${LINK_LIBS})

# ─── MoE benchmark tool ───────────────────────────────────────────────────────
add_executable(flash_moe_bench
    src/core/inference_engine.c
    src/cli/bench.c
)
target_include_directories(flash_moe_bench PRIVATE src/ src/core/ src/platform/ src/backends/)
target_link_libraries(flash_moe_bench PRIVATE ${LINK_LIBS})

# ─── Install targets ─────────────────────────────────────────────────────────
install(TARGETS flash_moe DESTINATION bin)
install(FILES
    scripts/extract_weights.py
    scripts/repack_experts.py
    scripts/export_tokenizer.py
    scripts/download_model.py
    DESTINATION share/flash_moe/scripts)

if(ENABLE_GUI)
    install(DIRECTORY gui/ DESTINATION share/flash_moe/gui)
endif()

# ─── Print summary ────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== Flash-MoE Universal Build Configuration ===")
message(STATUS "  Version     : ${PROJECT_VERSION}")
message(STATUS "  OS          : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Arch        : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C Compiler  : ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C Flags     : ${CMAKE_C_FLAGS}")
message(STATUS "  SIMD        : ${SIMD_FLAGS}")
message(STATUS "  BLAS        : ${BLAS_LIBS}")
message(STATUS "  Metal       : ${USE_METAL}")
message(STATUS "  OpenCL      : ${USE_OPENCL}")
message(STATUS "  CUDA        : ${USE_CUDA}")
message(STATUS "  Vulkan      : ${USE_VULKAN}")
message(STATUS "  GUI         : ${ENABLE_GUI}")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Build with:")
message(STATUS "  mkdir build && cd build && cmake .. && make -j\$(nproc)")
message(STATUS "")
