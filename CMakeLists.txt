cmake_minimum_required(VERSION 3.16)

project(voxtral_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VOXTRAL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(VOXTRAL_NATIVE_OPT "Enable native CPU tuning flags" ON)
option(VOXTRAL_AUTO_DETECT_CPU "Auto-detect CPU features and set ggml flags" ON)
option(VOXTRAL_AUTO_DETECT_BLAS "Auto-enable ggml BLAS if found" ON)
option(VOXTRAL_AUTO_DETECT_CUDA "Auto-enable ggml CUDA if found" ON)
option(VOXTRAL_AUTO_DETECT_VULKAN "Auto-enable ggml Vulkan if found" ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)

if(VOXTRAL_AUTO_DETECT_CPU AND NOT CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        include(CheckCXXSourceRuns)

        function(voxtral_check_cpu_supports feature outvar)
            set(_src "
                #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
                int main() { return 1; }
                #endif
                #if defined(__GNUC__) || defined(__clang__)
                int main() { return __builtin_cpu_supports(\"${feature}\") ? 0 : 1; }
                #else
                int main() { return 1; }
                #endif
            ")
            check_cxx_source_runs("${_src}" ${outvar})
        endfunction()

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            voxtral_check_cpu_supports("avx"    VOXTRAL_CPU_HAS_AVX)
            voxtral_check_cpu_supports("avx2"   VOXTRAL_CPU_HAS_AVX2)
            voxtral_check_cpu_supports("fma"    VOXTRAL_CPU_HAS_FMA)
            voxtral_check_cpu_supports("f16c"   VOXTRAL_CPU_HAS_F16C)
            voxtral_check_cpu_supports("bmi2"   VOXTRAL_CPU_HAS_BMI2)
            voxtral_check_cpu_supports("avx512f" VOXTRAL_CPU_HAS_AVX512)

            set(VOXTRAL_CPU_FEATURES_DETECTED ON)
        endif()
    endif()
endif()

if(VOXTRAL_CPU_FEATURES_DETECTED)
    # Use explicit ISA flags instead of -march=native when auto-detecting.
    set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize the build for the current system" FORCE)
    set(GGML_AVX    ${VOXTRAL_CPU_HAS_AVX}    CACHE BOOL "ggml: enable AVX" FORCE)
    set(GGML_AVX2   ${VOXTRAL_CPU_HAS_AVX2}   CACHE BOOL "ggml: enable AVX2" FORCE)
    set(GGML_FMA    ${VOXTRAL_CPU_HAS_FMA}    CACHE BOOL "ggml: enable FMA" FORCE)
    set(GGML_F16C   ${VOXTRAL_CPU_HAS_F16C}   CACHE BOOL "ggml: enable F16C" FORCE)
    set(GGML_BMI2   ${VOXTRAL_CPU_HAS_BMI2}   CACHE BOOL "ggml: enable BMI2" FORCE)
    set(GGML_AVX512 ${VOXTRAL_CPU_HAS_AVX512} CACHE BOOL "ggml: enable AVX512F" FORCE)
endif()

if(VOXTRAL_AUTO_DETECT_BLAS)
    if(NOT DEFINED CACHE{GGML_BLAS})
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(GGML_BLAS ON CACHE BOOL "ggml: use BLAS" FORCE)
        endif()
    endif()
endif()

if(VOXTRAL_AUTO_DETECT_CUDA)
    if(NOT DEFINED CACHE{GGML_CUDA})
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            include(CheckLanguage)
            check_language(CUDA)
            if(CMAKE_CUDA_COMPILER)
                set(GGML_CUDA ON CACHE BOOL "ggml: use CUDA" FORCE)
            endif()
        endif()
    endif()
endif()

if(VOXTRAL_AUTO_DETECT_VULKAN)
    if(NOT DEFINED CACHE{GGML_VULKAN})
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            set(GGML_VULKAN ON CACHE BOOL "ggml: use Vulkan" FORCE)
        endif()
    endif()
endif()

add_subdirectory(ggml)

add_library(voxtral_lib
    src/voxtral.cpp)

# On Windows with BUILD_SHARED_LIBS=ON, export symbols automatically so the
# import library (voxtral_lib.lib) is generated for dependents like `voxtral`.
if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(voxtral_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(voxtral_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(voxtral_lib
    PUBLIC
        ggml
        Threads::Threads)

if(NOT WIN32)
    target_link_libraries(voxtral_lib PUBLIC m)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(voxtral_lib PUBLIC ${CMAKE_DL_LIBS})
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral_lib PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    if(APPLE)
        message(STATUS "VOXTRAL_NATIVE_OPT: skipping -march/-mtune=native on Apple toolchains")
    else()
        target_compile_options(voxtral_lib PRIVATE -march=native -mtune=native)
    endif()
endif()

add_executable(voxtral src/main.cpp)
target_link_libraries(voxtral PRIVATE voxtral_lib)

# ── voxtral-server (HTTP transcription server) ──────────────────────
option(VOXTRAL_BUILD_SERVER "Build voxtral-server HTTP executable" ON)

if(VOXTRAL_BUILD_SERVER)
    include(FetchContent)
    FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG        v0.20.0
        GIT_SHALLOW    TRUE
    )
    set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(httplib)

    add_executable(voxtral-server src/server.cpp)
    target_link_libraries(voxtral-server PRIVATE voxtral_lib httplib::httplib)

    if(VOXTRAL_WARNINGS_AS_ERRORS)
        if(MSVC)
            target_compile_options(voxtral-server PRIVATE /W4 /WX)
        else()
            target_compile_options(voxtral-server PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
    endif()

    if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
        if(NOT APPLE)
            target_compile_options(voxtral-server PRIVATE -march=native -mtune=native)
        endif()
    endif()
endif()

add_executable(voxtral-quantize src/voxtral-quantize.cpp)
target_link_libraries(voxtral-quantize PRIVATE ggml Threads::Threads)

if(NOT WIN32)
    target_link_libraries(voxtral-quantize PRIVATE m)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(voxtral-quantize PRIVATE ${CMAKE_DL_LIBS})
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral-quantize PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral-quantize PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    if(NOT APPLE)
        target_compile_options(voxtral-quantize PRIVATE -march=native -mtune=native)
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_custom_target(voxtral_convert
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_voxtral_to_gguf.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print converter usage"
        VERBATIM)
    add_custom_target(voxtral_quantize_all
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/quantize_all.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print quantize_all usage"
        VERBATIM)
endif()

enable_testing()
if(Python3_Interpreter_FOUND)
    add_test(
        NAME voxtral_python_reference
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_voxtral_reference.py)
endif()

#add_subdirectory(examples/stream)
