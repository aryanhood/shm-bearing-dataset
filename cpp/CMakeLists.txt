cmake_minimum_required(VERSION 3.20)
project(bearingbench_edge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to extracted ONNX Runtime directory containing include/ and lib/")

if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.cache/onnxruntime-1.32.0")
endif()

find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH
)

if(WIN32)
  find_file(ORT_DLL_PATH
    NAMES onnxruntime.dll
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
  )
  find_library(ORT_LIB_PATH
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
  )
  # Clang/Zig targeting Windows GNU links cleanly directly against onnxruntime.dll; MSVC uses onnxruntime.lib.
  if(MSVC)
    set(ORT_LINK_TARGET "${ORT_LIB_PATH}")
  elseif(ORT_DLL_PATH)
    set(ORT_LINK_TARGET "${ORT_DLL_PATH}")
  else()
    set(ORT_LINK_TARGET "${ORT_LIB_PATH}")
  endif()
else()
  find_library(ORT_LINK_TARGET
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
  )
endif()

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LINK_TARGET)
  message(FATAL_ERROR
    "ONNX Runtime not found under ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}'. "
    "Run: python scripts/fetch_onnxruntime.py")
endif()

add_library(bearingbench_edge_lib STATIC
  src/edge_runner.cpp
  src/edge_runner.hpp
)
target_include_directories(bearingbench_edge_lib PUBLIC "${ORT_INCLUDE_DIR}")
target_link_libraries(bearingbench_edge_lib PUBLIC "${ORT_LINK_TARGET}")

if(MSVC)
  target_compile_options(bearingbench_edge_lib PRIVATE /W4 /WX)
else()
  target_compile_options(bearingbench_edge_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(bearingbench-edge src/main.cpp)
target_link_libraries(bearingbench-edge PRIVATE bearingbench_edge_lib)

if(WIN32 AND ORT_DLL_PATH)
  add_custom_command(TARGET bearingbench-edge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_DLL_PATH}"
      "$<TARGET_FILE_DIR:bearingbench-edge>"
    COMMENT "Copying onnxruntime.dll next to bearingbench-edge.exe"
  )
  set(ORT_SHARED_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
  if(EXISTS "${ORT_SHARED_DLL}")
    add_custom_command(TARGET bearingbench-edge POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_SHARED_DLL}"
        "$<TARGET_FILE_DIR:bearingbench-edge>"
    )
  endif()
elseif(UNIX)
  set_target_properties(bearingbench-edge PROPERTIES
    BUILD_RPATH "$ORIGIN;${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN;${ONNXRUNTIME_ROOT}/lib"
  )
endif()
