cmake_minimum_required(VERSION 3.16)
project(visiontrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# ONNX Runtime — set ONNXRUNTIME_DIR or ORT_ROOT to the install prefix
# e.g. cmake -DONNXRUNTIME_DIR=C:/onnxruntime ...
if(NOT ONNXRUNTIME_DIR AND DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif()

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        "${ONNXRUNTIME_DIR}/include"
        "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session"
        "${ORT_ROOT}/include"
)
find_library(ORT_LIBRARY
    NAMES onnxruntime
    PATHS
        "${ONNXRUNTIME_DIR}/lib"
        "${ONNXRUNTIME_DIR}/build/Linux/Release"
        "${ONNXRUNTIME_DIR}/build/Windows/Release"
        "${ORT_ROOT}/lib"
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set -DONNXRUNTIME_DIR=<path> or $env:ONNXRUNTIME_DIR.\n"
        "Download from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library:  ${ORT_LIBRARY}")

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/preprocessor.cpp
    src/postprocessor.cpp
)

add_executable(visiontrack ${SOURCES})

target_include_directories(visiontrack PRIVATE
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src
)

target_link_libraries(visiontrack PRIVATE
    ${ORT_LIBRARY}
    ${OpenCV_LIBS}
)

# Copy model to build directory for easy access
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/../models/yolo26n.onnx")
if(EXISTS "${MODEL_PATH}")
    add_custom_command(TARGET visiontrack POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_PATH}"
            "$<TARGET_FILE_DIR:visiontrack>/models/yolo26n.onnx"
        COMMENT "Copying ONNX model to build directory"
    )
endif()

# ---------------------------------------------------------------------------
# Tests (optional)
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_test(NAME visiontrack_help COMMAND visiontrack --help)
endif()
