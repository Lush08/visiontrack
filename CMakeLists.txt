cmake_minimum_required(VERSION 3.16)
project(VisionTrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build unit tests" ON)

# Add subdirectory for the main C++ inference engine
add_subdirectory(cpp)
