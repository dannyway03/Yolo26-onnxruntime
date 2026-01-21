cmake_minimum_required(VERSION 3.10)
project(yolo26-ORT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use Clang if available
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(CLANGXX clang++)
    if(CLANGXX)
        set(CMAKE_CXX_COMPILER ${CLANGXX})
    endif()
endif()

# --- Build Type ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC -fopenmp -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-anon-enum-enum-conversion")
# Fast math optimizations (avoid -ffast-math to keep infinity/NaN support for OpenCV)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -ftree-vectorize -fno-math-errno -fno-trapping-math -fassociative-math -freciprocal-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")


# Find OpenCV
find_package(OpenCV REQUIRED)

find_package(onnxruntime REQUIRED)
find_package(OpenVINO QUIET)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${OpenVINO_INCLUDE_DIRS}
                    src/)
## Add executable
add_executable(yolo26_demo archive/onnxruntime_yolo26.cpp
               src/instrumentation/global_metrics.cpp
               )

# Link libraries
target_link_libraries(yolo26_demo
                      ${OpenCV_LIBS}
                      onnxruntime::onnxruntime
                      )

add_executable(yoloByte_nhwc_demo
               src/main.cpp
               src/instrumentation/global_metrics.cpp
               )
target_link_libraries(yoloByte_nhwc_demo
                      ${OpenCV_LIBS}
                      onnxruntime::onnxruntime
                      openvino
                      pthread
                      )
