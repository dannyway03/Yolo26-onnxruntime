cmake_minimum_required(VERSION 3.10)
project(onnxruntime_yolo26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime paths (adjust these paths according to your installation)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# Add executable
add_executable(onnxruntime_yolo26 onnxruntime_yolo26.cpp)

# Link libraries
target_link_libraries(onnxruntime_yolo26
    ${OpenCV_LIBS}
    onnxruntime
)

