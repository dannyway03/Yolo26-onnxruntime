# YOLO26 C++ Implementation

This is a C++ implementation of the `onnxruntime_yolo26.py` script using OpenCV and ONNX Runtime.

## Requirements

- C++17 compatible compiler (MSVC, GCC, or Clang)
- CMake 3.10 or higher
- OpenCV 4.x (pre-built binaries should be in `opencv/build` directory)
- ONNX Runtime 1.23.2 (pre-built binaries should be in `onnxruntime-win-x64-1.23.2` directory)

## Building

### Windows (Visual Studio)

1. Open a command prompt in the project directory
2. Create a build directory:
   ```
   mkdir build
   cd build
   ```
3. Configure with CMake:
   ```
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```
   (Adjust the generator name based on your Visual Studio version)
4. Build the project:
   ```
   cmake --build . --config Release
   ```
   Or open the generated `.sln` file in Visual Studio and build from there.

### Linux/Mac

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Usage

After building, run the executable:

```bash
# Windows
.\build\Release\onnxruntime_yolo26.exe [model_path] [image_path]

# Linux/Mac
./build/onnxruntime_yolo26 [model_path] [image_path]
```

Default arguments:
- Model: `yolo26n.onnx`
- Image: `bus.jpg`

Example:
```bash
.\build\Release\onnxruntime_yolo26.exe yolo26n.onnx bus.jpg
```

The result will be saved as `result.jpg` in the current directory.

## Files

- `onnxruntime_yolo26.cpp` - Main C++ implementation
- `CMakeLists.txt` - CMake build configuration
- `onnxruntime_yolo26.py` - Original Python reference implementation

## Notes

- The implementation closely follows the Python version's logic
- Make sure the ONNX Runtime DLLs are in the same directory as the executable or in your system PATH
- The OpenCV DLLs should also be accessible (they're copied automatically on Windows during build)