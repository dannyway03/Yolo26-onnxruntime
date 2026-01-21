# ByteYOLOX NHWC + ByteTrack CPU Demo

## Build
mkdir build && cd build
cmake ..
make -j$(nproc)

## Patch ONNX
python ../add_nhwc_transpose.py ../yolox_nano.onnx ../yolox_nano_nhwc.onnx

## Run
./byteyolo_nhwc --mot17 /path/to/MOT17/train

## Environment Variables
export OMP_NUM_THREADS=4
export TBB_NUM_THREADS=2

## Notes
- NHWC input buffer, preallocated and aligned
- IO Binding used, bound once
- Preprocessing writes directly into buffer
- OpenVINO EP fuses Transpose efficiently
- Eliminates HWC->CHW CPU copy bottleneck
