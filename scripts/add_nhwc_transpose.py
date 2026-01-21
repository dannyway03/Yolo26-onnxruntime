#!/usr/bin/env python3
"""
Patch YOLOX-Nano ONNX model to accept NHWC input
and insert a Transpose (NHWC->NCHW) as first node.
"""
import onnx
from onnx import helper, TensorProto
import sys

if len(sys.argv) != 3:
    print("Usage: python add_nhwc_transpose.py yolox_nano.onnx yolox_nano_nhwc.onnx")
    sys.exit(1)
input_path = sys.argv[1]
output_path = sys.argv[2]
model = onnx.load(input_path)
graph = model.graph
input_tensor = graph.input[0]
# Modify input shape to NHWC
shape_proto = input_tensor.type.tensor_type.shape
shape_proto.dim[0].dim_value = 1
shape_proto.dim[1].dim_value = 480
shape_proto.dim[2].dim_value = 640
shape_proto.dim[3].dim_value = 3
# Add transpose NHWC->NCHW
transpose_node = helper.make_node(
    "Transpose",
    inputs=[input_tensor.name],
    outputs=["images_nchw"],
    perm=[0, 3, 1, 2],
    name="NHWC_to_NCHW"
)
graph.node.insert(0, transpose_node)
# Rewire downstream nodes
for node in graph.node[1:]:
    for i, inp in enumerate(node.input):
        if inp == input_tensor.name:
            node.input[i] = "images_nchw"
onnx.save(model, output_path)
print(f"Patched ONNX saved to {output_path}")
