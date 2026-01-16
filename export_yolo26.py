from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to ONNX format
model.export(format="onnx",opset=17)  # creates 'yolo26n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo26n.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")

print("results ",results)