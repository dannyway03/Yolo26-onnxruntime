from ultralytics import YOLO

# Load YOLO26n
model = YOLO("yolo26n.pt")

# Export to ONNX with 640x480 input
model.export(format="onnx", device="cpu", task="detect",opset=17, simplify=True,imgsz=(480, 640))  # produces yolo26n.onnx

# Load ONNX model
onnx_model = YOLO("yolo26n.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")

# Print results
for box in results[0].boxes.xyxy:
    print(box)

# from ultralytics import YOLO
#
# # Load the YOLO26 model
# model = YOLO("yolo26n.pt")
#
# # Export the model to ONNX format
# model.export(format="onnx",opset=17)  # creates 'yolo26n.onnx'
#
# # Load the exported ONNX model
# onnx_model = YOLO("yolo26n.onnx")
#
# # Run inference
# results = onnx_model("https://ultralytics.com/images/bus.jpg")
#
# print("results ",results)
