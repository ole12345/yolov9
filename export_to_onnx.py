import export
export.run(include=['onnx'],weights="./weights/best_striped.pt",imgsz=(640,640),batch_size=1,opset=12)