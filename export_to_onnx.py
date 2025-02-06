import export
export.run(include=['onnx'],weights="./results/sportimization-35/weights/best_striped.pt",imgsz=(640,640),batch_size=1,dynamic=True,opset=20, half=True)