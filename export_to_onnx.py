import export
export.run(include=['onnx'],weights="./runs/gelan-s-seg-small-22_epoch.pt",imgsz=(640,640),batch_size=1,half=True)