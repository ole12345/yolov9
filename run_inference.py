from detect import run
import os
project = r"my-project"
weights = os.path.join(project,"weights","best_striped.pt")
#source = r"C:\dev\sportimization\yolov9\data\football-players-detection-1\test\images"
source = r"test\images"
img_size = (640,640)
conf_thres = 0.05
iou_thres = 0.45
device = 'cpu'
visualize = True
name = "inference_visualise"
hide_labels=True,  # hide labels
hide_conf=False,  # hide confidences
line_thickness = 1

run(project=project,
    weights=weights,
    source=source,
    imgsz=img_size,
    conf_thres=conf_thres,
    iou_thres=iou_thres,
    device=device,
    visualize=visualize,
    name=name,
    hide_labels=hide_labels,
    hide_conf=hide_conf,
    line_thickness = line_thickness,
    exist_ok=True)