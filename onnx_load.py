import numpy

import onnxruntime as rt
#from onnxruntime.datasets import get_example


model = "/home/ole/Documents/dev/anhthu/yolov9_gpl/results/sportimization-35/weights/gelan-c-ball-player-opset12_no_dynamic_250206.onnx"
#model = "/home/ole/Documents/dev/anhthu/yolo/yolov9_clone/runs/gelan-s-seg-small-27_epoch/small2/export/gelan-c-seg-small-27_epoch.onnx"
sess = rt.InferenceSession(model, providers=rt.get_available_providers())


input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_names = sess.get_outputs()
print("number of outputs: ", len(output_names))
for i,output in enumerate(output_names):
    print("output name", output.name)
    print("output shape", output.shape)
    print("output type", output.type)