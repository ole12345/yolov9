from segment.train import run

run(data = "coco-2017-person-ball-small.yaml",
    device = "cpu",
    batch = 32,
    img = 640,
    cfg = "models/segment/gelan-c-s-seg.yaml",
    weights = "",
    name = "test_model",
    hyp = "hyp.scratch-high.yaml",
    close_mosaic = 10,
    epochs = 1
)