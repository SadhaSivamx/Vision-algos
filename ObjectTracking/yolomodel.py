from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os

model = YOLO("best.onnx", task="detect")

def getroi(frame, conf=0.5):
    results = model.predict(frame, conf=conf, verbose=False)

    for r in results:
        if len(r.boxes) > 0:
            b = r.boxes[0].xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(int, b)
            w = x2 - x1
            h = y2 - y1

            return [x1, y1, w, h]

    return None
