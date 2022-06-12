import time
import tkinter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as U
from torchvision.transforms import functional as F

import uutils as UU

def main():
    print("running...")

    cap = cv2.VideoCapture(0)

    yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    while True:
        ret, frame = cap.read()

        results = yolo([frame])
        results.print()

        output = results.__dict__

        preds = output['xyxy'][0].numpy()

        threshold = 0.5
        conf = preds[:,4]
        keep = conf > threshold
        bboxes = preds[:,:4]
        cls = list(map(lambda x: output['names'][int(x)], preds[:,-1])) 

        UU.tools.do_pred_fig(frame, bboxes=bboxes, cls=cls, conf=conf, keep=keep, mode='show')

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
