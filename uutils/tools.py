'postprocessing steps borrowed and modified from github.com/hustvl/YOLOS'

import os
import time
import tkinter

import matplotlib.pyplot as plt

from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as U
from torchvision.transforms import functional as F
from transformers import YolosFeatureExtractor, YolosForObjectDetection

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def xywh_to_xyxy(bbox):
    'changes a bbox representation from xywh to xyxy'

    x_c, y_c, w, h = bbox.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    'rescales bboxes from [0,1] to width of image'

    w, h = size
    b = xywh_to_xyxy(out_bbox)
    b = b * torch.tensor([w, h, w, h], dtype=torch.float32)
    return b


def plot_results(img, prob, boxes, *, conf=None):
    'plots bboxes onto an image and saves to'

    tl = 3 # thickness line
    tf = max(tl-1,1) # font thickness
    tempimg = img # copy.deepcopy(img)
    color = [255,0,0] # BGR...
    i = 0 # for counting conf ... cleanup later

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(tempimg, c1, c2, color, tl, cv2.LINE_AA)

        if type(p) is str:
            text = p
            if not conf is None:
                text += f': {round(float(conf[i]), 2)}'
        else: 
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'

        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]

        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(tempimg, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(tempimg, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        i += 1

    return tempimg


def save_pred_fig(img, outputs, keep):
    'todo add docstring'
    # im = cv2.imread(os.path.join(out_dir, "img.png"))

    h, w = img.shape[:2]

    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), (w,h))
    prob = outputs.logits.softmax(-1)[0, :, :-1] # last col is for prob of bbox?
    scores = prob[keep]

    tempimg = plot_results(img, scores, bboxes_scaled)
    fname = os.path.join('.','pred_img.png')
    cv2.imwrite(fname, tempimg)
    print(f"{fname} saved.")


def show_pred_fig(img, outputs, keep):
    'shows predictions in cv2'

    h, w = img.shape[:2]

    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), (w,h))
    prob = outputs.logits.softmax(-1)[0, :, :-1] # last col is for prob of bbox?
    scores = prob[keep]

    tempimg = plot_results(img, scores, bboxes_scaled)
    cv2.imshow('live feed', tempimg)


def do_pred_fig(img, *, bboxes=None, cls=None, conf=None, outputs=None, keep, mode=''):
    'plots predictions in matplotlib'
    'for yolo not yoloS'

    h, w = img.shape[:2]

    if not bboxes is None:
        bboxes_scaled = bboxes
    else:
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), (w,h))

    if not cls is None:
        print(keep)
        scores = [c for c,k in zip(cls, keep) if k]
    else:
        prob = outputs.logits.softmax(-1)[0, :, :-1] # last col is for prob of bbox?
        scores = prob[keep]


    if not conf is None:
        conf = conf[keep]
        print('conf', conf)

    tempimg = plot_results(img, scores, bboxes_scaled, conf=conf)
    plt.imshow(tempimg)
    plt.title('YOLOv5')

    if mode == 'plot':
        plt.show()
    if mode == 'save':
        plt.savefig('img.png')
    if mode == 'show':
        cv2.imshow('live feed', tempimg)
    plt.clf()

def save_gt_fig(output_dir, gt_anno):
    'plot ground truth annotations ... in theory ... poorly developed imo'

    im = cv2.imread(os.path.join(output_dir, "img.png"))
    h, w = im.shape[:2]
    bboxes_scaled = rescale_bboxes(gt_anno['boxes'], (w,h))
    labels = gt_anno['labels']
    plot_gt(im, labels, bboxes_scaled, output_dir)

def loss():
    'define loss function'


def loss_loc(src, tgt, indices, num_boxes):
    'localization loss ... mAP'

    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
    )
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses



def loss_cls():
    'class loss'

