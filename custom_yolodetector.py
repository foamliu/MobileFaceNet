import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.quantization
from torchvision import transforms

import copy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

import cv2

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.torch_utils import time_synchronized

import warnings

warnings.filterwarnings("ignore")


class FaceDetector:
    def __init__(self):
        self.file_dir = os.path.dirname(__file__)
        yolov5_model_file = os.path.join(self.file_dir, 'pretrained_model/yolov5n-0.5.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        ################ Face detector loading ########################
        print("Loading FACE DETECTOR")
        self.yolo_model = attempt_load(yolov5_model_file, map_location=self.device)  # load FP32 model
        print("Face detector loaded successfully")

    ################ Target source image face detection ########################
    def detect_path(self, image_path):
        # Load model
        img_size = 256
        conf_thres = 0.4
        iou_thres = 0.5

        orgimg = cv2.imread(image_path, 1)  # BGR
        orgimg = cv2.resize(orgimg, (img_size, img_size))
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.yolo_model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()

        start = time.time()
        pred = self.yolo_model(img)[0]
        # print("Time taken for prediction {} seconds".format(time.time()-start))
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image      
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                return orgimg, np.array(det[:, :4].tolist(), dtype='int')
            else:
                return orgimg, None

    def detect(self, image_path):
        image, bounding_boxes = self.detect_path(image_path)
        images = None
        start = time.time()
        if bounding_boxes is not None:  # Only works when there is face
            # print(bounding_boxes)
            # print("{} faces detected".format(len(bounding_boxes)))
            for box in bounding_boxes[:1]: # only the first bounding box is to be chosen
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                images = image[y1:y2, x1:x2]
        else:
            print(image_path)
            print("Source target image do not have face")
        return images

