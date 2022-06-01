"""
This code creates folder and images with cropped faces using yoloV5 detector, basically, the test_pair of LFW is used
to read the files required and then each of those images are passed through detector and lastly the image are saved in
the corresponding directory

"""

import numpy as np
import os
import cv2 as cv
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

from custom_config import device, image_w, image_h
from data_gen import data_transforms
from custom_utils import get_face_yolo
transformer = data_transforms['val']

filename = 'data/lfw_test_pair.txt'
lfw_dir = "data/lfw_funneled"
lfw_face_dir = "data/lfw_funneled_face"
with open(filename, 'r') as file:
    pair_lines = file.readlines()

lfw_path = []
for i in pair_lines:
    pths = i.split(' ')
    lfw_path.append(os.path.join(lfw_dir, pths[0]))
    lfw_path.append(os.path.join(lfw_dir, pths[1]))
lfw_path = list(set(lfw_path))

in_valid = 0
for pth in tqdm(lfw_path):
    face_path = pth
    dest_dir = os.path.join(lfw_face_dir, face_path.split("/")[-2])
    #print(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    is_valid, face = get_face_yolo(face_path)
    if is_valid:
        face = cv.resize(face, (image_w, image_h), interpolation=cv.INTER_LANCZOS4)
        dest_path = os.path.join(dest_dir, face_path.split("/")[-1])
        #print(dest_path)
        cv.imwrite(dest_path, face)
    else:
        in_valid+=1
        print("Not detectable face {}".format(face_path.split("/")[-2:]))
print("Total invalid faces = {}".format(in_valid))
