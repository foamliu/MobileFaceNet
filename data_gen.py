import os
import pickle

import cv2 as cv
from torch.utils.data import Dataset

from config import IMG_DIR, pickle_file


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        filename = os.path.join(IMG_DIR, filename)
        print(filename)
        img = cv.imread(filename)  # BGR
        img = (img - 127.5) / 128.

        label = sample['label']

        return img, label

    def __len__(self):
        return len(self.samples)
