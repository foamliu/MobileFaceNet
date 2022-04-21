"""
The custom dataset is not just face portion so we need to pass through the detector first to detect the faces

* the code will first read the images from the train folder consisting of the images
* second pass the images through the face detector to detect face only
* detected faces will be resized to 112x112x3  and saved in the folder name custom_images
"""

import os
import pdb
import pickle

import cv2
import cv2 as cv
from tqdm import tqdm

from custom_config import IMG_DIR, pickle_file, custom_folder, image_h, image_w
from custom_utils import ensure_folder
from custom_yolodetector import FaceDetector



if __name__ == "__main__":
    face_detector = FaceDetector()
    ensure_folder(IMG_DIR)

    class_folders = [os.path.join(custom_folder, fol) for fol in os.listdir(custom_folder)]
    print(class_folders)

    samples = []

    try:
        i = 0
        for index, class_dir in enumerate(class_folders):
            for imgs in tqdm(os.listdir(class_dir)):
                img_path = os.path.join(class_dir, imgs)
                # img = cv.imread(img_path)
                face = face_detector.detect(img_path)
                if face is not None:
                    img = cv2.resize(face, (image_w, image_h))
                    filename = '{}.jpg'.format(i)
                    destination_path = os.path.join(IMG_DIR, filename)
                    cv.imwrite(destination_path, img)
                    samples.append({'img': filename, 'label': index})
                    i += 1

    except Exception as err:
        print(err)

    with open(pickle_file, 'wb') as file:
        pickle.dump(samples, file)

    print('Total Individual : {}'.format(index+1))
    print('num_samples: ' + str(len(samples)))


