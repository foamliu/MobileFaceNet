import numpy as np
import os
import cv2 as cv
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from tqdm import tqdm
from PIL import Image

from custom_config import device, image_w, image_h
from data_gen import data_transforms
from custom_utils import get_face_yolo
transformer = data_transforms['val']

def get_path_and_labels(dir):
    unique_targets = os.listdir(dir)
    print("Unique targets in the directory : {}".format(len(unique_targets)))

    images_path = []
    labels = []
    for trg in unique_targets:
        dir_path = os.path.join(dir, trg)
        for images in os.listdir(dir_path):
            image_path = os.path.join(dir_path, images)
            images_path.append(image_path)
            labels.append(trg)

    print("Total images in {} : {}".format(dir, len(images_path)))
    print("Targets {} and \n counts = {}".format(Counter(labels).keys(), Counter(labels).values()))

    return images_path, labels

def get_faces_and_labels(face_dir):
    unique_targets = os.listdir(face_dir)
    print("Unique faces in the directory : {}".format(len(unique_targets)))

    faces = []
    labels = []
    for trg in unique_targets:
        dir_path = os.path.join(face_dir, trg)
        for face in os.listdir(dir_path):
            face_path = os.path.join(dir_path, face)
            faces.append(cv.imread(face_path))
            labels.append(trg)

    print("Total faces in {} : {}".format(dir, len(faces)))
    print("Targets {} and \n counts = {}".format(Counter(labels).keys(), Counter(labels).values()))

    return faces, labels

def get_faces(paths):
    print('Detecting faces...')
    faces = []
    valid = []
    for filename in tqdm(paths):
        is_valid, face = get_face_yolo(filename)

        if is_valid:
            face = cv.resize(face, (image_w, image_h), interpolation=cv.INTER_LANCZOS4)
            faces.append(face)
            valid.append(1)
        else:
            valid.append(0)
    return faces, valid

def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img

def get_feature(model, face):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    img = face
    imgs[0] = transform(img.copy(), False)
    imgs[1] = transform(img.copy(), True)
    # pdb.set_trace()
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)


def get_face_vectors(model, faces):
    face_vectors = []
    for face in faces:

        face_vectors.append(get_feature(model, face))

    return face_vectors

def get_model(model_path = "trained_models/pretrained_custom_0.1_testset_lr_0.005_epoch30_acc_97.05_lanczos/BEST_checkpoint.tar"):
    checkpoint = model_path
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    return model


def generate_plot(X, y, kind):
    df = pd.DataFrame()
    df["y"] = y
    z = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(Counter(y).keys())),
                    data=df).set(title="{} data T-SNE projection".format(kind))
    plt.show()

def save_faces(kind, images_path, faces):
    root_dir = "/home/ahmadob/dataset/facerecognition_dataset/"
    if kind == "train_set":
        data_type = "train_face_set"
    if kind == "test_set":
        data_type = "test_face_set"

    data_path = os.path.join(root_dir, data_type)
    for index, data in enumerate(images_path):
        target = data.split('/')[-2]
        destination_path = os.path.join(data_path, target)
        os.makedirs(destination_path, exist_ok=True)
        filepath = os.path.join(destination_path, str(index)+".jpg")
        cv.imwrite(filepath, faces[index])

    print("Saved Faces")

if __name__ == "__main__":
    trained_model_dir = "/home/ahmadob/Documents/adarsh/github/MobileFaceNet/trained_models/"
    model_dir = "pretrained_custom_0.1_testset_lr_0.001_epoch70_acc_93.09"
    model_path = os.path.join(trained_model_dir, model_dir, "BEST_checkpoint.tar")
    root = "/home/ahmadob/dataset/facerecognition_dataset/"
    kind = 'test_set'
    dir = os.path.join(root, kind)

    if kind == "train_set":
        data_type = "train_face_set"
    if kind == "test_set":
        data_type = "test_face_set"

    faces_path = os.path.join(root, data_type)
    print(faces_path)
    if os.path.exists(faces_path):
        # no need to run face detector
        print("Faces exist, no need to run detector")
        faces, labels = get_faces_and_labels(faces_path)
    else:
        images_path, labels = get_path_and_labels(dir)
        print("Getting face from images")
        faces, valid = get_faces(images_path)
        if sum(valid) == len(labels):
            print("Images are fine, no need to process labels")
        else:
            print("Need to remove the labels not containing face")
            # make function to do so

        save_faces(kind, images_path, faces)

    print("Loading Model")

    model = get_model(model_path)
    print("Getting face feature vectors")
    face_vectors = get_face_vectors(model, faces)
    generate_plot(np.array(face_vectors), labels, kind)




