import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
emb_size = 128

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 21
num_samples = 1119
DATA_DIR = 'data'
# custom_folder = '/home/ahmadob/dataset/facerecognition_dataset/overall_jpeg_data' # for processing overall_dataset
custom_folder = '/home/ahmadob/dataset/facerecognition_dataset/train_set'
IMG_DIR = 'data/images_custom'
pickle_file = 'data/custom_faces_112x112.pickle'
pretrained_model_path = "pretrained_model/mobilefacenet_scripted.pt"
