import os
import numpy as np
import torch
from torch import nn

from config import device, grad_clip, print_freq, pretrained_model_path
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
import shutil

import random
seed_value = 5
random.seed(seed_value)

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


class SiameseNetwork(nn.Module):

    def __init__(self, hidden_layer_neurons = 256 ):
        super(SiameseNetwork, self).__init__()
        trained_dir_path = "/home/ahmadob/Documents/adarsh/github/MobileFaceNet/trained_models/"
        select_model = "pretrained_custom_0.1_testset_lr_0.005_epoch30_acc_97.05_lanczos"
        best_checkpoint = 'BEST_checkpoint.tar' # mobile facenet trained backbone
        
        model_path = os.path.join(trained_dir_path, select_model, best_checkpoint)
        checkpoint = torch.load(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
        
        
        self.backbone = checkpoint['model'].module
        for param in self.backbone.parameters():
            param.requires_grad = False
#         self.bn = nn.BatchNorm1d(num_features=128) # 128 because the output of mobilefacenet is 128 dimensional
        # self.ln_backbone = nn.LayerNorm(128) # 128 because the output of mobilefacenet is 128 dimensional
#         self.ln_ffn = nn.LayerNorm(128) # 128 because the output of mobilefacenet is 128 dimensional
        self.fc_classifier = nn.Sequential(
            nn.Linear(128, hidden_layer_neurons), # 1 hidden layer with 32 neurons
            nn.ReLU(),
            nn.Linear(hidden_layer_neurons, 1),
            nn.Sigmoid()
        )
        self.mode = 'train' 
        

    def forward_once(self, x):
        output = self.backbone(x)
        #output = self.ln_backbone(output)
        return output
    
    def forward_train(self, data : dict):
        input1 = data['sample'].to(self.device)
        input2 = data['frame'].to(self.device)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.abs(output1 - output2)
        return output
    
    def forward_test(self, image):
        image = image.to(self.device)
        output2 = self.forward_once(image)
        output = torch.abs(self.sample - output2)
        return output

    
    def forward(self, data):
        if self.mode == 'test':
            output = self.forward_test(data)
        else:
            output= self.forward_train(data)
        
        output = self.fc_classifier(output)
        return output
    
    def init(self, images): # initialization of sample images
        images = images.to(self.device)
        samples = self.forward_once(images)
        if samples.shape[0] == 1:
            self.sample = samples
        else:
            self.sample = samples.mean(axis=0).unsqueeze(0)


def process_image(img):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = data_transforms(img)
    return img

def get_processed_face(pth):
    face = cv2.imread(pth)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = process_image(face).unsqueeze(0)
    return face


def calculate_accuracy(probs, label):
    pred = [0 if i < 0.5 else 1 for i in probs]
    
    acc = accuracy_score(label, pred)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    return [acc, tp, tn, fp, fn]

if __name__ == "__main__":
    classifier = SiameseNetwork().to(device=device)
    print(device)
    # classifier = nn.DataParallel(classifier)
    classifier_dir = "/home/ahmadob/Documents/adarsh/github/MobileFaceNet/classifier_model/WIth_subtraction/custom/no_LN/256_neurons"
    print(classifier_dir)
    classifier_model_path = os.path.join(classifier_dir, "classifier.tar")
    # classifier_model_path=  'classifier_model/classifier.tar'
    classifier.load_state_dict(torch.load(classifier_model_path)['weights'])

    test_pair_file = "data/custom_test_pair.txt"
    # lfw_dir = "data/lfw_funneled_face/"

    with open(test_pair_file, 'r') as file:
        pair_lines = file.readlines()



    lfw_labels = []
    preds = []
    with torch.no_grad():
        classifier.eval()
        for i in tqdm(pair_lines):
            data = {}
            pths = i.split(' ')
            img1 = pths[0]
            img2 = pths[1]
            lfw_labels.append(int(pths[2].strip()))

            data['sample'] = get_processed_face(img1)
            data['frame'] = get_processed_face(img2)
            prob = classifier(data)
            preds.append(prob.item())
    print(calculate_accuracy(preds, lfw_labels))
