# MobileFaceNets

![apm](https://img.shields.io/apm/l/vim-mode.svg)

PyTorch implementation of MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices.
[paper](https://arxiv.org/abs/1804.07573).

## Performance

|Accuracy|LFW|MegaFace|Download|
|---|---|---|---|
|paper|99.55%|92.59%||
|ours|99.48%|82.55%|[Link](https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt)|

## Dataset
### Introduction

Refined MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities. 
LFW and Megaface datasets for testing.

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data preprocess
Extract images:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```



