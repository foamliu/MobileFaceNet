"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import onnx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='pretrained_model/yolov5n-0.5.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[256, 256], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    #print(len(opt.img_size))
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand the input image size if only 1 is given then make it a square otherwise leave it alone
    print(opt)
    set_logging() # It outputs the model information when loaded
    t = time.time()
    
    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model.eval()
    labels = model.names # in labels there is ['face']
    #print("model = {}".format(model))
    #print("model.model = {}".format(model.model))
    # Checks
    gs = int(max(model.stride))  # grid size (max stride), model.stride outputs [8., 16., 32.] so gs will have 32
        
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples
    #print("opt.img_size = {}".format(opt.img_size))  # [256, 256] opt.img_size is same because 256 is divisible by 32
    
    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    #print("img.shape = {}".format(img.shape))    
    
    #print("model.named_modules = {}".format(model.named_modules())) # This is a generator object    
    
    # Update model
    for k, m in model.named_modules():
        # k refers to name of that module, m refers to specific named module
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility, assigning just set to the m
        if isinstance(m, models.common.Conv):  # assign export-friendly activations, checks the instance of modules
            if isinstance(m.act, nn.Hardswish): # checking if the activation module is of Hardswish type, which are all false
                m.act = Hardswish() # convert to Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()    
    #print("model = {}".format(model))
    
    #print("model.model = {}".format(model.model[-1]))

        
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    
    #print(y[0].shape)
    
    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['data'],
                      output_names=['stride_' + str(int(x)) for x in model.stride])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    #"""
    
