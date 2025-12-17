# MobileFaceNet to ONNX Conversion

Used in a local facial-recognition deployment pipeline for edge inference.  
This script converts a trained MobileFaceNet PyTorch model to ONNX for use with ONNX Runtime.

---

## Overview

This script loads a trained MobileFaceNet model in PyTorch and converts it to ONNX format for inference outside PyTorch. It keeps the process minimal and reproducible. It starts by adding the parent directory to the Python path so that the local `mobilefacenet` module can be imported. After that, it loads the model weights from `weights/mobilefacenet.pt` onto the CPU and measures how long the loading process takes.

Once the model is loaded, it is switched to evaluation mode and a dummy input tensor of shape `(1, 3, 112, 112)` is created to represent a standard image input. The script then defines export parameters such as input and output tensor names, uses ONNX opset version 18, and ensures that training mode is disabled. It wraps the export process inside a `torch.no_grad()` block to prevent gradient tracking and calls `torch.onnx.export` to create the onnx file at `weights/MobileFaceNet.onnx`.

After the export finishes, the script prints a confirmation message showing the output path. The purpose of the code is straightforward: load the trained model, prepare it for export, and convert it to a portable format suitable for inference in environments where PyTorch is not required.

---

## Summary

This script is a simple and direct example of bridging a research-grade PyTorch model to a deployable ONNX format. It reflects practical understanding of model preparation, export configuration, and deployment consistency.

## small note
I have put my .pt file in as well, so when runnign the onnx conversion file, just be careful and remember to delete all the extra hush first. 
that's all, from my side

p.s. i read through the license and through the apache one i'm sure its legal to post my conversion script and the onnx model. happy to contribute to os

open to feedback through issues or yk, my social handle details are there for your disposal

peace :)
