import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mobilefacenet import MobileFaceNet
import torch
import time

if __name__ == '__main__':
    filename = 'weights/mobilefacenet.pt'
    print('loading {}...'.format(filename))
    start = time.time()
    model = MobileFaceNet()
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    print('elapsed {} sec'.format(time.time() - start))
    print(model)
    output_onnx = 'weights/MobileFaceNet.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)
    export_kwargs = dict(export_params=True, verbose=False, input_names=input_names, output_names=output_names, opset_version=18, dynamo=False,)
    try:
        TrainingMode = torch.onnx.TrainingMode
        export_kwargs['training'] = TrainingMode.EVAL
    except Exception:
        pass
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            **export_kwargs)
    print('Export complete:', output_onnx)
