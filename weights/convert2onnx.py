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

    # ensure eval mode so BatchNorm uses running stats
    model.eval()

    # dummy input (batch=1)
    dummy_input = torch.randn(1, 3, 112, 112)

    # export kwargs: use modern opset and avoid dynamic_axes/dynamo conflicts
    export_kwargs = dict(
        export_params=True,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        # avoid dynamic_axes to prevent dynamo warnings/failures
        # force legacy path when needed
        dynamo=False,
    )

    # if torch supports TrainingMode, explicitly export in eval mode
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
            **export_kwargs
        )

    print('Export complete:', output_onnx)
