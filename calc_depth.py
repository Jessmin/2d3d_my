import os.path as path
import sys
import torch
import queue
import numpy as np
from torchvision import transforms
from packaging.version import parse as parse_version
import torch.autograd as autograd

sys.path.append(path.join(path.dirname(__file__), '../..'))
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

gpuDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batchSize = 1


def _process_batch_input(batchInput, model):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    torch_version = parse_version(torch.__version__)
    if torch_version.major <= 1 and torch_version.minor < 7:
        # before torch version 1.7.0, transform function can only work on single image
        transformedInput = []
        for i in range(batchInput.shape[0]):
            transformedInput.append(transform(batchInput[i]))
        batchInput = torch.stack(transformedInput)
    else:
        # since torch version 1.7.0, transform function can directly work on batch images
        batchInput = transform(batchInput)
    prediction_d = model.forward(batchInput)
    prediction_d = prediction_d.detach()
    depth = prediction_d.clone()
    prediction_d = prediction_d.cpu().numpy()
    depth_for_inpaint = []
    # for i in range(len(batchInput)):
    #     # useGF
    #     npFrame = batchInput[i].cpu().numpy()
    #     npFrame = np.transpose(npFrame, (1, 2, 0))
    #     # _ = _process_depth_to_inpaint(npFrame=npFrame, npDepth=prediction_d[i], useDilate=useDilate,
    #     #                               useGF=useGF,
    #     #                               modelName=modelName, maxDisparity=maxDisparity,
    #     #                               depth_mode=depth_mode, depth_scale=depthScale, mask=mask,
    #     #                               dispaOffset=dispaOffset, subtitle_depth=subtitle_depth)
    #     depth_for_inpaint.append(_)
    depth_for_inpaint = np.asarray(depth_for_inpaint)
    return depth, depth_for_inpaint


def calcDepth(inputQ, outputQ):
    maxDisparity = 40
    dispaOffset = 14
    useFloat16 = False
    batchInput = []
    elems = []
    inputEof = False
    import torch
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model = model.to(gpuDevice)
    model.eval()
    if useFloat16:
        model.half()
    # end
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except queue.Empty:
            qElem = None
        # end
        if qElem == 'EOF':
            inputEof = True
            inputQ.put('EOF')
            qElem = None
        if qElem is not None:
            calcDepthTen = qElem['calcDepthTen']
            calcDepthTen = autograd.Variable(calcDepthTen.cuda(device=gpuDevice, non_blocking=True),
                                             requires_grad=False)
            batchInput.append(calcDepthTen)
            elems.append(qElem)

        if len(batchInput) >= batchSize or (inputEof and len(batchInput) > 0):
            batchInput = batchInput.type(torch.float16 if useFloat16 else torch.float32)
            # calc depth
            depth, depth_for_inpaint = _process_batch_input(batchInput, model)
            for i in range(len(depth)):
                _qElem = elems[i]
                _qElem['depth'] = depth[i]
                _qElem['depth_for_inpaint'] = depth_for_inpaint[i]
                outputQ.put(_qElem)
            # clear
            batchInput.clear()
            elems.clear()
        if inputEof:
            break
    logger.info('All calc Depth Donw')
    outputQ.put("EOF")
