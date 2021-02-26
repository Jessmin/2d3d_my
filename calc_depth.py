import os.path as path
import sys
import torch
import queue
import numpy as np
from torchvision import transforms
from packaging.version import parse as parse_version
import torch.autograd as autograd
from model.midas import midas_net
import cv2

sys.path.append(path.join(path.dirname(__file__), '../..'))
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

gpuDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _process_depth_to_inpaint(npFrame: np.ndarray, npDepth: np.ndarray, maxDisparity,
                              dispOffset=14):
    useDilate = 3, 2
    useGF = 2, 50

    def do_dilate(np_depth):
        kernel, iteration = useDilate
        dilationKernal = np.ones((kernel, kernel), np_depth.dtype)
        np_depth = cv2.dilate(np_depth, dilationKernal, iterations=iteration)
        return np_depth

    def do_gf(np_depth):
        R, E = useGF
        np_guided_img = npFrame
        h, w = np_depth.shape
        np_guided_img = np.ascontiguousarray(np_guided_img)
        np_guided_img = cv2.resize(np_guided_img, (w, h))
        outGF = cv2.ximgproc.guidedFilter(guide=np_guided_img, src=np_depth, radius=R, eps=E, dDepth=-1)
        np_depth = outGF
        return np_depth

    npDepth = do_gf(do_dilate(npDepth))
    npDepth = (npDepth / 10000 * maxDisparity)

    # dispOffset
    depth_for_inpaint = npDepth - dispOffset

    if depth_for_inpaint.dtype != np.float32:
        depth_for_inpaint = depth_for_inpaint.astype(np.float32)
    return depth_for_inpaint


def _process_batch_input(batchInput, model, config):
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
    with torch.no_grad():
        prediction_d = model(batchInput)
    depth = prediction_d.clone()
    prediction_d = prediction_d.cpu().numpy()
    depth_for_inpaint = []

    for i in range(len(batchInput)):
        # useGF
        npFrame = batchInput[i].cpu().numpy()
        npFrame = np.transpose(npFrame, (1, 2, 0))
        _ = _process_depth_to_inpaint(npFrame=npFrame, npDepth=prediction_d[i], maxDisparity=config.maxDisparity,
                                      dispOffset=config.dispOffset)
        depth_for_inpaint.append(_)
    depth_for_inpaint = np.asarray(depth_for_inpaint)
    return depth, depth_for_inpaint


def calcDepth(inputQ, outputQ, config):
    try:
        batchInput = []
        elems = []
        inputEof = False
        # model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        # model = model.to(gpuDevice)
        # model.eval()
        import os
        checkpoint_dir = "/home/zhaohoj/Documents/checkpoint/MidaS/model-f6b98070.pt"
        model = midas_net.MidasNet(path=checkpoint_dir, non_negative=True)
        model = model.to(gpuDevice)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.eval()

        print('load Midas Done ')
        while True:
            try:
                qElem = inputQ.get(timeout=0.1)
            except TimeoutError:
                continue
            except queue.Empty:
                continue
            # end
            if qElem == 'EOF':
                inputEof = True
                qElem = None
            if qElem is not None:
                calcDepthTen = qElem['calcDepthTen']
                calcDepthTen = autograd.Variable(calcDepthTen.cuda(device=gpuDevice, non_blocking=True),
                                                 requires_grad=False)
                ch = calcDepthTen.shape[-1]
                frmHeight = calcDepthTen.shape[0]
                frmWidth = calcDepthTen.shape[1]
                calcDepthTen = calcDepthTen.view(-1, ch).transpose(1, 0).contiguous().view(ch, frmHeight, frmWidth)
                batchInput.append(calcDepthTen)
                elems.append(qElem)

            if len(batchInput) >= config.batchSize or (inputEof and len(batchInput) > 0):
                batchInput_for_calc = torch.stack(batchInput)
                batchInput_for_calc = batchInput_for_calc.type(torch.float32) / 255.
                # calc depth
                depth, depth_for_inpaint = _process_batch_input(batchInput_for_calc, model, config)
                for i in range(len(depth)):
                    _qElem = elems[i]
                    _qElem['depth'] = depth[i]
                    _qElem['depth_for_inpaint'] = depth_for_inpaint[i]
                    outputQ.put(_qElem)
                    # print(f'calc depth:{_qElem["index"]}')
                # clear
                batchInput.clear()
                elems.clear()
            if inputEof:
                break
    except Exception:
        import traceback
        traceback.print_exc()
    print('All calc Depth Donw')
    outputQ.put("EOF")
