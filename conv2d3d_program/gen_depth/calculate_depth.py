import os.path as path
import sys
sys.path.append(path.join(path.dirname(__file__), '../..'))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import numpy as np
from queue import Empty, Full
import torch
import torch.multiprocessing as tmp
import torch.autograd as autograd
from models.mannequin_challenge.train_options import TrainOptions
from models.mannequin_challenge import pix2pix_model
from models.midas import midas_net
import torchvision.transforms as transforms
from xzh_mediaunit.ext_tools import process_ndarray_guided_filter
import cv2
from packaging.version import parse as parse_version


def pid_exists(pid):
    if pid < 0:
        return False
    # end

    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # errno.ESRCH
        return False  # No such process
    except PermissionError:  # errno.EPERM
        return True  # Operation not permitted (i.e., process exists)
    else:
        return True
    # end
# end

def _process_depth_to_inpaint(npFrame: np.ndarray, npDepth: np.ndarray, useDilate, useGF, modelName, maxDisparity,
                              depth_mode, depth_scale=10000, mask=None, dispaOffset=14, subtitle_depth=0):
    def do_dilate(np_depth):
        if useDilate is not None:
            kernel, iteration = useDilate
            dilationKernal = np.ones((kernel, kernel), np_depth.dtype)
            np_depth = cv2.dilate(np_depth, dilationKernal, iterations=iteration)
        return np_depth

    def do_gf(np_depth):
        if useGF is not None:
            R, E = useGF
            np_guided_img = npFrame
            h, w = np_depth.shape
            np_guided_img = np.ascontiguousarray(np_guided_img)
            np_guided_img = cv2.resize(np_guided_img, (w, h))
            outGF = np.zeros_like(np_depth, dtype=np_depth.dtype)
            process_ndarray_guided_filter(np_guided_img, np_depth,
                                          outGF, R, E)
            np_depth = outGF
        return np_depth

    if depth_mode == 0:
        npDepth = do_dilate(do_gf(npDepth))
    else:
        npDepth = do_gf(do_dilate(npDepth))

    # if mask is not None:
    #     h, w = mask.shape
    #     npDepth = cv2.resize(npDepth, (w, h))
    #     npDepth[mask == 255] = subtitle_depth

    if modelName == 'midas':
        npDepth = (npDepth / depth_scale * maxDisparity)
    else:
        npDepth = (npDepth / np.exp(npDepth))

    # dispOffset
    dispOffset = 10000 / depth_scale * dispaOffset
    depth_for_inpaint = npDepth - dispOffset

    if depth_for_inpaint.dtype != np.float32:
        depth_for_inpaint = depth_for_inpaint.astype(np.float32)
    return depth_for_inpaint


# end

def _process_batch_input(batchInput, model, maxDisparity, modelName, useDilate,
                         useGF, depth_mode, batchDepthScale, batchMask, dispaOffset, subtitle_depth):
    if modelName == 'midas':
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
    else:
        prediction_d, calc_pred_confidence = model.netG.forward(batchInput)
        prediction_d = prediction_d.detach()
    depth = prediction_d.clone()
    prediction_d = prediction_d.cpu().numpy()
    depth_for_inpaint = []
    for i in range(len(batchInput)):
        # useGF
        npFrame = batchInput[i].cpu().numpy()
        depthScale = batchDepthScale[i]
        mask = batchMask[i]
        npFrame = np.transpose(npFrame, (1, 2, 0))
        _ = _process_depth_to_inpaint(npFrame=npFrame, npDepth=prediction_d[i], useDilate=useDilate,
                                      useGF=useGF,
                                      modelName=modelName, maxDisparity=maxDisparity,
                                      depth_mode=depth_mode, depth_scale=depthScale, mask=mask,
                                      dispaOffset=dispaOffset, subtitle_depth=subtitle_depth)
        depth_for_inpaint.append(_)
    # end
    depth_for_inpaint = np.asarray(depth_for_inpaint)
    return depth, depth_for_inpaint


# end

def _calcDepth_mp(inputQ, outputQ, maxDisparity, parentPid, procName, gpuId, modelName, useFloat16=False,
                  useDilate=None, useGF=None, depth_mode=0, dispaOffset=0, titleDepth=0):
    useDevice = torch.device(f'cuda:{gpuId}')
    torch.cuda.set_device(useDevice)
    logger.info(
        f'Initializing depth estimation model:{modelName} in process \'{procName}\'...'
    )
    if modelName == "midas":
        base_path = path.join(path.dirname(__file__), '../..')
        checkpoint_dir = "models/midas/checkpoints/model-f6b98070.pt"
        checkpoint_dir = os.path.join(base_path, checkpoint_dir)
        _model = midas_net.MidasNet(path=checkpoint_dir, non_negative=True)
        _model = _model.to(useDevice)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        _model.eval()
        if useFloat16:
            _model.half()
        # end
    else:
        opt = TrainOptions().parse(args=[
            "--input=single_view",
            "--checkpoints_dir=models/mannequin_challenge/checkpoints"
        ])
        _model = pix2pix_model.Pix2PixModel(opt, device=useDevice)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        _model.switch_to_eval()
        if useFloat16:
            _model.netG.half()
        # end
    logger.info(f'Model:{modelName} initialized in process \'{procName}\'.')
    inputEof = False
    while not inputEof:
        try:
            qElem = inputQ.get(timeout=0.1)
            if qElem == 'EOF':
                inputEof = True
                break
            # end
        except Empty:
            if not pid_exists(parentPid):
                print(
                    f'!!!!!!!!!! Parent process (pid={parentPid}) does not exist any more. Sub-process \'{procName}\' '
                    f'will exit !!!!!!!!!!! ')
                break
            else:
                continue
            # end
        except Exception as e:
            print(f'Exception! {e}')
            break
        # end
        tenInputs = qElem.pop('batchInput')
        batchDepthScale = qElem.pop('batchDepthScale')
        batchMask = qElem.pop('batchMask')

        depth, depth_for_inpaint = _process_batch_input(
            tenInputs.type(torch.float16 if useFloat16 else torch.float32) /
            255., _model, maxDisparity, modelName, useDilate, useGF, depth_mode, batchDepthScale, batchMask,
            dispaOffset, titleDepth)

        del tenInputs

        qElem['depth_for_inpaint'] = depth_for_inpaint
        qElem['depth'] = depth

        while True:
            try:
                outputQ.put(qElem)
                break
            except Full:
                if not pid_exists(parentPid):
                    print(
                        f'!!!!!!!!!! Parent process (pid={parentPid}) does not exist any more. Sub-process \'{procName}\' will exit !!!!!!!!!!! '
                    )
                    inputEof = True
                    break
                else:
                    continue
                # end
            except Exception as e:
                print(f'Exception! {e}')
                inputEof = True
                break
            # end
        # end
    # end
    inputQ.put('EOF')
    outputQ.put('EOF')


# end


def findMostIdleQueue(queueList):
    if len(queueList) < 1:
        return None
    # end

    leastQsize = queueList[0].qsize()
    if leastQsize == 0:
        return 0
    # end

    idx = 0
    for i in range(1, len(queueList)):
        _ = queueList[i].qsize()
        if leastQsize > _:
            idx = i
            leastQsize = _
            if leastQsize == 0:
                break
            # end
        # end
    # end

    if queueList[idx].full():
        return None
    else:
        return idx
    # end


# end

def calcDepth(inputQ, outputQ, config):
    maxDisparity = config.max_disparity
    dispaOffset = config.disparity_offset
    useFloat16 = config.use_float16
    modelName = config.model_name
    depth_mode = config.depth_mode
    useGF = config.use_gf
    useDilate = config.use_dilate
    batchSize = config.calc_depth_batch_size
    subtitle_depth = config.subtitle_depth
    if batchSize <= 0:
        raise ValueError('Parameter \'batchSize\' must be POSITIVE INTEGER!')
    maxInQueueSize = 2
    maxOutQueueSize = 3
    ipcInQueues = []
    ipcOutQueues = []
    mainProcPid = os.getpid()
    mpCtx = tmp.get_context('spawn')
    subProcs = []
    for idx in range(len(config.use_gpu_ids)):
        gpuId = config.use_gpu_ids[idx]
        try:
            ipcInQueue = mpCtx.Queue(maxInQueueSize)
            ipcOutQueue = mpCtx.Queue(maxOutQueueSize)
            _p = mpCtx.Process(target=_calcDepth_mp,
                               args=(ipcInQueue, ipcOutQueue, maxDisparity,
                                     mainProcPid, f'CalcDepth-SubProc-{gpuId}',
                                     gpuId, modelName, useFloat16, useDilate,
                                     useGF, depth_mode, dispaOffset, subtitle_depth))
            _p.start()
            subProcs.append(_p)
            ipcInQueues.append(ipcInQueue)
            ipcOutQueues.append(ipcOutQueue)
        # end
        except Exception as e:
            print(e)

    subBatchSize = batchSize // len(config.use_gpu_ids)
    if subBatchSize == 0:
        subBatchSize = 1
    # end
    batchInput = []
    batchDepthScale = []
    batchMask = []
    elems = []
    cachedBatches = {}

    # end
    inputEof = False
    subprocEof = False
    gpuIdx = None
    ipcOutQueueEof = [False for _ in range(len(ipcOutQueues))]
    ipcOutQueueEof = np.asarray(ipcOutQueueEof)
    while not (inputEof and subprocEof):
        if not inputEof and len(batchInput) < 1:
            gpuIdx = findMostIdleQueue(ipcInQueues)
            if gpuIdx is not None:
                gpuDevice = torch.device(f'cuda:{config.use_gpu_ids[gpuIdx]}')
            # end
        # end

        if not inputEof and gpuIdx is not None:
            try:
                qElem = inputQ.get(timeout=0.1)
            except Empty:
                qElem = None
            # end
            if qElem == 'EOF':
                inputEof = True
                inputQ.put('EOF')
                qElem = None
            # end
            try:
                if qElem is not None:
                    if 'depth' in qElem and qElem['depth'] is not None:
                        np_depth = qElem['depth']
                        np_frame = qElem['calcDepthTen']
                        depth_scale = qElem['depth_scale']
                        mask = qElem['mask']
                        np_depth_for_inpaint = _process_depth_to_inpaint(np_frame, np_depth, useDilate=useDilate,
                                                                         useGF=useGF, modelName=modelName,
                                                                         maxDisparity=maxDisparity,
                                                                         depth_mode=depth_mode,
                                                                         depth_scale=depth_scale,
                                                                         mask=mask,
                                                                         dispaOffset=dispaOffset,
                                                                         subtitle_depth=subtitle_depth)
                        qElem['depth_for_inpaint'] = np_depth_for_inpaint
                        qElem['depth'] = np_depth
                        outputQ.put(qElem)
                        if config.printStatus:
                            logger.info(f'calculate depth index:{qElem["index"]}')
                        continue
                    # end
                    calcDepthTen = qElem['calcDepthTen']
                    depth_scale = qElem['depth_scale']
                    mask = qElem['mask']
                    frmHeight = calcDepthTen.shape[0]
                    frmWidth = calcDepthTen.shape[1]
                    calcDepthTen = autograd.Variable(calcDepthTen.cuda(device=gpuDevice, non_blocking=True),
                                                     requires_grad=False)
                    ch = calcDepthTen.shape[-1]
                    calcDepthTen = calcDepthTen.view(-1, ch).transpose(1, 0).contiguous().view(ch, frmHeight, frmWidth)
                    batchInput.append(calcDepthTen)
                    batchDepthScale.append(depth_scale)
                    batchMask.append(mask)
                    elems.append(qElem)
                # end

                if len(batchInput) >= subBatchSize or (inputEof and len(batchInput) > 0):
                    firstIndex = elems[0]['index']
                    ipcInQelem = {
                        'index': firstIndex,
                        'batchInput': torch.stack(batchInput),
                        'batchDepthScale': batchDepthScale.copy(),
                        'batchMask': batchMask.copy(),
                    }
                    cachedBatches[firstIndex] = elems.copy()
                    logger.info(f'sending batch to gpu:{gpuIdx} index:{firstIndex}')
                    ipcInQueues[gpuIdx].put(ipcInQelem)
                    batchInput.clear()
                    batchMask.clear()
                    batchDepthScale.clear()
                    elems.clear()
                    gpuIdx = None
                # end
                if inputEof:
                    for ipcInQueue in ipcInQueues:
                        ipcInQueue.put('EOF')
                    # end
                # end
            except Exception:
                import traceback
                traceback.print_exc()
        # end

        gen_output(cachedBatches, ipcOutQueueEof, ipcOutQueues, outputQ)
        # end
        if ipcOutQueueEof.all():
            subprocEof = True
        # end
    # end
    while not subprocEof:
        gen_output(cachedBatches, ipcOutQueueEof, ipcOutQueues, outputQ)
        logger.info('wating subprocEof')
        if ipcOutQueueEof.all():
            subprocEof = True

    logger.info('Wait all calc-depth sub-processes to finish ...')
    for _p in subProcs:
        _p.join()
    # end
    outputQ.put('EOF')
    logger.info('All calc-depth sub-processes are finished.')


def gen_output(cachedBatches, ipcOutQueueEof, ipcOutQueues, outputQ):
    for i in range(len(ipcOutQueues)):
        try:
            ipcOutQelem = ipcOutQueues[i].get(timeout=0.1)
        except Empty:
            continue
        except TimeoutError:
            continue
        # end
        if ipcOutQelem == 'EOF':
            ipcOutQueues[i].put('EOF')
            ipcOutQueueEof[i] = True
            continue
        # end
        firstIndex = ipcOutQelem['index']
        if firstIndex not in cachedBatches:
            print('error')
            raise RuntimeError(
                f'\'firstIndex\'={firstIndex} CANNOT be found in the \'cachedBatches\'!')
        # end
        cachedBatch = cachedBatches.pop(firstIndex)
        depth_for_inpaint = ipcOutQelem.pop('depth_for_inpaint')
        depth = ipcOutQelem.pop('depth')
        np_depth_for_inpaint = depth_for_inpaint
        np_depth = depth.cpu().numpy()
        del depth

        for idx in range(len(cachedBatch)):
            qElem = cachedBatch[idx]
            disparity = np_depth_for_inpaint[idx]
            qElem['depth_for_inpaint'] = disparity
            qElem['depth'] = np_depth[idx]
            outputQ.put(qElem)
            # logger.info(f'calculate depth index:{qElem["index"]}')
        # end
    # end
# end
