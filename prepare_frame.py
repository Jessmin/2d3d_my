import cv2
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compare(frame, pre_hist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    n_pixel = frame.shape[0] * frame.shape[1]
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist * (1.0 / n_pixel)
    if pre_hist is not None:
        diff = np.sum(np.abs(np.subtract(hist, pre_hist)))
    else:
        diff = 0
    return diff, hist


def readFrame(config, outputQ: queue):
    threshold = 0.5
    input_path = config.input_file
    max_frame = config.maxFrame
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    _inputCnt = 0
    pre_hist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        diff, hist = compare(frame, pre_hist)
        pre_hist = hist
        is_seg = False
        if diff > threshold:
            is_seg = True
        if max_frame>0 and _inputCnt > max_frame:
            break
        qElem = {
            'index': _inputCnt,
            'decFrm': frame,
            'isSeg': is_seg,
            'total_cnt': total_cnt,
            'fps': fps
        }
        _inputCnt += 1
        # print(f'read frame:{_inputCnt}')
        outputQ.put(qElem)
    # end
    outputQ.put('EOF')
    logger.info('end readFrame')
    cap.release()
