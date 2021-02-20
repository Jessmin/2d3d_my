import cv2
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# filepath = '/home/zhaohoj/Videos/龙门客栈-withSTTN.mp4'
# cap = cv2.VideoCapture(filepath)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# # fourcc = cap.get(cv2.CAP_PROP_FOURCC)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
#
# while True:
#     ret, frame = cap.read()
#     if ret:
#         out.write(frame)
#     else:
#         break
# cap.release()
# out.release()
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


def readFrame(input_path, outputQ: queue):
    threshold = 2
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
            outputQ.put('EOF')
            break
        diff, hist = compare(frame, pre_hist)
        pre_hist = hist
        is_seg = False
        if diff > threshold:
            is_seg = True
        qElem = {
            'index': _inputCnt,
            'decFrm': frame,
            'isSeg': is_seg,
            'total_cnt': total_cnt,
            'fps': fps
        }
        _inputCnt += 1
        outputQ.put(qElem)
    # end
    logger.info('end readFrame')
    cap.release()
