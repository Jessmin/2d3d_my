import queue
import numpy as np
import torch
import logging
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resizeFrame(inputQ, outputQ,config):
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        if qElem == 'EOF':
            print('end')
            break
        try:
            calcDepthWidth = 512
            calcDepthHeight = 288
            decFrm = qElem['decFrm']
            calcDepthFrm = cv2.resize(decFrm, (calcDepthWidth, calcDepthHeight))
            calcDepthFrm = cv2.cvtColor(calcDepthFrm,cv2.COLOR_BGR2RGB)
            calcDepthTen = torch.from_numpy(np.asarray(calcDepthFrm))
            qElem['calcDepthTen'] = calcDepthTen
            outputQ.put(qElem)
            # print(f'Resize frame {qElem["index"]}')
        except Exception:
            import traceback
            traceback.print_exc()
    outputQ.put("EOF")
