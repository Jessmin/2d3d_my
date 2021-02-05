import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait as ftWait
import queue
import os
from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Conv2d3dCore(object):
    def __init__(self, config: Config):
        maxQueueSize = config.calc_depth_batch_size * 2
        self._readFrameOutputQ = queue.Queue(maxQueueSize)
        self._resizeOutputQ = queue.Queue(maxQueueSize)
        self._calcDepthOutputQ = queue.Queue(maxQueueSize)
        self._inpaintOutputQ = queue.Queue(maxQueueSize)

        self._executorReadFrame = ThreadPoolExecutor(max_workers=1, thread_name_prefix='readFrame')
        self._executorResize = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Resize')
        self._executorInpaintCaption = ThreadPoolExecutor(max_workers=1, thread_name_prefix='InpaintCaption')
        self._executorDepthCalc = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DepthCalc')
        self._executorInpainting = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Inpainting')

