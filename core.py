import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait as ftWait
import queue
import os
from config import Config
from prepare_frame import readFrame
from resize_frame import resizeFrame
from calc_depth import calcDepth
from inpaint_frame import inpaintFrame
from output_frame import outputFrame
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
        self._executorDepthCalc = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DepthCalc')
        self._executorInpainting = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Inpainting')
        self._executorOutput = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Output')
        self._executorWatcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Watcher')

        self._tasksReadFrame = []
        self._tasksResize = []
        self._tasksInpaintCaption = []
        self._tasksDepthCalc = []
        self._tasksInpainting = []
        self._tasksOutput = []

        self._inEof = False
        self._taskExcep = None

        task = self._executorReadFrame.submit(readFrame, self._readFrameOutputQ, config)
        task.name = f'ReadZip-task-{0}'
        self._tasksReadFrame.append(task)


        task = self._executorResize.submit(resizeFrame, self._readFrameOutputQ, self._resizeOutputQ, config)
        task.name = f'Resize-Task-{0}'
        self._tasksResize.append(task)

        task = self._executorDepthCalc.submit(calcDepth, self._resizeOutputQ, self._calcDepthOutputQ, config)
        task.name = f'DepthCalc-Task-{0}'
        self._tasksDepthCalc.append(task)

        task = self._executorInpainting.submit(inpaintFrame, self._calcDepthOutputQ, self._inpaintOutputQ,
                                               config.inpaint_threads, self.totalFrame)
        task.name = f'Inpainting-Task-{0}'
        self._tasksInpainting.append(task)

        task = self._executorOutput.submit(outputFrame, self._inpaintOutputQ, config)
        task.name = f'Output-Task-{0}'
        self._tasksOutput.append(task)

        self._tasksUnfinished = []
        self._tasksUnfinished += self._tasksReadFrame
        self._tasksUnfinished += self._tasksResize
        self._tasksUnfinished += self._tasksDepthCalc
        self._tasksUnfinished += self._tasksInpainting
        self._tasksUnfinished += self._tasksOutput

        self._watcherTask = self._executorWatcher.submit(self._watchTaskStatus)

