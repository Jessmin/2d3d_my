import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait as ftWait
import queue
from config import Config
from prepare_frame import readFrame
from resize_frame import resizeFrame
from calc_depth import calcDepth
from inpaint_frame import inpaintFrame
# from inpaint_test import inpaintFrame
from output_frame import outputFrame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Conv2d3dCore(object):
    def __init__(self, config: Config):
        maxQueueSize = config.batchSize * 2
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

        task = self._executorReadFrame.submit(readFrame, config, self._readFrameOutputQ)
        task.name = f'ReadZip-task-{0}'
        self._tasksReadFrame.append(task)

        task = self._executorResize.submit(resizeFrame, self._readFrameOutputQ, self._resizeOutputQ, config)
        task.name = f'Resize-Task-{0}'
        self._tasksResize.append(task)

        task = self._executorDepthCalc.submit(calcDepth, self._resizeOutputQ, self._calcDepthOutputQ, config)
        task.name = f'DepthCalc-Task-{0}'
        self._tasksDepthCalc.append(task)

        task = self._executorInpainting.submit(inpaintFrame, self._calcDepthOutputQ, self._inpaintOutputQ)
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

    def _watchTaskStatus(self):
        while len(self._tasksUnfinished) > 0:
            # check if any task raised an exception
            for task in self._tasksUnfinished:
                try:
                    te = task.exception(timeout=0.01)
                    if te is None:
                        # remove this task from the list if the task is finished
                        self._tasksUnfinished.remove(task)
                        # also remove this task from its belonged task group
                        if task in self._tasksReadFrame:
                            self._tasksReadFrame.remove(task)
                            if len(self._tasksReadFrame) == 0:
                                print('All read frame tasks are finished, shutting down resize executor...')
                                self._executorReadFrame.shutdown()
                                print('Read Frame executor is down')
                                self._readFrameOutputQ.put('EOF')
                        if task in self._tasksResize:
                            self._tasksResize.remove(task)
                            if len(self._tasksResize) == 0 and len(self._tasksReadFrame) == 0:
                                print('All resize tasks are finished, shutting down resize executor...')
                                self._executorResize.shutdown()
                                print('Resize executor is down.')
                                self._resizeOutputQ.put('EOF')
                            # end
                        elif task in self._tasksDepthCalc:
                            self._tasksDepthCalc.remove(task)
                            if len(self._tasksDepthCalc) == 0 and len(self._tasksInpainting) == 0:
                                print('All depth-calc tasks are finished, shutting down depth-calc executor...')
                                self._executorDepthCalc.shutdown()
                                print('Depth-calc executor is down.')
                                self._calcDepthOutputQ.put('EOF')
                            # end
                        elif task in self._tasksInpainting:
                            self._tasksInpainting.remove(task)
                            if len(self._tasksInpainting) == 0 and len(self._tasksDepthCalc) == 0:
                                print('All inpainting tasks are finished, shutting down inpainting executor...')
                                self._executorInpainting.shutdown()
                                print('Inpainting executor is down.')
                                self._inpaintOutputQ.put('EOF')
                            # end
                        # end
                        elif task in self._tasksOutput:
                            self._tasksOutput.remove(task)
                            if len(self._tasksOutput) == 0 and len(self._tasksInpainting) == 0:
                                print('All output tasks are finished,shutting down output executor...')
                                self._executorOutput.shutdown()
                                print('Output executor is down.')
                    else:
                        raise te
                    # end
                except TimeoutError:
                    # this is the normal case
                    continue
                except Exception as e:
                    self._taskExcep = e
                    import traceback
                    traceback.print_exc()
                    break
                # end
            # end
            # if exception raised, clear each queue and put 'EOF' to terminate each task
            if self._taskExcep is not None:
                self.stop()
            # end
        # end

    # end

    def stop(self):
        while not self._readFrameOutputQ.empty():
            self._readFrameOutputQ.get()
        self._readFrameOutputQ.put('EOF')
        while not self._resizeOutputQ.empty():
            self._resizeOutputQ.get()
        self._resizeOutputQ.put('EOF')
        while not self._calcDepthOutputQ.empty():
            self._calcDepthOutputQ.get()
        self._calcDepthOutputQ.put('EOF')
        while not self._inpaintOutputQ.empty():
            self._inpaintOutputQ.get()
        self._inpaintOutputQ.put('EOF')

    # end
    def __del__(self):
        self.stop()
        self._executorReadFrame.shutdown()
        self._executorResize.shutdown()
        self._executorDepthCalc.shutdown()
        self._executorInpainting.shutdown()
        self._executorOutput.shutdown()

    def wait(self, timeout=None):
        ftWait([self._watcherTask], timeout=timeout)
