from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait as ftWait
import queue
import os
from calculate_depth import calcDepth
from config import Config

from prepare_frame import readFrame
from resize_frame import resizeFrame
from inpaint_frame import inpaintFrame
from output_frame import outputFrame
from inpaint_caption import inpaint_caption

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Conv2d3dCore(object):
    def __init__(self, config: Config):
        if os.path.isfile(config.depthInfo):
            os.remove(config.depthInfo)
        # end
        self.use_gf = config.use_gf
        self.use_dilate = config.use_dilate

        max_frame = 0
        if config.start_frame != 0 or config.end_frame != 0:
            max_frame = (config.end_frame - config.start_frame) // 40
        self.totalFrame = max_frame if max_frame != 0 else self.get_file_count(config)
        logger.info(f'total Frame :{self.totalFrame}')

        self.totalFrame = self.get_file_count(config) if max_frame == 0 else max_frame
        maxQueueSize = config.calc_depth_batch_size * 2

        self._inputQ = queue.Queue(maxQueueSize)
        self._resizeOutputQ = queue.Queue(maxQueueSize)
        self._captionOutputQ = queue.Queue(maxQueueSize)
        self._calcDepthOutputQ = queue.Queue(maxQueueSize)
        self._inpaintOutputQ = queue.Queue(maxQueueSize)
        self._outputCnt = 0
        self._inputCnt = 0
        self.output_mono_width = config.output_mono_width

        self._executorReadFrame = ThreadPoolExecutor(max_workers=1, thread_name_prefix='readFrame')
        self._executorResize = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Resize')
        self._executorInpaintCaption = ThreadPoolExecutor(max_workers=1, thread_name_prefix='InpaintCaption')
        self._executorDepthCalc = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DepthCalc')
        self._executorInpainting = ThreadPoolExecutor(max_workers=config.inpaint_threads,
                                                      thread_name_prefix='Inpainting')
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

        task = self._executorReadFrame.submit(readFrame, self._inputQ, config)
        task.name = f'ReadZip-task-{0}'
        self._tasksReadFrame.append(task)

        task = self._executorInpaintCaption.submit(inpaint_caption, self._inputQ, self._captionOutputQ, config)
        task.name = f'InpaintCaption-Task-{0}'
        self._tasksInpaintCaption.append(task)

        task = self._executorResize.submit(resizeFrame, self._captionOutputQ, self._resizeOutputQ, config)
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

    def wait(self, timeout=None):
        ftWait([self._watcherTask], timeout=timeout)

    @staticmethod
    def get_file_count(config):
        file_count = 0
        basedir = config.image_data_dir if len(config.image_data_dir) > 0 else config.image_png_dir
        for dir_path, dir_names, filenames in os.walk(basedir):
            file_count += len(filenames)
        return file_count

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
                                self._inputQ.put('EOF')
                        if task in self._tasksInpaintCaption:
                            self._tasksInpaintCaption.remove(task)
                            if len(self._tasksInpaintCaption) ==0:
                                print('All Inpaint Caption tasks are finished,shutting down caption executor...')
                                self._executorInpaintCaption.shutdown()
                                print('Inpaint Caption executor is down')
                                self._captionOutputQ.put('EOF')
                        if task in self._tasksResize:
                            self._tasksResize.remove(task)
                            if len(self._tasksResize) == 0:
                                print('All resize tasks are finished, shutting down resize executor...')
                                self._executorResize.shutdown()
                                print('Resize executor is down.')
                                self._resizeOutputQ.put('EOF')
                            # end
                        elif task in self._tasksDepthCalc:
                            self._tasksDepthCalc.remove(task)
                            if len(self._tasksDepthCalc) == 0 and len(self._tasksInpainting):
                                print('All depth-calc tasks are finished, shutting down depth-calc executor...')
                                self._executorDepthCalc.shutdown()
                                print('Depth-calc executor is down.')
                                self._calcDepthOutputQ.put('EOF')
                            # end
                        elif task in self._tasksInpainting:
                            self._tasksInpainting.remove(task)
                            if len(self._tasksInpainting) == 0:
                                print('All inpainting tasks are finished, shutting down inpainting executor...')
                                self._executorInpainting.shutdown()
                                print('Inpainting executor is down.')
                                self._inpaintOutputQ.put('EOF')
                            # end
                        # end
                        elif task in self._tasksOutput:
                            self._tasksOutput.remove(task)
                            if len(self._tasksOutput) == 0:
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
        while not self._inputQ.empty():
            self._inputQ.get()
        self._inputQ.put('EOF')
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
        # self._executorWatcher.shutdown()
