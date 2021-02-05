import xzh_mediaunit as mu
import argparse
import sys
from xzh_mediaunit.constants import MediaType, PixelFormat
from xzh_mediaunit.errors import EofException
import os
import zipfile as zp
import numpy as np
import cv2
import multiprocessing as mp
from queue import Empty, Full
import timeit as tt
import logging
import shutil

logging.basicConfig(
    format='[%(asctime)-15s|%(name)-8s|%(threadName)-12s|%(funcName)-16s|%(lineno)-5d][%(levelname)8s] %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', dest='inputFile', required=True)
parser.add_argument('--output-dir', dest='outputDir', required=True)
parser.add_argument('--start-frame', dest='startFrame', type=int, default=0)
parser.add_argument('--end-frame', dest='endFrame', type=int, default=0)
parser.add_argument('--max-frame', dest='maxFrame', default=0, type=int)
parser.add_argument('--seek', dest='seek')
parser.add_argument('--process-num', dest='process', default=6, type=int)
args = parser.parse_args()


class ExtractVideo(object):
    def __init__(self):

        self.startFrame = args.startFrame
        self.endFrame = args.endFrame
        if os.path.exists(args.outputDir):
            shutil.rmtree(args.outputDir)
        os.makedirs(args.outputDir)
        # end
        mediaSource = mu.MediaSource()
        mediaSource.open(args.inputFile)

        mediaInfo = mediaSource.media_info
        videoStreamIndex = -1
        width = height = None
        for streamInfo in mediaInfo.streams:
            if streamInfo.media_type == MediaType.VIDEO:
                videoStreamIndex = streamInfo.index
                self.frame_width = streamInfo.width
                self.frame_height = streamInfo.height
                self.frame_rate = streamInfo.frame_rate
                self.time_base = streamInfo.time_base
                nb_frames = streamInfo.nb_frames
                if nb_frames == 0:
                    nb_frames = int(np.ceil(mediaInfo.duration / 1000 * (self.time_base.num / self.time_base.den) * (
                            self.frame_rate.num / self.frame_rate.den)))
                self.total_frames = nb_frames if args.maxFrame == 0 else args.maxFrame
                self.duration = streamInfo.duration
                break
            # end
        # end
        # self.check_free_space()
        self.saveFrameInfoJSon()
        if videoStreamIndex < 0:
            print("There is no video stream found in file '" + args.inputFile + "'.")
            sys.exit(-1)
        self.videoDecoder = mediaSource.createDecoderForStream(videoStreamIndex)  # , useHwaccel=True
        self.videoDecoder.setConfig(rectifyTsMode=mu.RectifyTimeStampMode.CFR, cscSliceCount=8)
        mu.linkSourceToSink(mediaSource, self.videoDecoder)

        if args.seek is not None:
            mediaSource.seekToI(args.seek)
            print(f"Seek to {args.seek:.3f} seconds.")
        # end

        self.decFrmCnt = 0
        self.workPixfmt = PixelFormat.RGB24
        mainProcPid = os.getpid()
        mpCtx = mp.get_context('fork')
        self.inputQ = [mpCtx.Queue(maxsize=10) for x in range(args.process)]
        self.sub_proc = []
        for i in range(args.process):
            _p = mpCtx.Process(target=self.saveFrame, args=(self.inputQ[i], mainProcPid))
            _p.start()
            self.sub_proc.append(_p)
        # end
        self.run()

    def check_free_space(self):
        st = os.statvfs(args.outputDir)
        mb = st.f_bavail * st.f_frsize / 1024 / 1024
        if mb < self.total_frames * 5:
            print(f'not enough space to extra all the videos!')
            sys.exit(-1)

    def run(self):
        idx = 0
        while args.maxFrame == 0 or self.decFrmCnt < args.maxFrame:
            try:
                decOutFrm = self.videoDecoder.decodeOneFrame(asFormat=self.workPixfmt)
            except EofException:
                print("Video input EOF.")
                break
            pts = decOutFrm.present_time
            self.saveFrameInfo(decOutFrm)
            decOutFrm = np.asarray(decOutFrm)
            qElem = {'decFrm': decOutFrm, 'presentTime': pts}
            if self.startFrame != 0 and int(pts) < self.startFrame:
                continue
            if self.endFrame != 0 and int(pts) > self.endFrame:
                break
            if idx == len(self.inputQ):
                idx = 0
            self.inputQ[idx].put(qElem)
            idx += 1
            self.decFrmCnt += 1
            logger.info(f'process:{((self.decFrmCnt / (self.total_frames + 1)) * 100):.02f}%')

        for i in range(args.process):
            self.inputQ[i].put('EOF')
            self.sub_proc[i].join()
        # end
        print('All work completed!')

    def encodeArr(self, arr, type='.png'):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(type, arr)
        if success:
            return encoded
        else:
            print(f'failed!')
            return None

    def saveFrameInfoJSon(self):
        frameInfoJson = os.path.join(args.outputDir, 'frameinfo.json')
        with open(frameInfoJson, 'w') as f:
            import json
            frame_rate_num = self.frame_rate.num
            frame_rate_den = self.frame_rate.den
            frame_rate = {'den': frame_rate_den, 'num': frame_rate_num}

            time_base_num = self.time_base.num
            time_base_den = self.time_base.den
            time_base = {'den': time_base_den, 'num': time_base_num}
            data = {'frame_rate': frame_rate, 'time_base': time_base,
                    'duration': self.duration, 'height': self.frame_height, 'width': self.frame_width}
            json.dump(data, f)

    def saveFrameInfo(self, frame: mu.MediaFrame):
        present_time = frame.present_time
        pts = frame.pts
        frameInfoFile = os.path.join(args.outputDir, 'frameinfo.txt')
        with open(frameInfoFile, 'a') as f:
            f.write(f'{present_time},{pts} \n')

    def pid_exists(self, pid):
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

    def saveFrame(self, inputQ, parentPid, isMain=False):
        inputEof = False
        while not inputEof:
            try:
                qElem = inputQ.get(timeout=0.1)
                if qElem == 'EOF':
                    inputEof = True
                    break
                # end
            except Empty:
                if not self.pid_exists(parentPid):
                    print(
                        f'!!!!!!!!!! Parent process (pid={parentPid}) does not exist any more. Sub-process \'{parentPid}\' will exit !!!!!!!!!!!')
                    inputEof = True
                    break
                else:
                    continue

            except Exception as e:
                print(f'Exception! {e}')
                inputEof = True
                break
            # end
            decFrm = qElem.pop('decFrm')
            pts = qElem.pop('presentTime')
            source_img_name = f'original.png'
            zip_file_name = f'{pts:08d}.zip'
            base_dir = args.outputDir
            dirName = f'{(pts // (1000 * 60)):03d}'
            saveDir = os.path.join(base_dir, dirName)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            zipFilepath = os.path.join(saveDir, zip_file_name)
            decFrm = np.asarray(decFrm)

            source_encoded = self.encodeArr(decFrm)
            with zp.ZipFile(zipFilepath, 'w') as myzip:
                myzip.writestr(source_img_name, source_encoded)

    def depthToDisplayImage(self, depth, minmax=None):
        if minmax is None:
            min = depth.min()
            max = depth.max()
        else:
            min = minmax[0]
            max = minmax[1]
        depImg = (((depth - min) / (max - min)).clip(0, 1) * 255).astype(np.uint8)
        return depImg


if __name__ == '__main__':
    start = tt.default_timer()
    ExtractVideo()
    end = tt.default_timer()
    print(f'useTime:{end - start}')
