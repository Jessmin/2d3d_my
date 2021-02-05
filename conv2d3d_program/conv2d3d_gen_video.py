import os
import argparse
import zipfile
import timeit as tt
import json
from traceback import print_exc
import numpy as np
from PIL import Image
import xzh_mediaunit as mu
from xzh_mediaunit.utils.media_sink_output import MediaSinkOutput

def parseArgs():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--input-dir', dest='inputDir', required=True)
    argParser.add_argument('--output-file', dest='outputFile', required=True)
    argParser.add_argument('--bitrate', dest='bitrate', type=int, default=40*1024*1024)
    argParser.add_argument('--frame-info-json', dest='frameInfoJson', default='')
    argParser.add_argument('--start-mts', dest='startMts', type=int)
    argParser.add_argument('--end-mts', dest='endMts', type=int)
    argParser.add_argument('--reset-ts', dest='resetTs', action="store_true")
    argParser.add_argument('--output-mono-width', dest='outputMonoWidth', type=int)
    argParser.add_argument('--redblue-mode', dest='redBlueMode', action="store_true")
    argParser.add_argument('--squeeze-width-half', dest='squeezeWidthHalf', action="store_true")
    argParser.add_argument('--output-resize-scale', dest='outputResizeScale', default=True)
    argParser.add_argument('--product-mode', dest='product_mode', default=False)
    args = argParser.parse_args()
    return args

class Make3dVideo:
    def __init__(self, config):
        if not os.path.isdir(config.inputDir):
            raise ValueError(f'INVALID input dir \'{config.inputDir}\'!')
        self._inputDir = config.inputDir
        self._parseVideoInfo(config.inputDir, frameInfoJsonPath=config.frameInfoJson)
        self._outMonoW = config.outputMonoWidth if config.outputMonoWidth is not None else self._inMonoW
        if self._outMonoW%2 == 1:
            self._outMonoW += 1
        self._outMonoH = int(self._inMonoH/self._inMonoW*self._outMonoW)
        if self._outMonoH%2 == 1:
            self._outMonoH += 1
        self._3dMode = 0
        if config.redBlueMode:
            self._3dMode = 1
            self._outW = self._outMonoW
            self._outH = self._outMonoH
        elif config.squeezeWidthHalf:
            self._outMonoH *= 2
            self._outMonoW = self._outMonoW//2
            self._outW = self._outMonoW*2
            self._outH = self._outMonoH
        self._parsePtsMinMax(config.startMts, config.endMts)
        self._ptsOffset = self._minPts if config.resetTs else 0
        self._frameFileNameL = 'stereo_l.png'
        self._frameFileNameR = 'stereo_r.png'

        if self._inMonoW != self._outMonoW or self._inMonoH != self._outMonoH:
            self._resizeFilter = mu.FilterGraph(mu.MediaType.VIDEO, f'scale={self._outMonoW}:{self._outMonoH}')
        else:
            self._resizeFilter = None
        self._vout = MediaSinkOutput(config.outputFile, codecName='h264_nvenc', fps=self._encFrameRate,
                                     pixfmt=mu.PixelFormat.YUV420P, bitrate=config.bitrate)
        # self._vout = MediaSinkOutput(config.outputFile, codecName='hevc_nvenc', fps=self._encFrameRate,
                                    #  pixfmt=mu.PixelFormat.YUV420P, bitrate=config.bitrate)
    #end

    def _parseVideoInfo(self, inputDir, frameInfoJsonPath=None):
        if frameInfoJsonPath is None or not os.path.exists(frameInfoJsonPath):
            frameInfoJsonPath = os.path.join(inputDir, 'frameinfo.json')
        if not os.path.isfile(frameInfoJsonPath):
            raise ValueError(f'Cannot find info json file at: \'{frameInfoJsonPath}\'!')
        with open(frameInfoJsonPath, 'r') as f:
            data = json.load(f)
            frame_rate_den = data['frame_rate']['den']
            frame_rate_num = data['frame_rate']['num']
            time_base_den = data['time_base']['den']
            time_base_num = data['time_base']['num']
            self._encFrameRate = mu.Rational(frame_rate_num, frame_rate_den)
            self._encFrameTimeBase = mu.Rational(time_base_num, time_base_den)
            self._inMonoW = data['width']
            self._inMonoH = data['height']
    #end

    def _parsePtsMinMax(self, startMts, endMts):
        frameInfoFilePath = os.path.join(self._inputDir, 'frameinfo.txt')
        if not os.path.isfile(frameInfoFilePath):
            raise ValueError(f'CANNOT find frame info file at \'{frameInfoFilePath}\'!')
        totalFrameCount = 0
        minMts = maxMts = None
        minPts = maxPts = None
        lineCount = 0
        with open(frameInfoFilePath) as f:
            while True:
                line = f.readline()
                if len(line) <= 0:
                    break
                lineCount += 1
                try:
                    parts = line.split(',')
                    mts = int(parts[0].strip())
                    pts = int(parts[1].strip())
                    if startMts is not None and mts < startMts:
                        continue
                    if endMts is not None and mts > endMts:
                        continue
                    if minMts is None or mts < minMts:
                        minMts = mts
                        minPts = pts
                        totalFrameCount = 1
                    elif maxMts is None or mts > maxMts:
                        maxMts = mts
                        maxPts = pts
                        totalFrameCount += 1
                    else:
                        totalFrameCount += 1
                except Exception as e:
                    print(f'Exception raised when parsing \'frameinfo.txt\' at line {lineCount}: {e}.')
        print(f'minPts={minMts}, maxPts={maxMts}, totalFrameCount={totalFrameCount}')
        self._minPts = minPts
        self._maxPts = maxPts
        self._totalFrameCount = totalFrameCount
        self._frameInfoFile = open(frameInfoFilePath)
    #end

    def _readNextFrame(self):
        line = self._frameInfoFile.readline()
        if len(line) <= 0:
            return None
        parts = line.split(',')
        mts = int(parts[0].strip())
        pts = int(parts[1].strip())
        if pts < self._minPts:
            return ()
        if pts > self._maxPts:
            return None
        minute = mts//60000
        zipFilePath = os.path.join(self._inputDir, f'{minute:03d}', f'{mts:08d}.zip')
        with zipfile.ZipFile(zipFilePath, 'r') as z:
            entryNames = z.namelist()
            if self._frameFileNameL not in entryNames or self._frameFileNameR not in entryNames:
                raise ValueError(f'Either \'{self._frameFileNameL}\' or \'{self._frameFileNameR}\' isn\'t in \'{zipFilePath}\'!')
            imgDataL = z.open(self._frameFileNameL)
            imgDataR = z.open(self._frameFileNameR)
            npImgL = np.asarray(Image.open(imgDataL))
            npImgR = np.asarray(Image.open(imgDataR))
        return (pts, npImgL, npImgR)
    #end

    def _composeStereoFrame(self, npImgL, npImgR):
        if self._resizeFilter is not None:
            tmpL = mu.MediaFrame.fromRgbArray(npImgL, mu.PixelFormat.RGB24)
            if not self._resizeFilter.is_inited:
                self._resizeFilter.initVideoFilterGraphByFrame(tmpL)
            tmpL = self._resizeFilter.writeFrame(tmpL)
            npImgL = np.asarray(tmpL)
            tmpR = mu.MediaFrame.fromRgbArray(npImgR, mu.PixelFormat.RGB24)
            tmpR = self._resizeFilter.writeFrame(tmpR)
            npImgR = np.asarray(tmpR)
        if self._3dMode == 0:
            npStereo = np.hstack((npImgL, npImgR))
        elif self._3dMode == 1:
            npImgLg = npImgL[:, :, 1]
            npImgRg = npImgR[:, :, 1]
            npImgRb = npImgR[:, :, 2]
            npStereo = np.stack((npImgLg, npImgRg, npImgRb), axis=-1)
        else:
            raise ValueError(f'Unsupported 3d-mode: {self._3dMode}!')
        return npStereo
    #end

    def make(self):
        while True:
            try:
                res = self._readNextFrame()
                if res is None:
                    break
                elif len(res) <= 0:
                    continue
            except Exception as e:
                print_exc()
                continue
            pts = res[0]
            npStereo = self._composeStereoFrame(res[1], res[2])
            frmStereo = mu.MediaFrame.fromRgbArray(np.ascontiguousarray(npStereo), mu.PixelFormat.RGB24)
            frmStereo.time_base = self._encFrameTimeBase
            frmStereo.pts = pts-self._ptsOffset
            self._vout.outputOneMediaFrame(frmStereo)
            print(f'pts={pts}, {(pts-self._minPts)*100/(self._maxPts-self._minPts):.2f}%', end='\r')
    #end
#end

if __name__ == '__main__':
    import time
    start_time = time.time()
    args = parseArgs()
    make3dVideo = Make3dVideo(args)
    make3dVideo.make()
    print('')
    print('Done.')
    end_time= time.time()
    print(f'use Time:{end_time - start_time}')
#end