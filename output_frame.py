import queue
import cv2
import logging
import os
import subprocess as sb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def outputFrame(inputQ, config):
    outputFile = config.output_file
    inputFile = config.input_file
    tempFile = './temp.mp4'
    try:
        out = None
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 25
        while True:
            try:
                qElem = inputQ.get(timeout=0.1)
            except queue.Empty:
                continue
            except TimeoutError:
                continue
            # end
            if qElem == 'EOF':
                inputQ.put('EOF')
                break
            if qElem is not None:
                img3d = qElem['img3d']
                h, w, c = img3d.shape
                if out is None:
                    out = cv2.VideoWriter(tempFile, fourcc, fps, (w, h))
                print(f'progress:{100*(qElem["index"]/qElem["total_cnt"]):.03f}%',end='\r')
                # end
                out.write(img3d)
        if out is not None:
            out.release()
        args = ['ffmpeg', '-i', inputFile, '-i', tempFile, '-c:v', 'copy', '-c:a', 'aac', outputFile, '-y']
        p = sb.Popen(args, shell=False)
        p.wait()
        os.remove(tempFile)
        print("All Done")
    except Exception:
        import traceback
        traceback.print_exc()
