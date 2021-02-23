import queue
import cv2


def outputFrame(inputQ, outputFile):
    out = None
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 25
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except queue.Empty:
            qElem = None
        # end
        if qElem == 'EOF':
            inputQ.put('EOF')
            qElem = None
            break
        if qElem is not None:
            img3d = qElem['img3d']
            h, w, c = img3d.shape
            if out is None:
                out = cv2.VideoWriter(outputFile, fourcc, fps, (w, h))
            #end
            out.write(img3d)
    out.release()

