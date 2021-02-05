import cv2
import numpy as np

filepath = '/home/zhaohoj/Videos/龙门客栈-withSTTN.mp4'
cap = cv2.VideoCapture(filepath)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fourcc = cap.get(cv2.CAP_PROP_FOURCC)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        break
cap.release()
out.release()

def readFrame():
    pass