import queue
import numpy as np
import cv2
from model.sttn.model.sttn import InpaintGenerator
from model.sttn.core.utils import Stack, ToTorchFormatTensor
from torchvision import transforms
import numba
import torch
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def inpaintFrame(inputQ, outputQ):
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        # end
        if qElem == 'EOF':
            break
        if qElem is not None:
            original = qElem['decFrm']
            depth_for_inpaint = qElem['depth_for_inpaint']
            left_img, right_img = calc_lr_img(depth_for_inpaint,original=original)
            img3d = np.hstack((left_img, right_img))
            qElem['img3d'] = img3d
            outputQ.put(qElem)
    outputQ.put('EOF')


def calc_lr_img(disp: np.ndarray, original: np.ndarray):
    @numba.njit()
    def gen_right(disp, filled_right, h, right_img, original, w):
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                new_j = int(j - disp[i][j])
                if new_j < 0 or new_j >= w:
                    continue
                if filled_right[i, new_j] == 0:
                    right_img[i, new_j, :] = original[i, j, :]
                    filled_right[i, new_j] = disp[i, j]
                else:
                    if disp[i, j] > filled_right[i, new_j]:
                        right_img[i, new_j, :] = original[i, j, :]
                        filled_right[i, new_j] = disp[i, j]
        # inpaint right
        for i in range(h - 1, -1, -1):
            for j in range(w - 2, -1, -1):
                if filled_right[i, j] == 0:
                    if filled_right[i, j + 1] != 0:
                        right_img[i, j, :] = right_img[i, j + 1, :]
        return right_img

    @numba.njit()
    def gen_left(disp, filled_left, h, left_img, original, w):
        for i in range(h):
            for j in range(w):
                new_j = int(j + disp[i, j])
                if new_j >= w or new_j < 0:
                    continue
                if filled_left[i, new_j] == 0:
                    left_img[i, new_j, :] = original[i, j, :]
                    filled_left[i, new_j] = disp[i, j]
                else:
                    if disp[i, j] > filled_left[i, new_j]:
                        left_img[i, new_j, :] = original[i, j, :]
                        filled_left[i, new_j] = disp[i, j]
                        if disp[i, j] ==0:
                            filled_left[i, new_j] = 0.1
        # inpaint left
        for i in range(h):
            for j in range(1, w):
                if filled_left[i, j] == 0:
                    if filled_left[i, j - 1] != 0:
                        left_img[i, j, :] = left_img[i, j - 1, :]
        return left_img

    h, w, c = original.shape
    disp = cv2.resize(disp, (w, h))
    # calc left
    left_img = np.zeros_like(original)
    filled_left = np.zeros((h, w))
    left_img = gen_left(disp, filled_left, h, left_img, original, w)
    # calc right
    right_img = np.zeros_like(original)
    filled_right = np.zeros((h, w))
    right_img = gen_right(disp, filled_right, h, right_img, original, w)
    return left_img, right_img


# if __name__ == '__main__':
#     import time
#
#     t0 = time.time()
#     img = cv2.imread('./test/original.png')
#     data = np.load('./test/01500000.npz')
#     disp = data['depth_for_inpaint']
#     h, w, c = img.shape
#     left_img, right_img = calc_lr_img(disp, img)
#     t1 = time.time()
#     print(f'use Time:{t1 - t0}')
#     cv2.imshow('img',right_img)
#     cv2.waitKey(0)
