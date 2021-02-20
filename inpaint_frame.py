import queue
import numpy as np
import cv2
from model.sttn.model.sttn import InpaintGenerator
from model.sttn.core.utils import Stack, ToTorchFormatTensor
from torchvision import transforms

import torch


def calc_lr_img(disp: np.ndarray, original: np.ndarray):
    h, w, c = original.shape
    disp = cv2.resize(disp, (w, h))
    # calc left
    left_img = np.zeros_like(original)
    filled_left = np.zeros((h, w))
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
    mask_left = ((filled_left == 0).astype(int) * 255).astype(np.uint8)
    # calc right
    right_img = np.zeros_like(original)
    filled_right = np.zeros((h, w))
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            new_j = int(j - disp[i][j])
            if new_j < 0 or new_j >= w:
                continue
            if filled_right[i, new_j] == 0:
                left_img[i, new_j, :] = original[i, j, :]
                filled_right[i, new_j] = disp[i, j]
            else:
                if disp[i, j] > filled_right[i, new_j]:
                    left_img[i, new_j, :] = original[i, j, :]
                    filled_right[i, new_j] = disp[i, j]
    mask_right = ((filled_right == 0).astype(int) * 255).astype(np.uint8)

    return left_img, mask_left, right_img, mask_right

def calc_STTN(frames,masks):
    if len(frames) > 0:
        import math
        frames_seg = []
        masks_seg = []
        max_calc_size = 50
        num = math.ceil(len(frames) / math.ceil(len(frames) / max_calc_size))
        for i in range(0, len(frames), num):
            frames_seg.append(frames[i:i + num])
            masks_seg.append(masks[i:i + num])



def inpaintFrame(inputQ: queue.Queue, outputQ: queue.Queue):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])

    neighbor_stride = 5
    W, H = 432 * 2, 240 * 1
    ref_length = 10
    frames_left = []
    frames_right = []
    masks_left = []
    masks_right = []

    net = InpaintGenerator()
    model = net.to(device)
    ckpt = '../model/sttn/checkpoints/sttn.pth'
    data = torch.load(ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    model.eval()

    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except queue.Empty:
            qElem = None
        # endl
        if qElem == 'EOF':
            inputQ.put('EOF')
            qElem = None
        if qElem is not None:
            # move
            original = qElem['decFrm']
            depth_for_inpaint = qElem['depth_for_inpaint']
            isSeg = qElem['isSeg']
            left_img, mask_left, right_img, mask_right = calc_lr_img(depth_for_inpaint, original=original)
            if not isSeg:
                frames_left.append(left_img)
                frames_right.append(right_img)
                masks_left.append(mask_left)
                masks_right.append(mask_right)
            else:
                pass


