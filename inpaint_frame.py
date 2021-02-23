import queue
import numpy as np
import cv2
from model.sttn.model.sttn import InpaintGenerator
from model.sttn.core.utils import Stack, ToTorchFormatTensor
from torchvision import transforms

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, ref_length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


def calc_STTN(frames, masks, model):
    _to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])
    neighbor_stride = 5
    # W, H = 432*4, 240 *2
    W, H = 432 * 2, 240 * 1
    ref_length = 10
    inpaint_imgs = []

    if len(frames) > 0:
        import math
        frames_seg = []
        masks_seg = []
        max_calc_size = 50
        num = math.ceil(len(frames) / math.ceil(len(frames) / max_calc_size))
        for i in range(0, len(frames), num):
            frames_seg.append(frames[i:i + num])
            masks_seg.append(masks[i:i + num])
        for i in range(len(frames_seg)):
            frames = frames_seg[i]
            masks = masks_seg[i]
            video_length = len(frames)
            feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
            frames = [np.array(f).astype(np.uint8) for f in frames]
            binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
            masks = _to_tensors(masks).unsqueeze(0)
            feats, masks = feats.to(device), masks.to(device)
            comp_frames = [None] * video_length
            with torch.no_grad():
                feats = model.encoder((feats * (1 - masks).float()).view(video_length, 3, H, W))
                _, c, feat_h, feat_w = feats.size()
                feats = feats.view(1, video_length, c, feat_h, feat_w)
            # completing holes by spatial-temporal transformers
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [i for i in
                                range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
                ref_ids = get_ref_index(neighbor_ids, video_length, ref_length)
                with torch.no_grad():
                    pred_feat = model.infer(feats[0, neighbor_ids + ref_ids, :, :, :],
                                            masks[0, neighbor_ids + ref_ids, :, :, :])
                    pred_img = torch.tanh(model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                    pred_img = (pred_img + 1) / 2
                    pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                    for i in range(len(neighbor_ids)):
                        idx = neighbor_ids[i]
                        img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[idx] + frames[idx] * (
                                1 - binary_masks[idx])
                        if comp_frames[idx] is None:
                            comp_frames[idx] = img
                        else:
                            comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(
                                np.float32) * 0.5
            # end
            for f in range(video_length):
                inpaint_img = np.array(comp_frames[f]).astype(np.uint8) * binary_masks[f] + frames[f] * (
                        1 - binary_masks[f])
                inpaint_imgs.append(inpaint_img)
    return inpaint_imgs


def inpaintFrame(inputQ: queue.Queue, outputQ: queue.Queue):
    frames_left = []
    frames_right = []
    masks_left = []
    masks_right = []
    qElems = []

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
                qElems.append(qElem)
            else:
                left_imgs = calc_STTN(frames_left, masks_left, model)
                right_imgs = calc_STTN(frames_right, masks_right, model)
                for i in range(len(left_imgs)):
                    imgL = left_imgs[i]
                    imgR = right_img[i]
                    img3d = np.hstack((imgL, imgR))
                    cached_qElem = qElems[i]
                    cached_qElem['img3d'] = cached_qElem
                    outputQ.put(cached_qElem)
                qElems = []
        outputQ.put('EOF')
