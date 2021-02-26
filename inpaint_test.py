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
W, H = 432 * 2, 240 * 1


def merge_with_STTN(inpaint_img: np.ndarray, original: np.ndarray, mask: np.ndarray):
    h, w, c = original.shape
    inpaint_img = cv2.resize(inpaint_img, (w, h))

    mask_ch3 = np.tile(np.expand_dims(mask, -1), (1, 1, 3))

    mask_ch3 = mask_ch3.astype(np.float32) / 255
    mask_ch3 = cv2.dilate(mask_ch3, cv2.getStructuringElement(cv2.MARKER_CROSS, (3, 3)), iterations=4)
    final_img =  inpaint_img * mask_ch3 + original * (1 - mask_ch3)
    final_img = np.clip(final_img, 0, 255)
    final_img = final_img.astype(np.uint8)
    return final_img


def outputInpaint(outputQ, frames_left, masks_left, frames_right, masks_right, model, qElems):
    print('start calc STTN')
    left_imgs = calc_STTN(frames_left, masks_left, model)
    right_imgs = calc_STTN(frames_right, masks_right, model)
    for i in range(len(left_imgs)):
        cached_qElem = qElems[i]

        inpaint_imgL = left_imgs[i]
        inpaint_imgR = right_imgs[i]

        original_L = cached_qElem['left_img']
        original_R = cached_qElem['right_img']
        mask_L = cached_qElem['mask_left']
        mask_R = cached_qElem['mask_right']

        # imgL = merge_with_STTN(inpaint_img=inpaint_imgL, original=original_L, mask=mask_L)
        # imgR = merge_with_STTN(inpaint_img=inpaint_imgR, original=original_R, mask=mask_R)
        imgL = inpaint_imgL
        imgR = inpaint_imgR
        img3d = np.hstack((imgL, imgR))

        cached_qElem['img3d'] = img3d
        print(f'inpaint:{cached_qElem["index"]}')
        outputQ.put(cached_qElem)


def inpaintFrame(inputQ, outputQ):
    frames_left = []
    frames_right = []
    masks_left = []
    masks_right = []
    qElems = []
    print('>>>>start load sttn model<<<<')
    net = InpaintGenerator()
    model = net.to(device)
    ckpt = 'model/sttn/checkpoints/sttn.pth'
    data = torch.load(ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    model.eval()
    print(f'sttn model loaded!')
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        # end
        if qElem == 'EOF':
            outputInpaint(outputQ, frames_left, masks_left, frames_right, masks_right, model, qElems)
            break
        if qElem is not None:
            original = qElem['decFrm']
            depth_for_inpaint = qElem['depth_for_inpaint']
            isSeg = qElem['isSeg']
            rsd_left_img, rsd_mask_left, rsd_right_img, rsd_mask_right = calc_lr_img(depth_for_inpaint,
                                                                                     original=original, qElem=qElem)
            # img3d = np.hstack((left_img, right_img))
            # qElem['img3d'] = img3d
            # outputQ.put(qElem)
            if not isSeg:
                frames_left.append(rsd_left_img)
                frames_right.append(rsd_right_img)
                masks_left.append(rsd_mask_left)
                masks_right.append(rsd_mask_right)
                qElems.append(qElem)
            else:
                outputInpaint(outputQ, frames_left, masks_left, frames_right, masks_right, model, qElems)
                qElems.clear()
    outputQ.put('EOF')


def calc_lr_img(disp: np.ndarray, original: np.ndarray, qElem):
    @numba.njit()
    def gen_right(disp, filled_right, h, left_img, original, w):
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
        return filled_right

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
        return filled_left

    h, w, c = original.shape
    disp = cv2.resize(disp, (w, h))
    # calc left
    left_img = np.zeros_like(original)
    filled_left = np.zeros((h, w))
    filled_left = gen_left(disp, filled_left, h, left_img, original, w)

    mask_left = ((filled_left == 0).astype(int) * 255).astype(np.uint8)
    
    mask_left = cv2.dilate(mask_left, cv2.getStructuringElement(cv2.MARKER_CROSS, (3, 3)), iterations=4)

    # calc right
    right_img = np.zeros_like(original)
    filled_right = np.zeros((h, w))
    filled_right = gen_right(disp, filled_right, h, right_img, original, w)

    mask_right = ((filled_right == 0).astype(int) * 255).astype(np.uint8)
    mask_right = cv2.dilate(mask_right, cv2.getStructuringElement(cv2.MARKER_CROSS, (3, 3)), iterations=4)
    

    qElem['mask_left'] = mask_left
    qElem['mask_right'] = mask_right
    qElem['left_img'] = left_img
    qElem['right_img'] = right_img

    left_img = Image.fromarray(left_img).resize((W, H))
    mask_left = Image.fromarray(mask_left).resize((W, H), Image.NEAREST)
    right_img = Image.fromarray(right_img).resize((W, H))
    mask_right = Image.fromarray(mask_right).resize((W, H), Image.NEAREST)

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
    ref_length = 10
    inpaint_imgs = []

    if len(frames) > 0:
        import math
        frames_seg = []
        masks_seg = []
        max_calc_size = 10
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

# if __name__ == '__main__':
#     import time
#
#     t0 = time.time()
#     img = cv2.imread('./test/original.png')
#     data = np.load('./test/01500000.npz')
#     disp = data['depth_for_inpaint']
#     h, w, c = img.shape
#     left_img, mask_left, right_img, mask_right = calc_lr_img(disp, img)
#     t1 = time.time()
#     print(f'use Time:{t1 - t0}')
#     cv2.imshow('img',left_img)
#     cv2.waitKey(0)
