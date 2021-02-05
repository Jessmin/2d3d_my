import numpy as np
import queue
from models.sttn.model.sttn import InpaintGenerator
import cv2
from PIL import Image
from torchvision import transforms
import torch
from models.sttn.core.utils import Stack, ToTorchFormatTensor
import xzh_mediaunit as mu
from config import Config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def cut_original_half(img: np.ndarray,W,H):
    img_half = img[img.shape[0] // 2:, ...]
    img = Image.fromarray(img_half)
    img = img.resize((W,H))
    return img

def cut_mask_half(mask:np.ndarray,W,H):
    mask_half = mask[mask.shape[0]//2:,...]
    mask_half = Image.fromarray(mask_half)
    mask_half = mask_half.resize((W, H), Image.NEAREST)
    mask_half = np.array(mask_half.convert('L'))
    mask_half = np.array(mask_half > 0).astype(np.uint8)
    # mask_half = cv2.dilate(mask_half,cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),iterations=4)
    mask_half = Image.fromarray(mask_half * 255)
    return mask_half

def merge_half(inpaint_img: np.ndarray, original: np.ndarray, mask: np.ndarray):
    h, w, c = original.shape
    inpaint_img = cv2.resize(inpaint_img, (w, h // 2))

    mask_ch3 = np.tile(np.expand_dims(mask,-1),(1,1,3))

    mask_ch3 = mask_ch3.astype(np.float32) / 255
    mask_ch3 = cv2.dilate(mask_ch3, cv2.getStructuringElement(cv2.MARKER_CROSS, (3, 3)), iterations=4)
    final_img = np.concatenate([original[:h // 2, :, :], inpaint_img]) * mask_ch3 + original * (1 - mask_ch3)
    final_img = np.clip(final_img, 0, 255)
    final_img = final_img.astype(np.uint8)
    return final_img


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, ref_length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index

def calc_STTN_output(frames_half, masks_half, qElems, masks_empty_num, outputQ, model,neighbor_stride,ref_length,W,H):
    if len(frames_half) == masks_empty_num:
        for i in range(len(frames_half)):
            qElem = qElems[i]
            decFrm = qElem['decFrm']
            qElem['original_inpaint'] = decFrm
            outputQ.put(qElem)
        return
    if len(frames_half) > 0:
        import math 
        frames_half_seg = []
        masks_half_seg = []
        num = len(frames_half) // math.ceil(len(frames_half)/70)
        for i in range(0,len(frames_half),num):
            frames_half_seg.append(frames_half[i:i+num])
            masks_half_seg.append(masks_half[i:i+num])
        for i in range(len(frames_half_seg)):
            frames_half = frames_half_seg[i]
            masks_half = masks_half_seg[i]
            logger.info(f"STTN len:{len(frames_half)}")
            video_length = len(frames_half)
            feats = _to_tensors(frames_half).unsqueeze(0) * 2 - 1
            frames_half = [np.array(f).astype(np.uint8) for f in frames_half]
            binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks_half]
            masks_half = _to_tensors(masks_half).unsqueeze(0)
            feats, masks = feats.to(device), masks_half.to(device)
            comp_frames = [None] * video_length
            with torch.no_grad():
                feats = model.encoder((feats * (1 - masks).float()).view(video_length, 3, H, W))
                _, c, feat_h, feat_w = feats.size()
                feats = feats.view(1, video_length, c, feat_h, feat_w)
            # completing holes by spatial-temporal transformers
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
                ref_ids = get_ref_index(neighbor_ids, video_length, ref_length)
                with torch.no_grad():
                    pred_feat = model.infer(feats[0, neighbor_ids + ref_ids, :, :, :],masks[0, neighbor_ids + ref_ids, :, :, :])
                    pred_img = torch.tanh(model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                    pred_img = (pred_img + 1) / 2
                    pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                    for i in range(len(neighbor_ids)):
                        idx = neighbor_ids[i]
                        img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[idx] + frames_half[idx] * (1 - binary_masks[idx])
                        if comp_frames[idx] is None:
                            comp_frames[idx] = img
                        else:
                            comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
            #end
            for i in range(len(frames_half)):
                inpaint_img = np.array(comp_frames[f]).astype(np.uint8) * binary_masks[f] + frames_half[f] * (1 - binary_masks[f])
                import cv2 
                qElem = qElems[i]
                original = np.array(qElem['decFrm'])
                mask = qElem['mask']
                inpaint_img = merge_half(inpaint_img, original, mask)
                inpaint_img = np.ascontiguousarray(inpaint_img)
                inpaint_frm = mu.MediaFrame.fromRgbArray(inpaint_img, mu.PixelFormat.RGB24)
                qElem['original_inpaint'] = inpaint_frm
                outputQ.put(qElem)
                # logger.info(f'caption frame {qElem["index"]}')
            #end
        #end

def inpaint_caption(inputQ: queue, outputQ: queue, config: Config):
    neighbor_stride = 5
    # W, H = 432*4, 240 *2
    W, H = 432*2, 240 *1
    ref_length = 10
    try:
        split_txt_path = config.frame_split_info
        net = InpaintGenerator()
        model = net.to(device)
        ckpt = 'models/sttn/checkpoints/sttn.pth'
        data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data['netG'])
        model.eval()
        f = open(split_txt_path, 'r')
        line = f.readline()
        frames_half = []
        masks_half = []
        masks_empty_num = 0
        qElems = []
        while True:
            try:
                qElem = inputQ.get(timeout=0.1)
            except TimeoutError:
                continue
            except queue.Empty:
                continue
            if qElem == 'EOF':
                calc_STTN_output(frames_half, masks_half, qElems, masks_empty_num,outputQ, model, neighbor_stride,ref_length, W, H)
                break
            else:
                pts = qElem['present_time']
                if int(pts) <= int(line):
                    original = np.array(qElem['decFrm'])
                    mask = qElem['mask']
                    if mask is None:
                        masks_empty_num += 1
                        h, w, c = original.shape
                        mask = np.zeros((h, w))
                    # end
                    original_half = cut_original_half(original, W, H)
                    mask_half = cut_mask_half(mask, W, H)
                    frames_half.append(original_half)
                    masks_half.append(mask_half)
                    qElems.append(qElem)
                else:
                    if len(frames_half) >1:
                        calc_STTN_output(frames_half, masks_half, qElems, masks_empty_num, outputQ, model,neighbor_stride,ref_length,W, H)
                        # read next line
                        frames_half = []
                        masks_half = []
                        qElems = []
                        masks_empty_num = 0
                    line = f.readline()
                # end
            # end
        inputQ.put("EOF")
    except Exception:
        import traceback
        traceback.print_exc()
