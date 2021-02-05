import queue
import numpy as np
import tempfile
import shutil
import zipfile
import os
import cv2
import xzh_mediaunit as mu
from config import Config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reinpaint(caption_moved_mask, seam_normal, caption_moved):
    h, w, c = seam_normal.shape
    for i in range(h):
        for j in range(w):
            if caption_moved_mask[i, j] > 0:
                seam_normal[i, j] = caption_moved[i, j]
    return seam_normal


def inpaint_with_caption(frmLR: np.ndarray, mask: np.ndarray, original: np.ndarray, offset: int):
    # get center
    h, w = mask.shape
    left = w
    top = h
    right = 0
    bottom = h//2
    for i in range(h//2,h):
        for j in range(w):
            if mask[i, j] > 0:
                if i < top:
                    top = i
                if j < left:
                    left = j
                if i > bottom:
                    bottom = i
                if j > right:
                    right = j
    # end
    center = ((right - left) // 2 + left, (bottom - top) // 2 + top)

    h, w, c = original.shape
    caption_roi = original[h // 2:, ...]

    mask_roi = np.zeros_like(mask)
    mask_roi[top:bottom, left:right] = 255
    mask_roi = mask_roi[h // 2:, ...]

    h, w, c = frmLR.shape
    frm_left = frmLR[:, :w // 2, :]
    frm_right = frmLR[:, w // 2:, :]

    center_left = (center[0] - offset, center[1])
    center_right = (center[0] + offset, center[1]) 

    img_left = cv2.seamlessClone(caption_roi, frm_left, mask_roi, center_left, cv2.MIXED_CLONE)
    img_right = cv2.seamlessClone(caption_roi, frm_right, mask_roi, center_right, cv2.MIXED_CLONE)
    
    # original = cv2.resize(original,(img_left.shape[1],img_left.shape[0]))
    
    # kernel = np.ones((5,5),np.uint8)  
    # caption_mask = cv2.erode(mask,kernel,iterations = 3)

    # caption_mask_left = np.roll(caption_mask, -1 * offset, axis=1)
    # caption_mask_right = np.roll(caption_mask, offset, axis=1)
    # caption_left = np.roll(original, -1 * offset, axis=1)
    # caption_right = np.roll(original, offset, axis=1)

    # img_left = reinpaint(caption_mask_left, img_left, caption_left)
    # img_right = reinpaint(caption_mask_right, img_right, caption_right)
    
    frmLR = np.concatenate((img_left,img_right),axis=1)
    return frmLR


def outputFrame(inputQ: queue, config: Config, totalFrame:int):
    image_data_dir = config.image_data_dir
    depthInfo = config.depthInfo
    crop = config.crop
    outputMonoSize = config.output_mono_size
    bottomOffset = config.bottom_offset
    output_mono_scale = config.output_mono_scale
    _outputCnt = 0
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        if qElem == 'EOF':
            break
        try:
            frmLR = qElem['frmLR']
            decFrm = qElem['decFrm']
            pst = qElem['present_time']
            frmLR = cropFrame(decFrm, frmLR, crop, outputMonoSize, bottomOffset)
            depth = qElem['depth']
            mask = qElem['mask']
            depth_for_inpaint = qElem['depth_for_inpaint']
            original = np.asarray(decFrm)
            depth = cropDepth(depth, original, crop)
            depth_for_inpaint = cropDepth(depth_for_inpaint, original, crop)
            with open(depthInfo, 'a') as f:
                f.write(f'{pst},{depth.max()},{depth.min()}\n')
            frmLR = np.asarray(frmLR)
            if mask is not None:
                frmLR = inpaint_with_caption(frmLR, mask, original, 30)
            h, w, c = frmLR.shape
            if output_mono_scale == 1:
                if h / (w // 2) != 9 / 16:
                    if h / (w // 2) < 9 / 16:
                        new_h = int(9 / 16 * w // 2)
                    padding = int((new_h - h) // 2)
                    new_frame = np.zeros((new_h, w, c), dtype=np.uint8)
                    new_frame[padding:padding + h, :, :] = frmLR
                    frmLR = new_frame
                else:
                    new_w = int(h * 32 / 9)
                    padding = int((new_w - w) // 2)
                    new_frame = np.zeros((h, new_w, c), dtype=np.uint8)
                    new_frame[:, padding:padding + w, :] = frmLR
                    frmLR = new_frame
            frmLR_l = frmLR[:, 0:w // 2, :]
            frmLR_r = frmLR[:, w // 2:, :]
            zipfilePath = qElem['zipfilePath']
            npzfilePath = qElem['npzfilePath']
            original_inpaint = qElem['original_inpaint']
            original_inpaint = np.asarray(original_inpaint)
            # save to zip
            image_map = {
                'inpaint_depth.png': depth_for_inpaint,
                'depth.png': depth,
                'stereo_l.png': frmLR_l,
                'stereo_r.png': frmLR_r,
                'original_inpaint.png':original_inpaint,
            }
            if len(image_data_dir) < 1:
                try:
                    image_map['original.png'] = np.asarray(decFrm)
                except Exception:
                    import traceback
                    traceback.print_exc()
            writeImgToZip(image_map, zipfilePath=zipfilePath)
            # save to npz
            h, w = depth.shape
            # reduce use space
            depth_for_inpaint_for_save = cv2.resize(depth_for_inpaint, (w, h))
            npz_dict = {'depth': depth, 'depth_for_inpaint': depth_for_inpaint_for_save}
            np.savez_compressed(npzfilePath, **npz_dict)
            _outputCnt += 1
            logger.info(f'progress:{((_outputCnt / totalFrame) * 100):.2f}%,output index:{qElem["index"]}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f'save npz file error:{e}')
    inputQ.put('EOF')
    logger.info(f'Output Done')


def remove_from_zip(zipfname, filenames):
    tempdir = tempfile.mkdtemp()
    try:
        tempname = os.path.join(tempdir, 'new.zip')
        with zipfile.ZipFile(zipfname, 'r') as zipread:
            with zipfile.ZipFile(tempname, 'w') as zipwrite:
                for item in zipread.infolist():
                    if item.filename not in filenames:
                        data = zipread.read(item.filename)
                        zipwrite.writestr(item, data)
        shutil.move(tempname, zipfname)
    except Exception as e:
        print(f'remove file:{zipfname} error :{e}')
    finally:
        shutil.rmtree(tempdir)


def encodeArr(arr, encode_type='.png'):
    if len(arr.shape) == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    success, encoded = cv2.imencode(encode_type, arr)
    if success:
        return encoded.tobytes()
    else:
        return None


def writeImgToZip(map_data: dict, zipfilePath: str):
    if not os.path.isfile(zipfilePath):
        logger.warning(f'{zipfilePath} does not exits!')
    try:
        files_to_del = []
        with zipfile.ZipFile(zipfilePath, 'a') as myZip:
            for key in map_data.keys():
                if key in myZip.namelist():
                    files_to_del.append(key)
        if len(files_to_del) > 0:
            remove_from_zip(zipfilePath, files_to_del)
        save_data = {}
        for (pngName, img_data) in map_data.items():
            if img_data.dtype != np.uint8:
                img_data = 255 * (img_data - np.min(img_data)) / (
                        np.max(img_data) - np.min(img_data))
            img_encoded = encodeArr(img_data)
            if img_encoded is not None:
                save_data[pngName] = img_encoded
            else:
                logger.warning(f'save image file:{pngName} failed')
        with zipfile.ZipFile(zipfilePath, 'a') as myZip:
            for (pngName, img_encoded) in save_data.items():
                myZip.writestr(pngName, img_encoded)
    except Exception:
        import traceback
        print(f'{traceback.print_exc()}')


def cropDepth(depth, original, crop):
    if crop is None:
        return depth
    h_depth, w_depth = depth.shape
    h_original, w_original, c = original.shape
    top_crop, h_crop, left_crop, w_crop = crop

    out_depth_w = w_depth / w_crop * w_original
    out_depth_h = h_depth / h_crop * h_original
    out_depth_top = top_crop / h_original * out_depth_h
    out_depth_left = left_crop / w_original * out_depth_w
    out_depth_h = int(out_depth_h)
    out_depth_w = int(out_depth_w)
    out_depth = np.zeros((out_depth_h, out_depth_w))
    out_depth_top = int(out_depth_top)
    out_depth_left = int(out_depth_left)
    out_depth[out_depth_top:out_depth_top + h_depth, out_depth_left:out_depth_left + w_depth] = depth

    return out_depth


def cropFrame(decFrm, outFrm, crop, outputMonoSize, bottomOffset):
    if crop is not None:
        source_outFrm = np.asarray(outFrm)
        h, w, c = source_outFrm.shape
        source_outFrm_left = source_outFrm[:, :w // 2, :]
        source_outFrm_right = source_outFrm[:, w // 2:, :]
        top, height, left, width = crop
        input_height, input_width = decFrm.height, decFrm.width
        out_width, out_height = outputMonoSize[0] * 2, outputMonoSize[1]
        height_ratio = out_height / input_height
        width_ratio = out_width / 2 / input_width
        top_out, height_out = int(top * height_ratio), int(height * height_ratio)
        left_out, width_out = int(left * width_ratio), int(width * width_ratio)
        import cv2
        decFrm_clone = mu.MediaFrame.clone(decFrm)
        if decFrm_clone.format != mu.PixelFormat.RGB24:
            decFrm_clone.toPixelFormat(mu.PixelFormat.RGB24, cscSliceCount=8)
        decFrm_clone = np.asarray(decFrm_clone)
        resized_decFrm = cv2.resize(decFrm_clone, (int(out_width // 2), int(out_height)))
        # bottom_offset
        np_bottom = resized_decFrm[top_out + height_out:, :, :]
        np_bottom_left = np.roll(np_bottom, bottomOffset, axis=1) if bottomOffset is not None else np_bottom
        np_bottom_right = np.roll(np_bottom, bottomOffset * -1, axis=1) if bottomOffset is not None else np_bottom
        npOutFrm = np.zeros((out_height, out_width, 3), dtype=source_outFrm.dtype)
        h, w, c = npOutFrm.shape
        leftOutFrm = npOutFrm[:, :w // 2, :]
        rightOutFrm = npOutFrm[:, w // 2:, :]
        leftOutFrm[top_out:top_out + height_out, left_out:left + width_out] = source_outFrm_left
        leftOutFrm[top_out + height_out:, :, :] = np_bottom_left
        rightOutFrm[top_out:top_out + height_out, left_out:left + width_out] = source_outFrm_right
        rightOutFrm[top_out + height_out:, :, :] = np_bottom_right
        outFrm = np.ascontiguousarray(npOutFrm)
        outFrm = mu.MediaFrame.fromRgbArray(outFrm, mu.PixelFormat.RGB24)
    return outFrm
