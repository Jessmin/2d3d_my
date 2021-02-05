import os
import shutil
from PIL import Image
import queue
import zipfile
import numpy as np
import xzh_mediaunit as mu
from xzh_mediaunit.utils import MediaSourceVideoInput
from config import FrameScaleConfig
import logging
import sys
import cv2
from config import Config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_npz_fixed_data(name: str, npz_fixed_data_dir: str):
    if len(npz_fixed_data_dir) > 0:
        for maindir, subdir, file_name_list in os.walk(npz_fixed_data_dir):
            npz_find_name = f'{name}.npz'
            if npz_find_name in file_name_list:
                apath = os.path.join(maindir, npz_find_name)
                logger.info(f'load npz data from:{apath} successful')
                return apath
    return None


def get_file_list(dir_name):
    result = []
    for maindir, subdir, file_name_list in os.walk(dir_name):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


def get_scale(frame_scale_config_list, present_time):
    for frame_scale in frame_scale_config_list:
        if isinstance(frame_scale, FrameScaleConfig):
            start_frame = frame_scale.start_frame
            end_frame = frame_scale.end_frame
            if start_frame is not None and (end_frame is None or end_frame == 0):
                end_frame = start_frame
            scale = frame_scale.scale
            if start_frame <= present_time <= end_frame:
                return scale
    return 10000


def get_mask(config:Config, original_img: np.ndarray, present_time):
    try:
        def get_crop():
            crop = config.frame_mask_config['crop'].split(',')
            crop = [int(x) for x in crop]
            specific_crop_list = config.frame_mask_config['specific_crop']
            for specific_crop in specific_crop_list:
                start_time = specific_crop['start_time']
                end_time = specific_crop['end_time']
                if int(start_time) <= int(present_time) <= int(end_time):
                    crop = specific_crop['crop'].split(',')
                    crop = [int(x) for x in crop]
                    return crop
            return crop

        def is_except_frame():
            except_frame_list = config.frame_mask_config['except_frame']
            for except_frame in except_frame_list:
                start_time = except_frame['start_time']
                end_time = except_frame['end_time']
                if int(start_time) <= int(present_time) <= int(end_time):
                    return True
            return False

        mask_file_name = f'{int(present_time):08d}.png'
        sub_dir = f'{(int(present_time) // (1000 * 60)):03d}'
        mask = None
        if config.frame_mask_dir is not None and len(config.frame_mask_dir) > 0:
            mask_file_path_abs = os.path.join(config.frame_mask_dir, f'{sub_dir}/{mask_file_name}')
            if os.path.exists(mask_file_path_abs):
                if config.frame_mask_config is not None and len(config.frame_mask_config)>0:
                    top, left, height, width, = get_crop()
                    if not is_except_frame():
                        npMask = Image.open(mask_file_path_abs)
                        npMask = np.asarray(npMask)
                        h, w, c = original_img.shape
                        npMask = cv2.resize(npMask, (w, h))
                        mask = np.zeros_like(npMask)
                        mask[top:top + height, left: left + width] = npMask[top:top + height, left: left + width]
                        mask[mask > 0] = 255
                else:
                    npMask = Image.open(mask_file_path_abs)
                    npMask = np.asarray(npMask)
                    mask = npMask.copy()
                    mask[mask > 0] = 255
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:,:,-1]
        return mask
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def readFrame(inputQ: queue, config):
    image_data_dir = config.image_data_dir
    source_img_dir = config.image_png_dir
    npz_fixed_data_dir = config.fixed_npz_data_dir
    npz_data_dir = config.npz_data_dir
    start_frame = config.start_frame
    end_frame = config.end_frame
    frame_scale_config = config.frame_scale_config
    _inputCnt = 0
    if len(image_data_dir) > 0:
        file_list = get_file_list(image_data_dir)
        file_list = list(filter(lambda x: x.endswith('zip'), file_list))
    else:
        file_list = get_file_list(source_img_dir)
        file_list = list(filter(lambda x: x.endswith('png'), file_list))
    # end
    file_list.sort()
    for file in file_list:
        try:
            filepath, _ = os.path.split(file)
            filename, suffix = os.path.splitext(_)
            if start_frame != 0 and int(filename) < start_frame:
                continue
            if end_frame != 0 and int(filename) > end_frame:
                break
            if suffix == '.zip':
                original_img_name = 'original.png'
                try:
                    with zipfile.ZipFile(file, 'r') as myzip:
                        if original_img_name not in myzip.namelist():
                            logger.warning(f'{original_img_name} not in file list in :{file}！please check the file')
                        else:
                            original_img = myzip.open(original_img_name)
                            original_img = np.array(Image.open(original_img))
                            frame = mu.MediaFrame.fromRgbArray(original_img, mu.PixelFormat.RGB24)
                except Exception as e:
                    logger.warning(f'read original.png from:{file} failed! please check! Exception:{e}')
            else:
                vin1 = MediaSourceVideoInput(file)
                frame = vin1.getOneFrame(asFormat=mu.PixelFormat.RGB24)
            if frame is not None:
                frame.present_time = int(filename)
                # find npz fixed data
                npz_file_path = find_npz_fixed_data(filename, npz_fixed_data_dir)
                if npz_file_path is None:
                    # find npz data path
                    if suffix == '.zip':
                        sub_dir_name = str(os.path.dirname(file).split('/')[-1])
                    else:
                        sub_dir_name = ''
                    npz_file_name = f'{filename}.npz'
                    npz_file_path = os.path.join(os.path.join(npz_data_dir, sub_dir_name), npz_file_name)
                # end
                npDepth = None
                if os.path.isfile(npz_file_path):
                    try:
                        depth_data = np.load(npz_file_path)
                        if 'depth' in depth_data:
                            npDepth = depth_data['depth']
                        else:
                            raise BaseException
                    except Exception as e:
                        logger.error(f'read npz_file :{npz_file_path} error ,please check it!')
                else:
                    dir_name = os.path.dirname(npz_file_path)
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                if len(image_data_dir) > 0:
                    # zip mode
                    zipfilePath = file
                else:
                    # png mode
                    base_dir = os.path.dirname(file)
                    zip_file_base_dir = os.path.join(base_dir, 'imagedata')
                    sub_dir_name = f'{int(filename) // (60 * 1000):03d}'
                    zip_file_base_dir = os.path.join(zip_file_base_dir, sub_dir_name)
                    if not os.path.exists(zip_file_base_dir):
                        os.mkdir(zip_file_base_dir)
                    zip_file_name = f'{filename}.zip'
                    zipfilePath = os.path.join(zip_file_base_dir, zip_file_name)
                depth_scale = get_scale(frame_scale_config, present_time=int(filename))
                mask = get_mask(config, original_img, present_time=int(filename))
                qElem = {
                    'index': _inputCnt,
                    'present_time': filename,
                    'decFrm': frame,
                    'zipfilePath': zipfilePath,
                    'npzfilePath': npz_file_path,
                    'depth_scale': depth_scale,
                    'mask': mask,
                }
                if npDepth is not None:
                    npDepth = np.ascontiguousarray(npDepth)
                    qElem['depth'] = npDepth
                # end
                inputQ.put(qElem)
                if config.printStatus:
                    logger.info(f'read Frame :{filename},index:{_inputCnt}')
                _inputCnt += 1
            else:
                logger.warning(f'load file :{file} to frame failed')
        except Exception:
            import traceback
            traceback.print_exc()
    inputQ.put('EOF')
    logger.info('End Read Frame')
    # end
