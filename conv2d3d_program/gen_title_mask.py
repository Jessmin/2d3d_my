from xzh_mediaunit.utils import MediaSourceVideoInput
import xzh_mediaunit as mu
import numpy as np
from PIL import Image
import os
import cv2
import argparse
import zipfile
import easyocr

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-dir', dest='dataBaseDir', required=True, type=str)
parser.add_argument('--start-frame', dest='startFrame', type=int, required=True)
parser.add_argument('--end-frame', dest='endFrame', type=int, required=True)
parser.add_argument('--crop', dest='crop', default='840,0,120,1920')
parser.add_argument('--dilate', dest='dilate', type=int, default=8)
args = parser.parse_args()

reader = easyocr.Reader(['ch_sim', 'en'])


def get_img(path):
    vin1 = MediaSourceVideoInput(path)
    frmSrc = vin1.getOneFrame(asFormat=mu.PixelFormat.RGB24)
    npFrame = np.asarray(frmSrc)
    return npFrame


def get_text_roi(reader, img):
    text_roi = np.zeros_like(img)
    horizontal_list, free_list = reader.detect(img)
    row_first = []
    row_second = []
    distance_y_threshold = 50
    for _ in horizontal_list:
        cur_x_min, cur_x_max, cur_y_min, cur_y_max = _
        if len(row_first) > 0 and len(row_second) == 0:
            x_min, x_max, y_min, y_max = row_first
            distance_y = cur_y_min - y_min
            if distance_y > distance_y_threshold:
                row_second = _
            else:
                if cur_x_min < x_min:
                    x_min = cur_x_min
                if cur_x_max > x_max:
                    x_max = cur_x_max
                if cur_y_min < y_min:
                    y_min = cur_y_min
                if cur_y_max > y_max:
                    y_max = cur_y_max
                row_first = [x_min, x_max, y_min, y_max]
        else:
            row_first = _
        if len(row_second) > 0:
            x_min, x_max, y_min, y_max = row_second
            if cur_x_min < x_min:
                x_min = cur_x_min
            if cur_x_max > x_max:
                x_max = cur_x_max
            if cur_y_min < y_min:
                y_min = cur_y_min
            if cur_y_max > y_max:
                y_max = cur_y_max
            row_second = [x_min, x_max, y_min, y_max]
    bbox = [row_first, row_second]
    for box in bbox:
        if len(box) > 0:
            x_min, x_max, y_min, y_max = box
            text_roi[y_min:y_max, x_min: x_max] = 1
        # img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return text_roi

def gen_mask_withDB(img:np.ndarray,crop:str,iterations:int = 8):
    subtitle_mask = np.zeros_like(img)
    top, left, height, width = crop.split(',')
    top, left, height, width = int(top), int(left), int(height), int(width)
    subtitle_mask[top:top + height, left:left + width, :] = img[top:top + height, left:left + width, :]


def gen_mask_withOCR(reader: easyocr.Reader, img: np.ndarray, crop: str, iterations: int = 8):
    subtitle_mask = np.zeros_like(img)
    top, left, height, width = crop.split(',')
    top, left, height, width = int(top), int(left), int(height), int(width)
    subtitle_mask[top:top + height, left:left + width, :] = img[top:top + height, left:left + width, :]

    # preapare for OCR
    subtitle_mask_gray = cv2.cvtColor(subtitle_mask, cv2.COLOR_BGR2GRAY)
    ret, subtitle_mask_binary = cv2.threshold(subtitle_mask_gray, 244, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=subtitle_mask_binary.dtype)
    subtitle_mask_for_ocr = cv2.dilate(subtitle_mask_binary, kernel, iterations=10)
    subtitle_mask_dilate = cv2.dilate(subtitle_mask_binary, kernel, iterations=iterations)

    subtitle_mask_copy = subtitle_mask.copy()
    subtitle_mask_copy = cv2.cvtColor(subtitle_mask_copy, cv2.COLOR_BGR2GRAY)
    subtitle_mask_copy[subtitle_mask_for_ocr < 100] = 0
    subtitle_mask_copy = cv2.dilate(subtitle_mask_copy, kernel, iterations=2)

    subtitle_mask_for_ocr = 255 - subtitle_mask_copy

    mask_text_roi = get_text_roi(reader, subtitle_mask_for_ocr)
    mask_text_roi[mask_text_roi > 255] = 255

    subtitle_mask = subtitle_mask_dilate * mask_text_roi
    if subtitle_mask.sum() == 0:
        return None
    return subtitle_mask.astype(np.uint8)


def read_npz_data(npz_file_path: str):
    data = np.load(npz_file_path)
    if 'depth' in data:
        return data['depth']
    return None


def gen_new_depth(np_depth: np.ndarray, mask: np.ndarray):
    h, w = np_depth.shape
    mask = cv2.resize(mask, (w, h))
    np_weight = mask / 255.
    np_depth_fixed = np.multiply(np_depth, 1 - np_weight) + np_weight * args.titleDepth
    return np_depth_fixed


def depth_to_display(np_depth):
    display_img = 255 * (np_depth - np_depth.min()) / (np_depth.max() - np_depth.min())
    return display_img.astype(np.uint8)


if __name__ == '__main__':
    if not os.path.exists(args.dataBaseDir):
        raise NotADirectoryError(f'{args.dataBaseDir} not Found')
    image_data_dir = os.path.join(args.dataBaseDir, 'imagedata')
    npz_data_dir = os.path.join(args.dataBaseDir, 'npzData')
    out_mask_dir = os.path.join(args.dataBaseDir, f'fixedSubtitleData-dilate{args.dilate}-test')

    total_frame = (args.endFrame - args.startFrame) / 40
    print(f'totalFrame:{total_frame}')

    if not os.path.exists(out_mask_dir):
        os.mkdir(out_mask_dir)
    # end
    frameinfoFile = os.path.join(image_data_dir, 'frameinfo.txt')
    frameinfo = open(frameinfoFile, mode='r')
    outFrame = 0
    while True:
        line = frameinfo.readline()
        if line == '':
            break
        present_time, pts = line.replace('\n', '').split(',')
        if args.startFrame > int(present_time):
            continue
        if args.endFrame != 0 and args.endFrame < int(present_time):
            break
        zip_file_name = f'{int(present_time):08d}.zip'
        npz_file_name = f'{int(present_time):08d}.npz'
        out_maks_name = f'{int(present_time):08d}.png'
        sub_dir = f'{(int(present_time) // (1000 * 60)):03d}'
        zip_file_path_abs = os.path.join(image_data_dir, f'{sub_dir}/{zip_file_name}')
        npz_file_path_abs = os.path.join(npz_data_dir, f'{sub_dir}/{npz_file_name}')
        out_mask_path_abs = os.path.join(out_mask_dir, f'{sub_dir}/{out_maks_name}')
        if not os.path.exists(os.path.dirname(out_mask_path_abs)):
            os.mkdir(os.path.dirname(out_mask_path_abs))
        if not os.path.exists(zip_file_path_abs) or not os.path.exists(npz_file_path_abs):
            print(f'could not find file:{zip_file_path_abs}')
            continue
        try:
            with zipfile.ZipFile(zip_file_path_abs, 'r') as myzip:
                if 'original.png' not in myzip.namelist():
                    print(f'original.png not in file list in :{zip_file_path_abs}！please check the file')
                    continue
                original = myzip.open('original.png')
                original = np.asarray(Image.open(original))
        except:
            import traceback

            traceback.print_exc()
            print(f'{zip_file_path_abs} error')
            continue
        # end
        np_depth = read_npz_data(npz_file_path_abs)
        if np_depth is None:
            print(f'depth in None')
            continue
        mask = gen_mask_withOCR(reader, original, crop=args.crop, iterations=args.dilate)
        if os.path.exists(out_mask_path_abs):
            os.remove(out_mask_path_abs)
        if mask is not None:
            cv2.imwrite(out_mask_path_abs, mask)
        outFrame += 1
        print(f'outFrame:{outFrame} progress:{(100 * (outFrame / total_frame)):.2f}%', end='\r')
    print('All Done!')
