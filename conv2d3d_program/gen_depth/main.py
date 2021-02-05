import logging

logging.basicConfig(
    format='[%(asctime)-15s | %(name)-8s | %(threadName)-12s | %(funcName)-16s | %(lineno)-5d][%(levelname)8s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
from core import Conv2d3dCore
from config import getConfig

argParser = argparse.ArgumentParser()
argParser.add_argument('--image-data-dir', dest='image_data_dir', help='image data dir')
argParser.add_argument('--image-png-dir', dest='image_png_dir', help='png dir ')
argParser.add_argument('--fixed-npz-data-dir', dest='fixed_npz_data_dir', help='fixed npz data')
argParser.add_argument('--frame-split-info',dest= 'frame_split_info',help = 'frame split info')
argParser.add_argument('--npz-data-dir', dest='npz_data_dir')
argParser.add_argument('--depth-mode', dest='depth_mode', default=1, help='0:gf->dilate 1:dilate->gf')
argParser.add_argument('--start-frame', dest='start_frame', default=0, type=int)
argParser.add_argument('--end-frame', dest='end_frame', default=0, type=int)
argParser.add_argument('--max-disparity', dest='max_disparity', type=float, default=40.0)
argParser.add_argument('--disparity-offset', dest='disparity_offset', type=float, default=14.0)
argParser.add_argument('--batch-size-per-gpu', dest='batch_size_per_gpu', type=int, default=1)
argParser.add_argument('--inpaint-threads', dest='inpaint_threads', type=int, default=4)
argParser.add_argument('--calc-depth-width', dest='calc_depth_width', type=int, default=512)
argParser.add_argument('--output-mono-width', dest='output_mono_width', type=int)
argParser.add_argument('--output-mono-scale', dest='output_mono_scale', choices=[0, 1], default=0)
argParser.add_argument('--use-float16', dest='use_float16', type=bool, default=False)
argParser.add_argument('--model-name', choices=['midas', 'mannequin'], dest='model_name', default='midas')
argParser.add_argument("--crop", type=str, dest='crop')
argParser.add_argument('--dilate', dest='use_dilate', default='3,2', action='store_true')
argParser.add_argument('--useGF', dest='use_gf', default='2,50', type=str)
argParser.add_argument('--use-gpu-ids', dest='use_gpu_ids', default='0', type=str)
argParser.add_argument('--bottom-offset', dest='bottom_offset', default=0, type=int)
argParser.add_argument('--frame-scale-config', dest='frame_scale_config', default='', type=str)
argParser.add_argument('--frame-mask-config', dest='frame_mask_config', default='', type=str)
argParser.add_argument('--frame-mask-dir', dest='frame_mask_dir', default='', type=str)
argParser.add_argument('--subtitle-depth', dest='subtitle_depth', default=10000)
args = argParser.parse_args()

def checkArgs():
    if args.use_gf is not None:
        try:
            arr = args.use_gf.split(',')
            if len(arr) != 2:
                raise ValueError(f"useGF:{args.use_gf}is INVALID")
            R = int(arr[0])
            E = float(arr[1])
            args.use_gf = (R, E)
        except Exception:
            raise ValueError(f"useGF:{args.use_gf}is INVALID")
    if args.crop is not None:
        try:
            crops = args.crop.split(',')
            if len(crops) != 4:
                raise ValueError(f'args.crop:{args.crop} is INVALID')
            crops = [int(x) for x in crops]
            top, height, left, width = tuple(crops)
            args.crop = (top, height, left, width)
        except Exception:
            raise ValueError(f'args.crop:{args.crop} is INVALID')
    # end
    if args.use_dilate is not None:
        try:
            arr = args.use_dilate.split(',')
            if len(arr) != 2:
                raise ValueError(f"use_dilate:{args.use_dilate}is INVALID")
            kernel = int(arr[0])
            iteration = int(arr[1])
            args.use_dilate = (kernel, iteration)
        except Exception:
            raise ValueError(f"use_dilate:{args.use_dilate}is INVALID")


if __name__ == '__main__':
    checkArgs()
    config = getConfig(args)
    core = Conv2d3dCore(config)
    core.wait()
    print('All Done!')