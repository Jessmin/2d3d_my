import json
import os
import zipfile
from PIL import Image
import numpy as np
import torch


class Config(object):
    def __init__(self):
        self.image_data_dir = ""
        self.image_png_dir = ""
        self.fixed_npz_data_dir = ""
        self.npz_data_dir = ""
        self.start_frame = 0
        self.end_frame = 0
        self.crop = None
        self.max_disparity = 40
        self.disparity_offset = 7
        self.frame_scale_config = None
        self.batch_size_per_gpu = 4
        self.inpaint_threads = 1
        self.calc_depth_width = 512
        self.depth_mode = 0
        self.output_mono_width = None
        self.output_mono_scale = 0
        self.use_float16 = False
        self.model_name = 'midas'
        self.use_dilate = '3,2'
        self.use_gf = '2,50'
        self.bottom_offset = 10
        # extra
        self.output_mono_size = None
        self.calc_depth_size = None
        self.calc_depth_batch_size = 0
        self.printStatus = True
        self.frame_mask_config = ''
        self.frame_mask_dir = ''
        self.frame_split_info = ''
        self.subtitle_depth = 0
        self.depthInfo = ''
        self.use_gpu_ids = '0'


class FrameScaleConfig(object):
    def __init__(self):
        self.start_frame = None
        self.end_frame = None
        self.scale = None


def get_frame_shape(config: Config):
    if config.image_data_dir is not None and len(config.image_data_dir) > 0:
        sub_dirs = os.listdir(config.image_data_dir)
        if len(sub_dirs) < 1:
            raise ValueError(f'image_data_dir:{config.image_data_dir} is a Empty Dir')
        sub_dirs = [
            x for x in sub_dirs if os.path.isdir(os.path.join(config.image_data_dir, x))
        ]
        for sub_dir in sub_dirs:
            sub_dir_abs = os.path.join(config.image_data_dir, sub_dir)
            zip_files = os.listdir(sub_dir_abs)
            zip_files = [x for x in zip_files if x.endswith('.zip')]
            for zip_file in zip_files:
                zip_file_path_abs = os.path.join(sub_dir_abs, zip_file)
                original_img_name = 'original.png'
                with zipfile.ZipFile(zip_file_path_abs, 'r') as myzip:
                    if original_img_name in myzip.namelist():
                        original_img = myzip.open(original_img_name)
                        original_img = np.array(Image.open(original_img))
                        h, w, c = original_img.shape
                        return h, w
        raise ValueError(f'image_data_dir:{config.image_data_dir} is a Empty Dir')
    elif len(config.image_png_dir) > 0:
        # end with png
        files = os.listdir(config.image_png_dir)
        files = [x for x in files if x.endswith('png')]
        if len(files) < 1:
            raise FileNotFoundError(f'source_img_dir :{config.image_png_dir} is a Empty Dir')
        img = np.asarray(Image.open(os.path.join(config.image_png_dir, files[0])))
        h, w, c = img.shape
        return h, w
    else:
        raise ValueError(
            f'image_data_dir:{config.image_data_dir} and source_img_dir :{config.image_png_dir} are empty')


def getConfig(args):
    config = Config()
    specificConfig_arr = []
    if args.frame_scale_config is not None and len(args.frame_scale_config) > 0:
        if not os.path.exists(args.frame_scale_config):
            raise FileNotFoundError(f'frame_scale_config file :{args.frame_scale_config} does not exits!')
        with open(args.frame_scale_config, 'r') as f:
            frame_scale_config_json = json.load(f)
            specific_config = frame_scale_config_json['specific_config']
            for i in range(len(specific_config)):
                specificConfig = FrameScaleConfig()
                specificConfig.__dict__ = specific_config[i]
                specificConfig_arr.append(specificConfig)
    config.frame_scale_config = np.asarray(specificConfig_arr)

    if args.frame_mask_config is not None and len(args.frame_mask_config) > 0:
        if not os.path.exists(args.frame_mask_config):
            raise FileNotFoundError(f'frame_mask_config:{args.frame_mask_config} does not exits!')
        f = open(args.frame_mask_config)
        config.frame_mask_config = json.load(f)
        crop = config.frame_mask_config['crop'].split(',')
        crop = [int(x) for x in crop]
        if len(crop) != 4:
            raise ValueError(f'config file{crop} is INVAILD')
    else:
        config.frame_mask_config = None
    if args.frame_mask_dir is not None and len(args.frame_mask_dir) > 0:
        if not os.path.exists(args.frame_mask_dir):
            raise FileNotFoundError(f'frame_mask_dir:{args.frame_mask_dir} does not exits!')
        config.frame_mask_dir = args.frame_mask_dir

    config.crop = args.crop
    config.image_data_dir = args.image_data_dir
    config.image_png_dir = args.image_png_dir
    config.npz_data_dir = args.npz_data_dir
    config.start_frame = args.start_frame
    config.end_frame = args.end_frame
    config.max_disparity = args.max_disparity
    config.disparity_offset = args.disparity_offset
    config.fixed_npz_data_dir = args.fixed_npz_data_dir
    config.output_mono_scale = args.output_mono_scale
    config.frame_split_info = args.frame_split_info

    if config.npz_data_dir is None:
        config.npz_data_dir = ''
    if config.image_data_dir is None:
        config.image_data_dir = ''
    if config.fixed_npz_data_dir is None:
        config.fixed_npz_data_dir = ''

    config.depth_mode = args.depth_mode
    config.batch_size_per_gpu = args.batch_size_per_gpu
    config.inpaint_threads = args.inpaint_threads
    config.calc_depth_width = args.calc_depth_width
    config.output_mono_width = args.output_mono_width
    config.use_float16 = args.use_float16
    config.model_name = args.model_name
    config.use_dilate = args.use_dilate
    config.use_gf = args.use_gf
    config.bottom_offset = args.bottom_offset
    config.subtitle_depth = args.subtitle_depth
    config.use_gpu_ids = args.use_gpu_ids.split(',')

    # extra
    height_source, width_source = get_frame_shape(config)
    if config.crop is not None:
        crop_top, crop_height, crop_left, crop_width = config.crop
    else:
        crop_height = height_source
        crop_width = width_source
    calc_depth_height = crop_height * config.calc_depth_width // crop_width

    if config.model_name == 'midas':
        if calc_depth_height % 32 != 0:
            calc_depth_height = np.round(calc_depth_height / 32) * 32
        # end
    # end
    config.calc_depth_size = (config.calc_depth_width, int(calc_depth_height))

    # deal unsupport media
    output_mono_width = config.output_mono_width if config.output_mono_width is not None else width_source
    output_mono_height = np.ceil(height_source / width_source * output_mono_width)

    if output_mono_width % 16 != 0:
        output_mono_width = np.round(output_mono_width / 16) * 16
    config.output_mono_size = (int(output_mono_width), int(output_mono_height))
    print(f'mono:{config.output_mono_size}')
    config.calc_depth_batch_size = config.batch_size_per_gpu * torch.cuda.device_count()

    if config.image_data_dir is not None and len(config.image_data_dir) > 0:
        if len(config.npz_data_dir) == 0:
            config.npz_data_dir = os.path.join(os.path.dirname(config.image_data_dir), 'npzData')
        config.depthInfo = os.path.join(config.image_data_dir, 'depthinfo.txt')
    elif config.image_png_dir is not None and len(config.image_png_dir) > 0:
        config.npz_data_dir = config.npz_data_dir
        config.depthInfo = os.path.join(config.image_png_dir, 'depthinfo.txt')
    else:
        raise ValueError(f'image_data_dir and source_img_dir both are None')
    return config
