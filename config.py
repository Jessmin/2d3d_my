class Config(object):
    def __init__(self):
        self.input_file = ''
        self.calc_depth_width = 512
        self.output_mono_width = None
        self.use_gpu_ids = '0'
        self.calc_depth_batch_size = 1
        self.maxDisparity = 40
        self.dispOffset = 14
        self.output_file = 'testout.mp4'


def getConfig(args):
    config = Config()
    # check Args
    if args.input_file == '':
        raise ValueError(f'input_file')
    if args.output_file == '':
        raise ValueError(f'output_file')
    config.input_file = args.input_file
    config.output_file = args.output_file
