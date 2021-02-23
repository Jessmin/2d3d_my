import time
from core import Conv2d3dCore
import argparse
from config import getConfig

argParser = argparse.ArgumentParser()
argParser.add_argument('--i', 'input_file')
argParser.add_argument('--o', 'output_file')
args = argParser.parse_args()

if __name__ == '__main__':
    t0 = time.time()
    config = getConfig(args)
    core = Conv2d3dCore(config)
    core.wait()
    t1 = time.time()
    print('All Done!')
    print(f'Total use Time:{t1 - t0}')
