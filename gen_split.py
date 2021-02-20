import cv2
import numpy as np
import glob
import zipfile
from PIL import  Image
import os

def compare(frame, pre_hist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    n_pixel = frame.shape[0] * frame.shape[1]
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist * (1.0 / n_pixel)
    if pre_hist is not None:
        diff = np.sum(np.abs(np.subtract(hist, pre_hist)))
    else:
        diff = 0
    return diff, hist

def read_png(zipfilepath:str):
    with zipfile.ZipFile(zipfilepath,'r') as z:
        entryNames = z.namelist()
        if 'original.png' in entryNames:
            original = z.open('original.png')
            original = np.asarray(Image.open(original))
            return original
        return None
        #end



if __name__ == '__main__':
    # main_dir = '/home/zhaohoj/development_sshfs/2d3d/20210121/4.八佰片段/imagedata'
    main_dir = '/home/zhaohoj/development_sshfs/2d3d/龙门客栈/新龙门客栈-全片/imagedata'
    start_pts = 1482000 
    end_pts = 1782000
    def filter_file(fs:str):
        fs_filtered =[]
        for filepath in fs:
            _,filename = os.path.split(filepath)
            pts,_ = os.path.splitext(filename)
            if int(pts)<start_pts:
                continue
            if end_pts > 0 and int(pts)>end_pts:
                break
            fs_filtered.append(filepath)
        return fs_filtered

    fs = glob.glob(f'{main_dir}/**/*.zip')
    fs.sort()
    fs = filter_file(fs)
    pre_hist = None
    import matplotlib.pyplot as plt 
    diffs =[]
    filenames = []
    for f in fs:
        _,filename = os.path.split(f)
        filename,_ = os.path.splitext(filename)
        original = read_png(f)
        diff,hist = compare(original,pre_hist)
        pre_hist = hist
        diffs.append(diff)
        filenames.append(filename)
    diffs  = np.asarray(diffs)
    diffs_mean = np.mean(diffs)
    for i in range(len(diffs)):
        d = diffs[i]
        if d>diffs_mean*2 :
            with open('frame_split.txt','a') as f:
                f.write(f'{filenames[i]}\n')    
