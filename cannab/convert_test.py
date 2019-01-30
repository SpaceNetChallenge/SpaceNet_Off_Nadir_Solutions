import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
import gc
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from tqdm import tqdm
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import measure, io

from shapely.wkt import loads

test_png = '/wdata/test_png'
test_png2 = '/wdata/test_png_5_3_0'
test_png3 = '/wdata/test_png_pan_6_7'
test_dir = '/data/test'

if len(sys.argv) > 0:
    test_dir = sys.argv[1]

threshold = 3000

def process_image(img_id):
    if 'Pan-Sharpen_' in img_id:
        img_id = img_id.split('Pan-Sharpen_')[1]
    img = io.imread(path.join(test_dir, '_'.join(img_id.split('_')[:4]), 'Pan-Sharpen', 'Pan-Sharpen_' + img_id+'.tif'))
    nir = img[:, :, 3:]
    img = img[:, :, :3]
    np.clip(img, None, threshold, out=img)
    img = np.floor_divide(img, threshold / 255).astype('uint8')
    cv2.imwrite(path.join(test_png, img_id + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    img2 = io.imread(path.join(test_dir, '_'.join(img_id.split('_')[:4]), 'MS', 'MS_' + img_id+'.tif'))
    img2 = np.rollaxis(img2, 0, 3)
    img2 = cv2.resize(img2, (900, 900), interpolation=cv2.INTER_LANCZOS4)
    
    img_0_3_5 = (np.clip(img2[..., [0, 3, 5]], None, (2000, 3000, 3000)) / (np.array([2000, 3000, 3000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(test_png2, img_id + '.png'), img_0_3_5, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    pan = io.imread(path.join(test_dir, '_'.join(img_id.split('_')[:4]), 'PAN', 'PAN_' + img_id+'.tif'))
    pan = pan[..., np.newaxis]
    img_pan_6_7 = np.concatenate([pan, img2[..., 7:], nir], axis=2)
    img_pan_6_7 = (np.clip(img_pan_6_7, None, (3000, 5000, 5000)) / (np.array([3000, 5000, 5000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(test_png3, img_id + '.png'), img_pan_6_7, [cv2.IMWRITE_PNG_COMPRESSION, 9])

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(test_png, exist_ok=True)
    makedirs(test_png2, exist_ok=True)
    makedirs(test_png3, exist_ok=True)

    all_ids0 = []
    for d in listdir(test_dir):
        for f in listdir(path.join(test_dir, d, 'Pan-Sharpen')):
            if '.tif' in f:
                all_ids0.append(f.split('.tif')[0])
    
    with Pool() as pool:
        _ = pool.map(process_image, all_ids0)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))