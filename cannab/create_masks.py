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
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import io

from shapely.wkt import loads

labels_dir = '/wdata/labels'
masks_dir = '/wdata/masks'
occluded_masks_dir = '/wdata/masks_occluded'
train_png = '/wdata/train_png'
train_png2 = '/wdata/train_png_5_3_0'
train_png3 = '/wdata/train_png_pan_6_7'

train_dir = '/data/training'
if len(sys.argv) > 0:
    train_dir = sys.argv[1]

threshold = 3000

def mask_for_polygon(poly, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

dfs = {}
for f in listdir(path.join(train_dir, 'summaryData')):
    if '.csv' in f:
        df = pd.read_csv(path.join(train_dir, 'summaryData', f))
        dfs[f.split('_')[3]] = df

def process_image(img_id):
    img_id0 = img_id
    if 'Pan-Sharpen_' in img_id:
        img_id = img_id.split('Pan-Sharpen_')[1]

    img = io.imread(path.join(train_dir, '_'.join(img_id.split('_')[:4]), 'Pan-Sharpen', 'Pan-Sharpen_' + img_id+'.tif'))
    nir = img[:, :, 3:].copy()
    img = img[:, :, :3]
    np.clip(img, None, threshold, out=img)
    img = np.floor_divide(img, threshold / 255).astype('uint8')
    cv2.imwrite(path.join(train_png, img_id + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    img2 = io.imread(path.join(train_dir, '_'.join(img_id.split('_')[:4]), 'MS', 'MS_' + img_id+'.tif'))
    img2 = np.rollaxis(img2, 0, 3)
    img2 = cv2.resize(img2, (900, 900), interpolation=cv2.INTER_LANCZOS4)
    
    img_0_3_5 = (np.clip(img2[..., [0, 3, 5]], None, (2000, 3000, 3000)) / (np.array([2000, 3000, 3000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(train_png2, img_id + '.png'), img_0_3_5, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    pan = io.imread(path.join(train_dir, '_'.join(img_id.split('_')[:4]), 'PAN', 'PAN_' + img_id+'.tif'))
    pan = pan[..., np.newaxis]
    img_pan_6_7 = np.concatenate([pan, img2[..., 7:], nir], axis=2)
    img_pan_6_7 = (np.clip(img_pan_6_7, None, (3000, 5000, 5000)) / (np.array([3000, 5000, 5000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(train_png3, img_id + '.png'), img_pan_6_7, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
    df = dfs[img_id.split('_')[3]]
    
    vals = df[(df['ImageId'] == img_id0)][['BuildingId', 'PolygonWKT_Pix', 'Occluded']].values
    labels = np.zeros((900, 900), dtype='uint16')
    occluded_mask = np.zeros((900, 900), dtype='uint8')
    cur_lbl = 0
    for i in range(vals.shape[0]):
        if vals[i, 0] >= 0:
            cur_lbl += 1
            msk = mask_for_polygon(loads(vals[i, 1]), (900, 900))
            labels[msk > 0] = cur_lbl
            if vals[i, 2] == 1:
                occluded_mask[msk > 0] = 255
    
    cv2.imwrite(path.join(labels_dir, img_id + '.tif'), labels)
    cv2.imwrite(path.join(occluded_masks_dir, img_id + '.png'), occluded_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    msk = np.zeros((900, 900), dtype='uint8')
    
    if cur_lbl > 0:
        border_msk = np.zeros_like(labels, dtype='bool')
        for l in range(1, labels.max() + 1):
            tmp_lbl = labels == l
            _k = square(3)
            tmp = erosion(tmp_lbl, _k)
            tmp = tmp ^ tmp_lbl
            border_msk = border_msk | tmp

        tmp = dilation(labels > 0, square(9))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = tmp | border_msk
        tmp = dilation(tmp, square(9))
        
        msk0 = labels > 0

        msk1 = np.zeros_like(labels, dtype='bool')
        
        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    sz = 3
                else:
                    sz = 1
                uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
        
        msk1 = 255 * msk1
        msk1 = msk1.astype('uint8')

        msk0 = 255 * msk0
        msk0 = msk0.astype('uint8')

        msk2 = 255 * border_msk
        msk2 = msk2.astype('uint8')
        msk = np.stack((msk0, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)

    cv2.imwrite(path.join(masks_dir, img_id + '.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(labels_dir, exist_ok=True)
    makedirs(masks_dir, exist_ok=True)
    makedirs(occluded_masks_dir, exist_ok=True)
    makedirs(train_png, exist_ok=True)
    makedirs(train_png2, exist_ok=True)
    makedirs(train_png3, exist_ok=True)

    all_ids0 = []
    for k in dfs:
        all_ids0 += pd.unique(dfs[k]['ImageId']).tolist()

    with Pool() as pool:
        _ = pool.map(process_image, all_ids0)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))