# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
import pandas as pd
from multiprocessing import Pool

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

df = pd.read_csv('train_folds.csv')

pred_folders = [
    'pred_50_9ch_oof_0',
    'pred_92_9ch_oof_0',
    'pred_154_9ch_oof_0',
    'pred_101_9ch_oof_0'
]

coefs = [2, 2, 2, 1]

def process_image(fid):
    fid = fid + '.png'

    used_msks = []

    for pr_f in pred_folders:
        msk1 = cv2.imread(path.join('/wdata/', pr_f, '{0}.png'.format(fid.split('.')[0])), cv2.IMREAD_UNCHANGED)
        used_msks.append(msk1)

    msk = np.zeros_like(used_msks[0], dtype='float')

    for i in range(len(pred_folders)):
        p = used_msks[i]
        msk += (coefs[i] * p.astype('float'))
    msk /= np.sum(coefs)

    cv2.imwrite(path.join('/wdata/merged_oof', fid), msk.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs('/wdata/merged_oof', exist_ok=True)

    val_files = df[df['fold'] < 8]['id'].values
    
    with Pool() as pool:
        results = pool.map(process_image, val_files)
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))