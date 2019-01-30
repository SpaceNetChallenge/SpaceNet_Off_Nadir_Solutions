from os import path, listdir, makedirs
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from tqdm import tqdm

pred_folders = [
    'pred_50_9ch_fold_0_0',
    'pred_50_9ch_fold_1_0',
    'pred_50_9ch_fold_2_0',
    'pred_50_9ch_fold_3_0',
    'pred_50_9ch_fold_4_0',
    'pred_50_9ch_fold_5_0',
    'pred_50_9ch_fold_6_0',
    'pred_50_9ch_fold_7_0',
    'pred_92_9ch_fold_0_0',
    'pred_92_9ch_fold_1_0',
    'pred_92_9ch_fold_2_0',
    'pred_92_9ch_fold_3_0',
    'pred_92_9ch_fold_4_0',
    'pred_92_9ch_fold_5_0',
    'pred_92_9ch_fold_6_0',
    'pred_92_9ch_fold_7_0',
    'pred_154_9ch_fold_0_0',
    'pred_154_9ch_fold_1_0',
    'pred_154_9ch_fold_2_0',
    'pred_154_9ch_fold_3_0',
    'pred_154_9ch_fold_4_0',
    'pred_154_9ch_fold_5_0',
    'pred_154_9ch_fold_6_0',
    'pred_154_9ch_fold_7_0',
    'pred_101_9ch_fold_0_0',
    'pred_101_9ch_fold_1_0',
    'pred_101_9ch_fold_2_0',
    'pred_101_9ch_fold_3_0'
]

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs('/wdata/merged_pred', exist_ok=True)

    for fid in tqdm(listdir(path.join('/wdata', pred_folders[0]))):
        used_msks = []

        for pr_f in pred_folders:
            msk1 = cv2.imread(path.join('/wdata', pr_f, '{0}.png'.format(fid.split('.')[0])), cv2.IMREAD_UNCHANGED)
            used_msks.append(msk1)

        msk = np.zeros_like(used_msks[0], dtype='float')

        for i in range(len(pred_folders)):
            p = used_msks[i]
            msk += p.astype('float')
        msk /= len(used_msks)
        cv2.imwrite(path.join('/wdata/merged_pred', fid), msk.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))