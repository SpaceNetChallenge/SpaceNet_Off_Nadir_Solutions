# -*- coding: utf-8 -*-
from os import path, listdir, mkdir

import numpy as np
from skimage.color import label2rgb

np.random.seed(1)
import random

random.seed(1)
import timeit
import cv2
import os
from multiprocessing import Pool
import lightgbm as lgb
from train_classifier import get_inputs

pred_folder = path.join('predictions')

test_pred_folder = path.join(pred_folder, 'masks/ensemble')

lgbm_models_folder = 'lgbm_models'

test_out_folders = ['lgbm_test_sub1']
color_out_folders = ['color_test_sub1']

DATA_THREADS = 8

num_split_iters = 1
folds_count = 30

pixels_threshold = 76

sep_count = 3
sep_thresholds = [0.6, 0.7, 0.8]

best_thrs = [0.3]
step_size=20

def process_images(step):
    gbm_models = []

    for it in range(num_split_iters):
        for it2 in range(folds_count):
            gbm_models.append(
                lgb.Booster(model_file=path.join(lgbm_models_folder, 'gbm_model_{0}_{1}.txt'.format(it, it2))))

    paramss = []
    files = list(reversed(sorted(listdir(test_pred_folder))))
    for filename in files[step:step + step_size]:
        if path.isfile(path.join(test_pred_folder, filename)) and '.png' in filename:
            paramss.append((filename, test_pred_folder, None))
    inputs = []
    inputs2 = []
    labels = []
    labels2 = []
    separated_regions = []
    results = [get_inputs(param[0], param[1]) for param in paramss]
    for i in range(len(results)):
        inp, lbl, inp2, lbl2, sep_regs = results[i]
        inputs.append(inp)
        inputs2.append(inp2)
        labels.append(lbl)
        labels2.append(lbl2)
        separated_regions.append(sep_regs)
    for sub_id in range(1):
        bst_k = np.zeros((sep_count + 1))
        removed = 0
        replaced = 0
        total_cnt = 0
        im_idx = 0

        empty_cnt = 0

        for filename in files[step:step + step_size]:
            if path.isfile(path.join(test_pred_folder, filename)) and '.png' in filename:
                img_id = filename

                inp = inputs[im_idx]
                pred = np.zeros((inp.shape[0]))
                pred2 = [np.zeros((inp2.shape[0])) for inp2 in inputs2[im_idx]]

                for m in gbm_models:
                    if pred.shape[0] > 0:
                        pred += m.predict(inp)
                    for k in range(len(inputs2[im_idx])):
                        if pred2[k].shape[0] > 0:
                            pred2[k] += m.predict(inputs2[im_idx][k])
                if pred.shape[0] > 0:
                    pred /= len(gbm_models)
                for k in range(len(pred2)):
                    if pred2[k].shape[0] > 0:
                        pred2[k] /= len(gbm_models)

                pred_labels = np.zeros_like(labels[im_idx], dtype='uint16')

                clr = 1

                for i in range(pred.shape[0]):
                    max_sep = -1
                    max_pr = pred[i]
                    for k in range(len(separated_regions[im_idx])):
                        if len(separated_regions[im_idx][k][i]) > 0:
                            pred_lvl2 = pred2[k][separated_regions[im_idx][k][i]]
                            if len(pred_lvl2) > 1 and pred_lvl2.mean() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.mean()
                                break
                            if len(pred_lvl2) > 1 and pred_lvl2.max() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.max()

                    if max_sep >= 0:
                        pred_lvl2 = pred2[max_sep][separated_regions[im_idx][max_sep][i]]
                        replaced += 1
                        for j in separated_regions[im_idx][max_sep][i]:
                            if pred2[max_sep][j] > best_thrs[sub_id]:
                                pred_labels[labels2[im_idx][max_sep] == j + 1] = clr
                                clr += 1
                            else:
                                removed += 1
                    else:
                        if pred[i] > best_thrs[sub_id]:
                            pred_labels[labels[im_idx] == i + 1] = clr
                            clr += 1
                        else:
                            removed += 1
                    bst_k[max_sep + 1] += 1

                clr_labels = label2rgb(pred_labels, bg_label=0)
                clr_labels *= 255
                clr_labels = clr_labels.astype('uint8')
                cv2.imwrite(path.join(pred_folder, color_out_folders[sub_id], img_id), clr_labels, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                total_cnt += pred_labels.max()

                cv2.imwrite(path.join(pred_folder, test_out_folders[sub_id], filename[:-4]+".tif"), pred_labels)
                im_idx += 1

        print('total_cnt', total_cnt, 'removed', removed, 'replaced', replaced, 'empty:', empty_cnt)
        print(bst_k)


if __name__ == '__main__':
    t0 = timeit.default_timer()
    os.makedirs(pred_folder, exist_ok=True)
    for f in test_out_folders:
        if not path.isdir(path.join(pred_folder, f)):
            mkdir(path.join(pred_folder, f))

    for f in color_out_folders:
        if not path.isdir(path.join(pred_folder, f)):
            mkdir(path.join(pred_folder, f))



    steps = [step for step in range(0, 940, step_size)]
    with Pool(processes=DATA_THREADS) as pool:
        results = pool.map(process_images, steps)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
