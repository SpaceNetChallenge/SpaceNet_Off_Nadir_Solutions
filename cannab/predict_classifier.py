# -*- coding: utf-8 -*-
from os import path, listdir, mkdir, makedirs
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from skimage.color import label2rgb
from tqdm import tqdm
from multiprocessing import Pool
import lightgbm as lgb
from train_classifier import get_inputs
import pandas as pd

test_pred_folder = '/wdata/merged_pred'

lgbm_models_folder = '/wdata/lgbm_models'

out_pred = '/wdata/lgbm_pred'

DATA_THREADS = 4 #for p2.xlarge instance

num_split_iters = 5
folds_count = 5

sep_count = 3

best_thr = [0.25, 0.25, 0.2]
            
if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(out_pred, exist_ok=True)
            
    all_files = []
    inputs = []
    outputs = []
    inputs2 = []
    outputs2 = []
    labels = []
    labels2 = []
    separated_regions = []
    fns = []
    paramss = []
    
    gbm_models = []
    
    for it in range(num_split_iters):
        for it2 in range(folds_count):
            gbm_models.append(lgb.Booster(model_file=path.join(lgbm_models_folder, 'gbm_model_{0}_{1}.txt'.format(it, it2))))
    
    inputs = []
    paramss = []
    used_ids = []
    all_nadir_idxs = []
    for f in tqdm(listdir(test_pred_folder)):
        if '.png' in f and 'nadir' in f:
            tmp = f.split('_')

            nadir = int(tmp[1].split('nadir')[1])

            nad_idx = 0
            if nadir > 40:
                nad_idx = 2
            elif nadir > 25:
                nad_idx = 1
            all_nadir_idxs.append(nad_idx)

            paramss.append((f, test_pred_folder, [nadir], True, None))
            used_ids.append(f)
    
    inputs = []
    inputs2 = []
    labels= []
    labels2 = []
    separated_regions= []
    with Pool(processes=DATA_THREADS) as pool:
        results = pool.starmap(get_inputs, paramss, len(paramss)//DATA_THREADS)
    for i in range(len(results)):
        inp, lbl, inp2, lbl2, sep_regs = results[i]
        inputs.append(inp)
        inputs2.append(inp2)
        labels.append(lbl)
        labels2.append(lbl2)
        separated_regions.append(sep_regs)

        
    print('Predicting...')

    new_test_ids = []
    rles = []

    bst_k = np.zeros((sep_count+1))
    removed = 0
    replaced = 0
    total_cnt = 0
    im_idx = 0
    
    non_empty_cnt = 0
    
    for f in tqdm(used_ids):
        if path.isfile(path.join(test_pred_folder, f)) and '.png' in f:
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
                        if pred2[max_sep][j] > best_thr[all_nadir_idxs[im_idx]]:
                            pred_labels[labels2[im_idx][max_sep] == j+1] = clr
                            clr += 1
                        else:
                            removed += 1
                else:
                    if pred[i] > best_thr[all_nadir_idxs[im_idx]]:
                        pred_labels[labels[im_idx] == i+1] = clr
                        clr += 1
                    else:
                        removed += 1
                bst_k[max_sep+1] += 1
                
            total_cnt += pred_labels.max()
    
            cv2.imwrite(path.join(out_pred, f.replace('.png', '.tif')), pred_labels)
    
            im_idx += 1

    print('total_cnt', total_cnt, 'removed', removed, 'replaced', replaced, 'not empty:', non_empty_cnt)
    print(bst_k)
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))