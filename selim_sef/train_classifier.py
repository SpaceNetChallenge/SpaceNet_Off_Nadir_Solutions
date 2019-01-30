# -*- coding: utf-8 -*-
import os
from os import path, mkdir

import numpy as np

np.random.seed(1)
import random

random.seed(1)
import timeit
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from tqdm import tqdm
from skimage import measure
from multiprocessing import Pool
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.neighbors import KDTree
from skimage.morphology import watershed
import warnings
warnings.filterwarnings("ignore")
pred_folder = 'oof/masks'
masks_folder = "train_labels/labels"


lgbm_models_folder = 'lgbm_models'
train_pred_folder = os.path.join(pred_folder, "ensemble")

DATA_THREADS = 16
LGBM_THREADS = 12

num_split_iters = 1
folds_count = 30

pixels_threshold = 76

sep_count = 3
sep_thresholds = [0.6, 0.7, 0.8]


def get_nadir(f):
    return  int(f.split("_")[2][5:])


def get_inputs(filename, pred_folder, truth_folder=None):
    inputs = []
    pred = cv2.imread(path.join(pred_folder, filename), cv2.IMREAD_COLOR)

    pred_msk = pred / 255.
    pred_msk = pred_msk[..., 0] * (1 - pred_msk[..., 2])
    pred_msk = 1 * (pred_msk > 0.7)

    pred_msk = pred_msk.astype(np.uint8)

    y_pred = measure.label(pred_msk, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < 20:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (255 - pred[..., 0])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=((pred[..., 0] > pixels_threshold)), watershed_line=True)

    props = measure.regionprops(y_pred)

    for i in range(len(props)):
        if props[i].area < 50:
            y_pred[y_pred == i + 1] = 0

    pred_labels = measure.label(y_pred, neighbors=8, background=0)
    pred_props = measure.regionprops(pred_labels)

    #    img = cv2.imread(path.join(img_folder, filename), cv2.IMREAD_COLOR)

    coords = [pr.centroid for pr in pred_props]
    if len(coords) > 0:
        t = KDTree(coords)
        neighbors100 = t.query_radius(coords, r=50)
        neighbors200 = t.query_radius(coords, r=100)
        neighbors300 = t.query_radius(coords, r=150)
        neighbors400 = t.query_radius(coords, r=200)
    med_area = np.median(np.asarray([pr.area for pr in pred_props]))

    lvl2_labels = [np.zeros_like(pred_labels, dtype=np.int) for i in range(sep_count)]

    separated_regions = [[] for i in range(sep_count)]
    main_regions = [[] for i in range(sep_count)]

    for i in range(len(pred_props)):
        msk_reg = pred_labels[pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]] == i + 1
        pred_reg = pred[pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]]

        contours = cv2.findContours((msk_reg * 255).astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[1]) > 0:
            cnt = contours[1][0]
            min_area_rect = cv2.minAreaRect(cnt)

        inp = []
        nadir = get_nadir(filename)
        inp.append(nadir)
        inp.append(pred_props[i].area)
        inp.append(0)
        if len(contours[1]) > 0:
            inp.append(cv2.isContourConvex(cnt) * 1.0)
            inp.append(min(min_area_rect[1]))
            inp.append(max(min_area_rect[1]))
            if max(min_area_rect[1]) > 0:
                inp.append(min(min_area_rect[1]) / max(min_area_rect[1]))
            else:
                inp.append(0)
            inp.append(min_area_rect[2])
        else:
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
        inp.append(pred_props[i].convex_area)
        inp.append(pred_props[i].solidity)
        inp.append(pred_props[i].eccentricity)
        inp.append(pred_props[i].extent)
        inp.append(pred_props[i].perimeter)
        inp.append(pred_props[i].major_axis_length)
        inp.append(pred_props[i].minor_axis_length)
        if (pred_props[i].minor_axis_length > 0):
            inp.append(pred_props[i].minor_axis_length / pred_props[i].major_axis_length)
        else:
            inp.append(0)

        pred_values = pred_reg[..., 0][msk_reg]

        inp.append(pred_values.mean())
        inp.append(pred_values.std())
        #
        inp.append(neighbors100[i].shape[0])
        median_area = med_area
        if neighbors100[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors100[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors200[i].shape[0])
        median_area = med_area
        if neighbors200[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors200[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors300[i].shape[0])
        median_area = med_area
        if neighbors300[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors300[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors400[i].shape[0])
        median_area = med_area
        if neighbors400[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors400[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        bst_j = 0
        pred_reg[~msk_reg] = 0
        pred_reg0 = pred_reg / 255.
        pred_reg0 = pred_reg0[..., 0] * (1 - pred_reg0[..., 1])
        max_regs = 1
        for j in range(1, sep_count + 1):
            sep_regs = []
            if bst_j > 0:
                separated_regions[j - 1].append(sep_regs)
                continue
            pred_reg2 = 255 * (pred_reg0 > sep_thresholds[j - 1])
            pred_reg2 = pred_reg2.astype(np.uint8)
            if j > sep_count - 1:
                kernel = np.ones((3, 3), np.uint8)
                pred_reg2 = cv2.erode(pred_reg2, kernel, iterations=2)
            lbls = measure.label(pred_reg2, neighbors=4, background=False)
            num_regs = lbls.max()
            if num_regs > 1:
                bst_j = j
                max_regs = num_regs
            if num_regs > 1 or (j < sep_count and num_regs > 0):
                lbls = lbls.astype(np.int32)
                labels_ws = watershed((255 - pred_reg[..., 0]), lbls, mask=msk_reg)
                #                if num_regs > 1:
                #                    labels_ws = watershed((255 - pred_reg[..., 0]), lbls, mask=msk_reg)
                #                else:
                #                    labels_ws = lbls
                start_num = len(main_regions[j - 1])
                labels_ws += start_num
                labels_ws[labels_ws == start_num] = 0
                for k in range(num_regs):
                    sep_regs.append(k + start_num)
                    main_regions[j - 1].append(i)
                lvl2_labels[j - 1][pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]] += labels_ws
            separated_regions[j - 1].append(sep_regs)

        inp.append(bst_j)
        inp.append(max_regs)
        inp.append(1)
        inp.append(0)

        inputs.append(np.asarray(inp))
    inputs = np.asarray(inputs)
    all_sep_props = []
    all_sep_inputs = []
    for j in range(sep_count):
        inputs_lvl2 = []
        pred_props2 = measure.regionprops(lvl2_labels[j])
        for i in range(len(pred_props2)):
            msk_reg = lvl2_labels[j][pred_props2[i].bbox[0]:pred_props2[i].bbox[2], pred_props2[i].bbox[1]:pred_props2[i].bbox[3]] == i + 1
            pred_reg = pred[pred_props2[i].bbox[0]:pred_props2[i].bbox[2], pred_props2[i].bbox[1]:pred_props2[i].bbox[3], 0]

            contours = cv2.findContours((msk_reg * 255).astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours[1]) > 0:
                cnt = contours[1][0]
                min_area_rect = cv2.minAreaRect(cnt)

            inp = []
            inp.append(nadir)
            inp.append(pred_props2[i].area)
            main_area = inputs[main_regions[j][i]][0]
            inp.append(pred_props2[i].area / main_area)
            if len(contours[1]) > 0:
                inp.append(cv2.isContourConvex(cnt) * 1.0)
                inp.append(min(min_area_rect[1]))
                inp.append(max(min_area_rect[1]))
                if max(min_area_rect[1]) > 0:
                    inp.append(min(min_area_rect[1]) / max(min_area_rect[1]))
                else:
                    inp.append(0)
                inp.append(min_area_rect[2])
            else:
                inp.append(0)
                inp.append(0)
                inp.append(0)
                inp.append(0)
                inp.append(0)
            inp.append(pred_props2[i].convex_area)
            inp.append(pred_props2[i].solidity)
            inp.append(pred_props2[i].eccentricity)
            inp.append(pred_props2[i].extent)
            inp.append(pred_props2[i].perimeter)
            inp.append(pred_props2[i].major_axis_length)
            inp.append(pred_props2[i].minor_axis_length)
            if (pred_props2[i].minor_axis_length > 0):
                inp.append(pred_props2[i].minor_axis_length / pred_props2[i].major_axis_length)
            else:
                inp.append(0)

            pred_values = pred_reg[msk_reg]

            inp.append(pred_values.mean())
            inp.append(pred_values.std())
            #
            inp.append(inputs[main_regions[j][i]][-16])
            median_area = inputs[main_regions[j][i]][-15]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-13])
            median_area = inputs[main_regions[j][i]][-12]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-10])
            median_area = inputs[main_regions[j][i]][-9]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-7])
            median_area = inputs[main_regions[j][i]][-6]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            bst_j = inputs[main_regions[j][i]][-4]
            max_regs = inputs[main_regions[j][i]][-3]

            inp.append(bst_j)
            inp.append(max_regs)
            inp.append(len(separated_regions[j][main_regions[j][i]]))
            inp.append(j + 1)

            inputs_lvl2.append(np.asarray(inp))
        all_sep_props.append(pred_props2)
        inputs_lvl2 = np.asarray(inputs_lvl2)
        all_sep_inputs.append(inputs_lvl2)

    if truth_folder is None:
        return inputs, pred_labels, all_sep_inputs, lvl2_labels, separated_regions
    else:
        outputs = []

        truth_labels = cv2.imread(path.join(truth_folder, "mask_" + "_".join(filename[:-4].split("_")[-2:]) + ".tif"), cv2.IMREAD_GRAYSCALE)
        truth_labels = measure.label(truth_labels, neighbors=8, background=0)
        truth_props = measure.regionprops(truth_labels)

        m = np.zeros((len(pred_props), len(truth_props)))

        for x in range(pred_labels.shape[1]):
            for y in range(pred_labels.shape[0]):
                if pred_labels[y, x] > 0 and truth_labels[y, x] > 0:
                    m[pred_labels[y, x] - 1, truth_labels[y, x] - 1] += 1

        truth_used = set([])
        for i in range(len(pred_props)):
            max_iou = 0
            for j in range(len(truth_props)):
                if m[i, j] > 0:
                    iou = m[i, j] / (pred_props[i].area + truth_props[j].area - m[i, j])
                    if iou > max_iou:
                        max_iou = iou
                    if iou > 0.5:
                        truth_used.add(j)
            if max_iou <= 0.5:
                max_iou = 0
            outputs.append(max_iou)

        outputs = np.asarray(outputs)
        fn = len(truth_props) - len(truth_used)

        all_sep_outputs = []

        for k in range(sep_count):
            pred_props2 = all_sep_props[k]
            outputs_lvl2 = []
            m2 = np.zeros((len(pred_props2), len(truth_props)))

            for x in range(lvl2_labels[k].shape[1]):
                for y in range(lvl2_labels[k].shape[0]):
                    if lvl2_labels[k][y, x] > 0 and truth_labels[y, x] > 0:
                        m2[lvl2_labels[k][y, x] - 1, truth_labels[y, x] - 1] += 1

            truth_used2 = set([])
            for i in range(len(pred_props2)):
                tp = 0
                max_iou = 0
                for j in range(len(truth_props)):
                    if m2[i, j] > 0:
                        iou = m2[i, j] / (pred_props2[i].area + truth_props[j].area - m2[i, j])
                        if iou > max_iou:
                            max_iou = iou
                        if iou > 0.5:
                            tp = 1
                            truth_used2.add(j)
                if max_iou <= 0.5:
                    max_iou = 0
                outputs_lvl2.append(max_iou)

            outputs_lvl2 = np.asarray(outputs_lvl2)
            all_sep_outputs.append(outputs_lvl2)

        return inputs, all_sep_inputs, separated_regions, outputs, all_sep_outputs, fn


def check_id(id: str):
    labels = cv2.imread(path.join(masks_folder, "mask_" + "_".join(id[:-4].split("_")[-2:]) + ".tif"),
                        cv2.IMREAD_UNCHANGED)
    if np.max(labels) > 255:
        print("FUCK", str(np.max(labels)))
    mask = cv2.imread(os.path.join(train_pred_folder, id), cv2.IMREAD_COLOR)[..., 0]
    return id if np.max(labels) > 1 and np.sum(mask) > 200 * 255 else None

def get_id(f):
    return "_".join(f.rstrip(".png").split("_")[-2:])

if __name__ == '__main__':
    t0 = timeit.default_timer()
    import os

    all_ids = np.array([f for f in os.listdir(train_pred_folder) if f.endswith("png")])


    with Pool(processes=DATA_THREADS) as pool:
        all_ids = pool.map(check_id, all_ids)

    all_ids = [id for id in all_ids if id]

    groups = [get_id(id) for id in all_ids]
    if not path.isdir(lgbm_models_folder):
        mkdir(lgbm_models_folder)

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

    for _i in tqdm(range(len(all_ids))):
        f = all_ids[_i]
        if path.isfile(path.join(train_pred_folder, f)) and '.png' in f:
            all_files.append(path.join(train_pred_folder, f))
            paramss.append((f, train_pred_folder, masks_folder))

    with Pool(processes=DATA_THREADS) as pool:
        results = pool.starmap(get_inputs, paramss)
    for i in range(len(results)):
        inp, inp2, sep_regs, otp, otp2, fn = results[i]
        inputs.append(inp)
        outputs.append(otp)
        inputs2.append(inp2)
        outputs2.append(otp2)
        separated_regions.append(sep_regs)
        fns.append(fn)

    gbm_models = []

    train_pred = [np.zeros_like(otp) for otp in outputs]
    train_pred2 = [[np.zeros_like(otp) for otp in otp2] for otp2 in outputs2]

    for it in range(num_split_iters):
        kf = GroupShuffleSplit(n_splits=folds_count)
        it2 = -1
        for train_idxs, test_idxs in kf.split(all_files, groups=groups):
            it2 += 1

            random.seed(it * 1000 + it2)
            np.random.seed(it * 1000 + it2)

            lr = random.random() * 0.1 + 0.02
            ff = random.random() * 0.5 + 0.5
            nl = random.randint(6, 50)
            print('training lgb', it, it2, 'lr:', lr, 'ff:', ff, 'nl:', nl)

            train_inp = []
            train_otp = []
            test_inp = []
            test_otp = []
            for i in train_idxs:
                if inputs[i].shape[0] > 0:
                    train_inp.append(inputs[i])
                    train_otp.append(outputs[i])
                for j in range(len(inputs2[i])):
                    if inputs2[i][j].shape[0] > 0:
                        train_inp.append(inputs2[i][j])
                        train_otp.append(outputs2[i][j])
            for i in test_idxs:
                if inputs[i].shape[0] > 0:
                    test_inp.append(inputs[i])
                    test_otp.append(outputs[i])
                for j in range(len(inputs2[i])):
                    if inputs2[i][j].shape[0] > 0:
                        test_inp.append(inputs2[i][j])
                        test_otp.append(outputs2[i][j])
            train_inp = np.concatenate(train_inp, axis=0)
            test_inp = np.concatenate(test_inp, axis=0)
            train_otp = np.concatenate(train_otp, axis=0)
            test_otp = np.concatenate(test_otp, axis=0)

            lgb_train = lgb.Dataset(train_inp, train_otp)
            lgb_eval = lgb.Dataset(test_inp, test_otp, reference=lgb_train)

            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'num_leaves': nl,
                'learning_rate': lr,
                'feature_fraction': ff,
                'bagging_fraction': 0.95,
                'bagging_freq': 1,
                'verbosity': 0,
                'seed': it * 1000 + it2,
                'num_threads': LGBM_THREADS
            }

            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=600,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=20)

            gbm.free_dataset()
            gbm_models.append(gbm)
            gbm.save_model(path.join(lgbm_models_folder, 'gbm_model_{0}_{1}.txt'.format(it, it2)))

            for i in test_idxs:
                if train_pred[i].shape[0] > 0:
                    train_pred[i] = train_pred[i] + gbm.predict(inputs[i])
                for j in range(len(train_pred2[i])):
                    if train_pred2[i][j].shape[0] > 0:
                        train_pred2[i][j] = train_pred2[i][j] + gbm.predict(inputs2[i][j])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
