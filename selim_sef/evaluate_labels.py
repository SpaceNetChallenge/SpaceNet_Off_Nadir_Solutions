import os
from multiprocessing.pool import Pool

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from training.metric import calc_score
import warnings
warnings.filterwarnings("ignore")
preds_dir = "predictions/lgbm_oof_sub1"
labels_dir = "/media/selim/sota/datasets/spacenet/SpaceNet-Off-Nadir_Train/labels"


def calc(f):
    label = cv2.imread(os.path.join(preds_dir, f), cv2.IMREAD_UNCHANGED)
    label_file = "mask_" + "_".join(f[:-4].split("_")[-2:]) + ".tif"
    gt_label = cv2.imread(os.path.join(labels_dir, label_file), cv2.IMREAD_UNCHANGED)
    res = calc_score(gt_label, label)
    nadir = int(f.split("_")[2].lstrip("nadir"))

    return nadir, res


if __name__ == '__main__':
    pool = Pool(32)
    f_scores = []
    for nadir_range in (range(0, 26),range(26, 40),range(40, 55),):
        results = pool.map(calc, [ f for f in os.listdir(os.path.join(preds_dir)) if int(f.split("_")[2].lstrip("nadir")) in nadir_range])
        tp, fp, fn = 0, 0, 0

        for res in results:
            nadir, preds = res
            if preds:
                _tp, _fp, _fn = preds
                if nadir in nadir_range:
                    tp += _tp
                    fp += _fp
                    fn += _fn
        if tp > 0 and fp > 0 and fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * precision * recall / (precision + recall),

            print("{} F1 {}".format(nadir_range, f_score))
            f_scores.append(f_score)

    print("Average: {}".format(np.mean(f_scores)))
