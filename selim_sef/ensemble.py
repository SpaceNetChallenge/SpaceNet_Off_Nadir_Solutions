import argparse
from multiprocessing.pool import Pool

import numpy as np
from cv2 import cv2

cv2.setNumThreads(0)
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def average_strategy(images):
    return np.average(images, axis=0)


def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.


def ensemble_image(params):
    file, dirs, ensembling_dir, strategy = params
    images = []
    for dir in dirs:
        file_path = os.path.join(dir, file)
        images.append(cv2.imread(file_path, cv2.IMREAD_COLOR))
    images = np.array(images)

    if strategy == 'average':
        ensembled = average_strategy(images)
    elif strategy == 'hard_voting':
        ensembled = hard_voting(images)
    else:
        raise ValueError('Unknown ensembling strategy')
    cv2.imwrite(os.path.join(ensembling_dir, file), ensembled)


def ensemble(dirs, strategy, ensembling_dir, n_threads):
    files = os.listdir(dirs[0])
    params = []

    for file in files:
        params.append((file, dirs, ensembling_dir, strategy))
    pool = Pool(n_threads)
    pool.map(ensemble_image, params)
test_dirs = ['d161', 'd121', 'r34', 'sc50', 'r101', 'sc50_1', 'd161_1', 'd161_2', 'd161_3', 'd161_4','d161_5',
                                                  'd121_1','d121_2','d121_3','d121_4','d121_5', 'r34_1', 'r34_2', 'r34_3', 'sc50_1', 'sc50_2', 'sc50_3'
                                                  ]
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Ensemble masks")
    arg = parser.add_argument
    arg('--ensembling_cpu_threads', type=int, default=8)
    arg('--ensembling_dir', type=str, default='predictions/masks/ensemble')
    arg('--strategy', type=str, default='average')
    arg('--folds_dir', type=str, default='predictions/masks')
    arg('--dirs_to_ensemble', nargs='+', default=test_dirs)
    args = parser.parse_args()

    folds_dir = args.folds_dir
    dirs = [os.path.join(folds_dir, d) for d in args.dirs_to_ensemble]
    for d in dirs:
        if not os.path.exists(d):
            raise ValueError(d + " doesn't exist")
    os.makedirs(args.ensembling_dir, exist_ok=True)
    ensemble(dirs, args.strategy, args.ensembling_dir, args.ensembling_cpu_threads)
