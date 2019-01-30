import argparse
import os

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import KFold


def get_id(f):
    return "_".join(f.rstrip(".tif").split("_")[-2:])

def get_nadir(f):
    return  int(f.split("_")[2][5:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PyTorch Segmentation Pipeline")
    arg = parser.add_argument
    arg('--data-path', type=str, default='/media/selim/sota/datasets/spacenet/SpaceNet-Off-Nadir_Train', help='Path to dataset folder')
    arg('--folds', type=int, default=16, help='Num folds')
    arg('--seed', type=int, default=777, help='Seed')
    args = parser.parse_args()
    train_dir = args.data_path
    nadir_ranges = [range(0, 11), range(11, 21), range(21, 31), range(31, 41), range(41, 55)]
    dirs = sorted([d for d in os.listdir(train_dir) if d.startswith("Atlanta")])
    folds = []
    ids = None
    data = {}
    np.random.seed(args.seed)

    for d in dirs:
        if os.path.isdir(os.path.join(train_dir, d)):
            dir = os.path.join(train_dir, d, "Pan-Sharpen")
            files = sorted([f for f in os.listdir(dir) if f.endswith("tif")])

            if ids is None:
                ids = list(set([get_id(f) for f in files]))
                kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RandomState(args.seed))
                for fold, splits in enumerate(kfold.split(files)):
                    for idx in splits[1]:
                        data[ids[idx]] = fold

            for f in files:
                id = get_id(f)
                fold = data[id]
                folds.append([f, fold])

    frame = pd.DataFrame(folds, columns=["id", "fold"])
    print(frame.groupby('fold').count())
    frame.to_csv("folds_angles.csv", index=False)

