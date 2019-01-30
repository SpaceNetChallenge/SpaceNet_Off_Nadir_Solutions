from os import path, makedirs, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import time
#import matplotlib.pyplot as plt
#import seaborn as sns

train_dir = 'train_png'

df = []

for f in sorted(listdir(train_dir)):
    if '.png' in f and 'catid' in f:
        img_id = f.split('.png')[0]
        tmp = f.split('_')
        df.append({'id': img_id, 'nadir': int(tmp[1].split('nadir')[1]), 'catid': tmp[3], 'x': int(tmp[4]), 'y': int(tmp[5].split('.png')[0]) })
        
df = pd.DataFrame(df, columns=['id', 'nadir', 'catid', 'x', 'y'])

df['tile_id'] = df['x'].astype(str) + '_' + df['y'].astype(str)

tiles = df.groupby('tile_id')['x'].first().index.values
tiles_x = df.groupby('tile_id')['x'].first().values

df['fold'] = -1
kf = StratifiedKFold(n_splits=16, shuffle=True, random_state=111)
it = -1
for train_idx, val_idx in kf.split(tiles, tiles_x):
    it += 1
    df.loc[df['tile_id'].isin(tiles[val_idx]), 'fold'] = it

df.to_csv('train_folds.csv', index=False)