import sys
from os import path, mkdir, listdir, makedirs
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from tqdm import tqdm
from skimage import measure
from skimage.morphology import square, erosion, dilation
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage.color import label2rgb
from scipy import ndimage
import pandas as pd
from sklearn.model_selection import KFold
from shapely.wkt import dumps
from shapely.geometry import shape, Polygon
from collections import defaultdict

lgbm_pred = '/wdata/lgbm_pred'

def mask_to_polygons(mask, min_area=8.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return Polygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    if len(all_polygons) > 1:
        print('more than one polygon!')
    wkt = dumps(all_polygons[0], rounding_precision=0)

    return wkt

if __name__ == '__main__':
    t0 = timeit.default_timer()

    sub_name = 'submission.csv'
    if len(sys.argv) > 1:
        sub_name = sys.argv[2]

    df = []
    for fid in tqdm(listdir(lgbm_pred)):
        y_pred = cv2.imread(path.join(lgbm_pred, fid), cv2.IMREAD_UNCHANGED)

        if y_pred.max() > 0:
            for i in range(1, y_pred.max() + 1):
                mask = 255 * (y_pred == i)
                mask = mask.astype('uint8')
                wkt = mask_to_polygons(mask)
                df.append({'ImageId': fid.split('.tif')[0], 'BuildingId': i, 'PolygonWKT_Pix': wkt, 'Confidence': 1})
        else:
            df.append({'ImageId': fid.split('.tif')[0], 'BuildingId': -1, 'PolygonWKT_Pix': 'POLYGON EMPTY', 'Confidence': 1})

    df = pd.DataFrame(df, columns=['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
    df.to_csv('/wdata/' + sub_name, index=False)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))