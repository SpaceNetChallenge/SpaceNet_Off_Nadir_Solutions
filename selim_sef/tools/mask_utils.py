import os
import warnings

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from skimage import measure
from skimage.morphology import dilation, square, watershed

from multiprocessing.pool import Pool
from os import path

import click
import numpy as np
from scipy.ndimage import binary_erosion

np.random.seed(1)
import random

random.seed(1)
import cv2

warnings.simplefilter("ignore")

from scipy.ndimage import label

def create_separation(labels):
    tmp = dilation(labels > 0, square(12))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    props = measure.regionprops(labels)

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 5
            else:
                sz = 7
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 5
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 6
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1


def create_mask(img_id, data_dir):
    labels_dir = os.path.join(data_dir, "labels")
    masks_dir = os.path.join(data_dir, "masks_all")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    labels = cv2.imread(path.join(labels_dir, '{0}.tif'.format(img_id)), cv2.IMREAD_UNCHANGED)
    final_mask = np.zeros((labels.shape[0], labels.shape[1], 3))
    if np.sum(labels) == 0:
        cv2.imwrite(path.join(masks_dir, '{0}.png'.format(img_id)), final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return final_mask

    ships_num = np.max(labels)
    if ships_num > 0:
        for i in range(1, ships_num + 1):
            ship_mask = np.zeros_like(labels, dtype='bool')
            ship_mask[labels == i] = 1
            area = np.sum(ship_mask)
            if area < 200:
                contour_size = 1
            elif area < 500:
                contour_size = 2
            else:
                contour_size = 3
            eroded = binary_erosion(ship_mask, iterations=contour_size)
            countour_mask = ship_mask ^ eroded
            final_mask[..., 0] += ship_mask
            final_mask[..., 1] += countour_mask
    final_mask[..., 2] = create_separation(labels)
    msk = np.clip(final_mask * 255, 0, 255)
    cv2.imwrite(path.join(masks_dir, '{0}.png'.format(img_id)), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def save_mask_and_label(image_name):
    mask_path = os.path.join("train_labels", "masks", image_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    labeled_array, _ = label(mask)
    label_path = os.path.join("train_labels", "labels", image_name)
    os.makedirs(os.path.join("train_labels", "labels"), exist_ok=True)
    cv2.imwrite(label_path, labeled_array)
    create_mask(image_name[:-4], "train_labels")


def main():
    img_names = os.listdir(os.path.join("train_labels", "masks"))
    pool = Pool(8)
    pool.map(save_mask_and_label, img_names)


if __name__ == '__main__':
    main()
