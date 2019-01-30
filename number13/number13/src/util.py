from shapely.geometry import LineString, Polygon, MultiPolygon
import numpy as np
import cv2
import shapely
from collections import defaultdict
from skimage.transform import rescale
import glob
from skimage import io
from tqdm import tqdm
import sys


def mask_for_polygons(polygons, im_size=(900, 900)):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    all_mask = np.zeros(im_size, np.uint8)
    for i, poly in enumerate(polygons):
        exteriors = []
        interiors = []
        if poly.has_z:
            exteriors.append(int_coords([(item[0:2]) for item in poly.exterior.coords]))
            for pi in poly.interiors:
                pi = [(item[0:2]) for item in pi.coords]
                interiors.append(int_coords(pi))
        else:
            exteriors.append(int_coords(poly.exterior.coords))
            for pi in poly.interiors:
                interiors.append(int_coords(pi.coords))
        cv2.fillPoly(all_mask, exteriors, i + 1)
        cv2.fillPoly(all_mask, interiors, 0)
    return all_mask


def contours_hierarchy(mask):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_TC89_KCOS)  # cv2.CHAIN_APPROX_SIMPLE,#orig cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    return contours, hierarchy


def mask_to_polygons(mask, epsilon=0.0, min_area=0):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = contours_hierarchy(mask)
    if epsilon == 0.0:
        approx_contours = contours
    else:
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return Polygon(), [], []
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = shapely.geometry.Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    #DO ALL polygon validation on returned poly
    return all_polygons


def mask_to_multipolygons(mask, epsilon=0.0, min_area=0, shift=(0, 0)):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = contours_hierarchy(mask)
    # create approximate contours to have reasonable submission size
    if epsilon == 0.0:
        approx_contours = contours
    else:
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if shift != (0, 0):
        sx, sy = shift
        approx_contours_shift = []
        for item in approx_contours:
            for i in item:
                i = i[0]
                approx_contours_shift.append([[i[0] + sx, i[1] + sy]])

        approx_contours = [np.asarray(approx_contours_shift).astype('int32')]

    if not contours:
        return Polygon(), [], []
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = shapely.geometry.Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    return all_polygons, approx_contours, hierarchy


def stretch_8bit(bands, lower_percent=2, higher_percent=98, chan=3):
    out = np.zeros_like(bands)
    for i in range(chan):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


def pansharpen(m, pan):
    #reference https://www.kaggle.com/resolut/panchromatic-sharpening
    # get m_bands
    rgbn = np.empty((m.shape[1], m.shape[2], 3))
    rgbn[:, :, 0] = m[7, :, :]  # ir2
    rgbn[:, :, 1] = m[6, :, :]  # ir1
    rgbn[:, :, 2] = m[4, :, :]  # red
    # scaled them
    rgbn_scaled = np.empty((m.shape[1] * 4, m.shape[2] * 4, 4))
    for i in range(3):
        img = rgbn[:, :, i]
        scaled = rescale(img, (4, 4))
        rgbn_scaled[:, :, i] = scaled
    # check size and crop for pan band
    if pan.shape[0] < rgbn_scaled.shape[0]:
        rgbn_scaled = rgbn_scaled[:pan.shape[0], :, :]
    else:
        pan = pan[:rgbn_scaled.shape[0], :]

    if pan.shape[1] < rgbn_scaled.shape[1]:
        rgbn_scaled = rgbn_scaled[:, :pan.shape[1], :]
    else:
        pan = pan[:, :rgbn_scaled.shape[1]]
    R = rgbn_scaled[:, :, 0]
    G = rgbn_scaled[:, :, 1]
    B = rgbn_scaled[:, :, 2]
    image = None
    all_in = R + G + B
    prod = np.multiply(all_in, pan)
    r = np.multiply(R, pan / all_in)[:, :, np.newaxis]
    g = np.multiply(G, pan / all_in)[:, :, np.newaxis]
    b = np.multiply(B, pan / all_in)[:, :, np.newaxis]
    image = np.concatenate([r, g, b], axis=2)
    return image

#bounding box strip for spacenet style output
def strip_tail(annotations):
    return [item[0:-1] for item in annotations]


def comput_mean_jpg(dir):
    g=glob.glob(''.join([dir,'*/','*.jpg']))
    ssum = np.array([0.0, 0.0, 0.0])
    shape = io.imread(g[0]).shape
    print ('imaga shapes::',shape)
    for item in tqdm(g):
        img = io.imread(item)
        ssum += np.asarray([np.sum(img[:, :, ch]) for ch in range(0, img.shape[-1])])
    avg = ssum / (len(g) * shape[0] * shape[1])
    return avg

if __name__=='__main__':
    data_means=comput_mean_jpg(sys.argv[1])
    print (data_means)
