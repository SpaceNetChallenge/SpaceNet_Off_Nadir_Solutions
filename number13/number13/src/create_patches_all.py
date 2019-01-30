import pandas as pd
import numpy as np
import json
from shapely.wkt import loads
import tifffile
from tqdm import tqdm
import os
import util
from util import pansharpen
import patchify
from cocoeval import maskUtils
from skimage import io
from skimage import measure
from config import TRAIN_DATA_DIR_MPAN, TRAIN_DATA_DIR_IRGB, NADIR, OFF_NADIR, VERY_NADIR , TRAIN_DATA_HOME
import argparse
import warnings
warnings.filterwarnings('ignore')

np.random.seed(10)
DEBUG = False

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_pan_sharpend(Dir, imageid):
    pan, ms = ''.join([Dir, 'PAN/', 'PAN_', imageid, '.tif']), ''.join([Dir, 'MS/', 'MS_', imageid, '.tif'])
    pan, ms = tifffile.imread(pan), tifffile.imread(ms)
    if np.argmin(ms.shape) == 2:
        ms = np.transpose(ms, (2, 0, 1))
    img = pansharpen(ms, pan)
    return img


def create_coco_anns(file_name, counter, mask_patch):
    anns = []
    images = []
    counter = counter
    for i in np.unique(mask_patch):
        if i != 0:
            m = (mask_patch == i).astype('uint8')
            image_id = file_name.split('/')[-1].replace('.jpg', '').replace('.tif', '')
            a = {'area': 0, 'category_id': 100, 'image_id': image_id, 'segmentation': [], 'id': counter, 'iscrowd': 0,
                 'bbox': []}
            d = {'file_name': file_name.split('/')[-1], 'height': stepsize, 'id': image_id, 'width': stepsize}
            _mask = maskUtils.encode(np.asfortranarray(m))
            a['bbox'] = maskUtils.toBbox(_mask).tolist()
            a['area'] = float(maskUtils.area(_mask))
            segmentation = binary_mask_to_polygon(m, tolerance=0)
            a['segmentation'] = segmentation
            anns.append(a)
            images.append(d)
            counter +=1
    return anns, images, counter


def patches_and_cocoann(gt, outdir_rgb, outdir_mpan, count=1,create_anns=True):
    gt.index = gt.ImageId
    gt_flt = gt[gt.PolygonWKT_Geo != 'POLYGON EMPTY']
    gv = gt_flt.ImageId.value_counts()
    annotations = []
    images = []
    counter = count
    for u in gt_flt.ImageId.unique():
        try:
            pan_sharpen_dir = ''.join([RAW_DIR, gt.name, '/Pan-Sharpen/'])
            image_file = ''.join([pan_sharpen_dir, 'Pan-Sharpen', '_', u, '.tif'])
            img_rgb = tifffile.imread(image_file)
            if np.argmin(img_rgb.shape) == 0:
                img_rgb = np.transpose(img_rgb, (1, 2, 0))
            img_rgb = img_rgb[:, :, [3, 2, 1, 0]]
            img_mpan = get_pan_sharpend(''.join([RAW_DIR, gt.name, '/']), u)

        except:
            print('load error..', u)
            continue
        if gv[u] > 1:
            poly = gt.loc[u].PolygonWKT_Pix.apply(lambda x: loads(x)).values.tolist()
        else:
            poly = [loads(gt.loc[u].PolygonWKT_Pix)]
        mask = util.mask_for_polygons(poly, im_size=imsize)
        img_patches_rgb, mask_patches, kp = patch_creator.create(img=img_rgb, mask=mask, nonzero=True)
        img_patches_mpan, _, _ = patch_creator.create(img=img_mpan, mask=mask, nonzero=True)

        for i, k in enumerate(kp.keys()):
            file_name_rgb = os.path.join(outdir_rgb, ''.join([u, '_', str(k), '.tif']))
            file_name_mpan = os.path.join(outdir_mpan, ''.join([u, '_', str(k), '.tif']))
            if create_anns:
                anns, images_d, counter = create_coco_anns(file_name_rgb, counter, mask_patches[i].squeeze())
                annotations.extend(anns)
                images.extend(images_d)
            tifffile.imsave(file_name_mpan, img_patches_mpan[i].astype('uint16'))
            tifffile.imsave(file_name_rgb, img_patches_rgb[i].astype('uint16'))
        if DEBUG:
            break
    return annotations, images, counter


def make_patch_data(gt, outdir_rgb, outdir_mpan, count=1, create_anns=False):
    outdir_rgb = os.path.join(STORE_DIR, outdir_rgb, gt.name)
    if not os.path.exists(outdir_rgb):
        os.mkdir(outdir_rgb)
    outdir_mpan = os.path.join(STORE_DIR, outdir_mpan,gt.name)
    if not os.path.exists(outdir_mpan):
        os.mkdir(outdir_mpan)
    annotations, images, counter = patches_and_cocoann(gt, outdir_rgb, outdir_mpan, count=count,create_anns = create_anns)
    return annotations, images, counter


def main_single(gt_file, outdir_rgb, outdir_mpan):
    final_annotation = {}
    final_annotation['info'] = []
    final_annotation['categories'] = [{'id': 100, 'name': 'building', 'supercategory': 'building'}]
    gt = pd.read_csv(gt_file)
    gt.name = gt_file.split('/')[-1].replace('_Train.csv', '')
    create_anns = gt.name=='Atlanta_nadir7_catid_1030010003D22F00'
    annotations, images, _ = make_patch_data(gt,  outdir_rgb, outdir_mpan, count=1, create_anns=create_anns)
    if create_anns:
        annpath_rgb = ''.join([outdir_rgb,'/', gt.name, '_annotation.json'])
        annpath_mpan = ''.join([outdir_mpan,'/', gt.name, '_annotation.json'])
        final_annotation['images'] = images
        final_annotation['annotations'] = annotations
        json.dump(final_annotation, open(annpath_rgb, 'w'))
        json.dump(final_annotation, open(annpath_mpan, 'w'))


def main(angles, outdir_rgb, outdir_mpan):
    if not os.path.isdir(outdir_rgb):
        os.mkdir(os.path.join(STORE_DIR,outdir_rgb))
    if not os.path.isdir(outdir_mpan):
        os.mkdir(os.path.join(STORE_DIR,outdir_mpan))
    for n in tqdm(angles):
        filepath = ''.join([SUMMARY_DIR, n])
        main_single(filepath, outdir_rgb , outdir_mpan)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='patch creator')
    parser.add_argument('--raw_dir', required=True,
                        help='Dir to read train raw images from')  # change required to true
    parser.add_argument('--summary_dir', required=True, help='ground truth firectory')
    parser.add_argument('--step_size' ,required=True, type=int, help= 'stepsize to create patch')
    parser.add_argument('--im_size' ,required=False, type=tuple, default=(900,900), help= 'image dimension')

    args = parser.parse_args()
    RAW_DIR = args.raw_dir
    SUMMARY_DIR = args.summary_dir
    STORE_DIR = TRAIN_DATA_HOME
    imsize = args.im_size
    stepsize = args.step_size

    patch_creator = patchify.PatchGenerator(stepsize=stepsize, imsize=imsize)
    
    angles = np.concatenate(([NADIR, OFF_NADIR, VERY_NADIR]))
    angles = [''.join([item, '_Train.csv']) for item in angles]
    main(angles, outdir_rgb=TRAIN_DATA_DIR_IRGB, outdir_mpan=TRAIN_DATA_DIR_MPAN)

