import argparse
import os

import cv2
import rasterio
import shapely
import skimage

import numpy as np
from skimage import measure
from skimage.morphology import watershed
import pandas as pd
from tqdm import tqdm
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry

parser = argparse.ArgumentParser("Postprocessing")
arg = parser.add_argument
arg('--masks-path', type=str, default='predictions/lgbm_test_sub1', help='Path to predicted masks')
arg('--output-path', type=str, help='Path for output file')

args = parser.parse_args()


def label_mask(pred, main_threshold=0.3, seed_threshold=0.7, w_pixel_t=20, pixel_t=100):
    av_pred = pred / 255.
    av_pred = av_pred[..., 0] * (1 - av_pred[..., 2])
    av_pred = 1 * (av_pred > seed_threshold)
    av_pred = av_pred.astype(np.uint8)

    y_pred = measure.label(av_pred, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < w_pixel_t:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (255 - pred[..., 0])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=(pred[..., 0] > main_threshold * 255), watershed_line=True)

    props = measure.regionprops(y_pred)

    for i in range(len(props)):
        if props[i].area < pixel_t:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)
    return y_pred


MIN_AREA = 100

def _internal_test(mask_dir, out_file):

    # Postprocessing phase

    fn_out = out_file
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        test_image_list = os.listdir(os.path.join(mask_dir))
        for idx, image_id in tqdm(enumerate(test_image_list),
                                       total=len(test_image_list)):
            img1 = cv2.imread(os.path.join(mask_dir, image_id), cv2.IMREAD_UNCHANGED)
            labels = img1.astype(np.uint16)
            df_poly = mask_to_poly(labels, min_polygon_area_th=MIN_AREA)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id.lstrip("Pan-Sharpen_").rstrip(".tif"),
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))



def mask_to_poly(mask, min_polygon_area_th=MIN_AREA):
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df


def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
            line_prefix +
            '))",' +
            line_terminate
        )
    return line

if __name__ == '__main__':
    _internal_test(args.masks_path, args.output_path)