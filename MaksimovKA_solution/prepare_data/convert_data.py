import sys
import os
from params.params import args
import shutil
import gdal
from osgeo import ogr
import numpy as np
from albumentations import Resize
from tqdm import tqdm
import cv2
gdal.UseExceptions()

if __name__ == '__main__':
    default_path = args.training_data
    save_path = args.output_data

    subpath = 'train'
    path = os.path.join(save_path, subpath)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path, 'images'))
    os.mkdir(os.path.join(path, 'masks'))

    path = os.path.join(default_path, 'geojson/spacenet-buildings/')
    files = os.listdir(path)
    files = ['_'.join(el.split('.')[0].split('_')[-2:]) for el in files]

    non_empty_files = []
    print('Getting non empty files')
    for _file in tqdm(files[:]):

        json_path = os.path.join(default_path, 'geojson/spacenet-buildings/')
        original_id = '_'.join(_file.split('_')[-2:])
        id_json = os.path.join(json_path, 'spacenet-buildings_' + original_id.split('.')[0] + '.geojson')
        try:
            Polys_ds = ogr.Open(id_json)
            Polys = Polys_ds.GetLayer()
            non_empty_files.append(_file)
        except:
            continue

    files = sorted(non_empty_files[:])
    files = files[:]
    path =default_path
    all_nadirs = os.listdir(path)
    all_nadirs = sorted([el for el in all_nadirs if el.split('_')[0] == 'Atlanta' and os.path.isdir(os.path.join(path, el))])

    print('Creating masks and contours')
    for _file in tqdm(files[:]):
        nadir_name = all_nadirs[0]

        img_id = '_'.join([nadir_name, _file])
        _type = 'Pan-Sharpen'
        file_path = os.path.join(default_path,  nadir_name, _type, _type + '_' + img_id + '.tif')
        # print(file_path)
        tileHdl = gdal.Open(file_path, gdal.GA_ReadOnly)
        tileGeoTransformationParams = tileHdl.GetGeoTransform()
        projection = tileHdl.GetProjection()
        json_path = os.path.join(default_path, 'geojson/spacenet-buildings/')
        original_id = '_'.join(_file.split('_')[-2:])
        id_json = os.path.join(json_path, 'spacenet-buildings_' + original_id.split('.')[0] + '.geojson')

        rasterDriver = gdal.GetDriverByName('MEM')

        final_mask = rasterDriver.Create('',
                                         900,
                                         900,
                                         1,  # missed parameter (band)
                                         gdal.GDT_Byte)

        final_mask.SetGeoTransform(tileGeoTransformationParams)
        final_mask.SetProjection(projection)
        tempTile = final_mask.GetRasterBand(1)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)


        Polys_ds = ogr.Open(id_json)
        Polys = Polys_ds.GetLayer()
        gdal.RasterizeLayer(final_mask, [1], Polys, burn_values=[255])
        mask = final_mask.ReadAsArray()
        # print(np.unique(mask))
        # print(mask.shape)
        final_mask = None

        out_path = os.path.join(save_path, 'train', 'masks', img_id + '.tif')
        rasterDriver = gdal.GetDriverByName('GTiff')
        final_mask = rasterDriver.Create(out_path,
                                         900,
                                         900,
                                         2,  # missed parameter (band)
                                         gdal.GDT_Byte)

        final_mask.SetGeoTransform(tileGeoTransformationParams)
        final_mask.SetProjection(projection)
        tempTile = final_mask.GetRasterBand(1)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)
        tempTile.WriteArray(mask[:, :])
        h, w = mask.shape
        all_contours = np.zeros((h, w), dtype=np.uint8)
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            img = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(img, [cnt], 0, 255, 1)
            contour = img.astype(np.uint8)
            all_contours += contour
        tempTile = final_mask.GetRasterBand(2)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)

        tempTile.WriteArray(all_contours[:, :])
        final_mask = None

    print('Copying masks')
    for _file in tqdm(files[:]):
        for nadir_name in all_nadirs[1:]:
            main_image_id = '_'.join([all_nadirs[0], _file])
            img_id = '_'.join([nadir_name, _file])
            src = os.path.join(save_path, 'train', 'masks', main_image_id + '.tif')
            dst = os.path.join(save_path, 'train', 'masks', img_id + '.tif')
            shutil.copy(src=src, dst=dst)

    print('Getting rasters')
    for _file in tqdm(files[:]):
        for nadir_name in all_nadirs[:]:

            img_id = '_'.join([nadir_name, _file])
            _type = 'Pan-Sharpen'
            to_concat = []
           
            file_path = os.path.join(default_path, nadir_name, _type, _type + '_' + img_id + '.tif')
            # print(file_path)
            tileHdl = gdal.Open(file_path, gdal.GA_ReadOnly)
            data = tileHdl.ReadAsArray()
            
            tileGeoTransformationParams = tileHdl.GetGeoTransform()
            projection = tileHdl.GetProjection()

            tileHdl = None
            data = np.swapaxes(data, 0, -1)
            resize = Resize(900, 900)
            resized = resize(image=data)
            data = resized['image']

            data = np.swapaxes(data, 0, 1)
            image = data[:, :, :]
            image = image[:, :, [2, 1, 0]] # take only rgb for baseline

            rasterDriver = gdal.GetDriverByName('GTiff')
            out_path = os.path.join(save_path, 'train', 'images', img_id + '.tif')
            # print(out_path)
            final_image = rasterDriver.Create(out_path,
                                             900,
                                             900,
                                             3,
                                             gdal.GDT_UInt16)

            final_image.SetGeoTransform(tileGeoTransformationParams)
            final_image.SetProjection(projection)

            for band in range(image.shape[2]):
                output_band = final_image.GetRasterBand(band + 1)
                output_band.SetNoDataValue(0)
                output_band.WriteArray(image[:, :, band])
                output_band.FlushCache()
                output_band.ComputeStatistics(False)

            final_image = None

