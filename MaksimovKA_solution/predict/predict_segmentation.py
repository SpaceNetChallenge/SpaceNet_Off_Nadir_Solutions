import os
from params.params import args
from train.model_factory import make_model
from albumentations import PadIfNeeded, CenterCrop, Resize
from os import path
import numpy as np
import timeit
import cv2
from tqdm import tqdm
import gdal
import rasterio.features
import shapely.ops
import shapely.wkt
import shapely.geometry
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import watershed
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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


def my_watershed(what, mask1, mask2):
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(what, markers, mask=mask1, watershed_line=True)
    return labels


def wsh(mask_img, threshold, border_img, seeds, shift):
    img_copy = np.copy(mask_img)
    m = seeds * border_img

    img_copy[m <= threshold + shift] = 0
    img_copy[m > threshold + shift] = 1
    img_copy = img_copy.astype(np.bool)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    labeled_array = my_watershed(mask_img, mask_img, img_copy)
    return labeled_array


def preprocess_input(batch_x):
    means = [963.0, 805.0, 666.0]
    stds = [473.0, 403.0, 395.0]
    for i in range(batch_x.shape[3]):
        batch_x[:, :, :, i] = (batch_x[:, :, :, i] - means[i]) / stds[i]
    return batch_x


if __name__ == '__main__':

    submit_output_file = args.submit_output_file


    test_folder = args.test_folder

    prob_trs_nadir = 0.4
    shift_nadir = 0.4

    prob_trs_off_nadir = 0.4
    shift_off_nadir = 0.4

    prob_trs_high_nadir = 0.4
    shift_high_nadir = 0.4

    MIN_POLYGON_AREA = 150

    cut_size = args.default_size
    save_size = args.predict_size
    t0 = timeit.default_timer()
    crop_shape = args.crop_size

    models = []
    model_name = 'inceptionresnet_unet_borders'
    weight = '/wdata/models_weights/best_loss_double_head_inceptionresnet_unet_borders_fold{}.h5'
    
    for i in range(5):
        model = make_model(model_name, predict_flag=1)
        if not os.path.exists(weight.format(i)):
            if not os.path.exists('/wdata/models_weights/'):
                os.mkdir('/wdata/models_weights/')
            src = os.path.join('/project/default_inference_weights', (weight.split('/')[-1]).format(i))
            dst = weight.format(i)
            shutil.copy(src=src, dst=dst)
        print("Building model {} from weights {} ".format(model_name, weight.format(i)))
        model.load_weights(weight.format(i))
        models.append(model)

    model_name = 'inceptionresnet_fpn_borders'
    weight = '/wdata/models_weights/best_loss_double_head_inceptionresnet_fpn_borders_fold{}.h5'
    for i in range(5):

        model = make_model(model_name, predict_flag=1)
        if not os.path.exists(weight.format(i)):
            if not os.path.exists('/wdata/models_weights/'):
                os.mkdir('/wdata/models_weights/')
            src = os.path.join('/project/default_inference_weights', (weight.split('/')[-1]).format(i))
            dst = weight.format(i)
            shutil.copy(src=src, dst=dst)
        print("Building model {} from weights {} ".format(model_name, weight.format(i)))
        model.load_weights(weight.format(i))
        models.append(model)

    all_nadir_ids = sorted(os.listdir(test_folder))
    all_nadir_ids = sorted([el for el in all_nadir_ids if el.split('_')[0] == 'Atlanta' and os.path.isdir(os.path.join(test_folder, el))])
    
    n_augs = 1 + 1
    n_outputs = 2

    f = open(submit_output_file, 'w')
    f.write('ImageId,BuildingId,PolygonWKT_Pix,Confidence\n')

    for nadir_type in tqdm(all_nadir_ids[:]):
        sample_path = os.path.join(test_folder, nadir_type, 'MS')
        files = os.listdir(sample_path)
        files = [el for el in files if el.split('.')[-1] == 'tif']
        all_ids = sorted(['_'.join(el.split('.')[0].split('_')[-2:]) for el in files])
        for _id in tqdm(all_ids[:]):
            to_concat = []
            _type = 'Pan-Sharpen'

            file_path = os.path.join(test_folder, nadir_type, _type, _type + '_' + nadir_type + '_' + _id + '.tif')
            tileHdl = gdal.Open(file_path, gdal.GA_ReadOnly)
            data = tileHdl.ReadAsArray()

            data = np.swapaxes(data, 0, -1)
            resize = Resize(save_size, save_size)
            resized = resize(image=data)
            data = resized['image']

            data = np.swapaxes(data, 0, 1)

            image = data[:, :, :]
            image = image[:, :, :3]
            image = image[:, :, ::-1]

            pad = PadIfNeeded(cut_size, cut_size, p=1)
            padded = pad(image=image)
            image_padded = padded['image']

            final_mask = np.zeros((cut_size, cut_size, n_outputs))

            mask = np.zeros((image_padded.shape[0], image_padded.shape[1], n_outputs, len(models), n_augs),
                            dtype=np.float64)

            for model_id, model in enumerate(models):
                inp0 = [image_padded]
                inp0 = np.asarray(inp0)
                inp0 = inp0.astype(np.float32)
                inp0 = preprocess_input(np.array(inp0, "float32"))
                pred0 = model.predict(inp0)
                pred0 = pred0[:, :, :, :n_outputs]
                mask[:, :, :, model_id, 0] += pred0[0, :, :, :]

                inp0 = [np.fliplr(image_padded)]
                inp0 = np.asarray(inp0)
                inp0 = inp0.astype(np.float32)
                inp0 = preprocess_input(np.array(inp0, "float32"))
                pred0 = model.predict(inp0)
                pred0 = pred0[:, :, :, :n_outputs]
                mask[:, :, :, model_id, 1] += np.fliplr(pred0[0, :, :, :])

            mask = np.mean(mask, axis=(3, 4))

            mask = mask[:, :, :n_outputs]
            final_mask[:, :, :n_outputs] += mask
            final_mask = final_mask * 255
            
            tmp = np.zeros(final_mask.shape[:2], dtype=final_mask.dtype)
            tmp = np.expand_dims(tmp, -1)
            final_mask = np.concatenate([final_mask, tmp], axis=-1)

            final_mask = final_mask.astype('uint8')
            aug = CenterCrop(p=1, height=save_size, width=save_size)
            augmented = aug(image=final_mask)
            image_center_cropped = augmented['image']
            
            fid = nadir_type + '_' + _id
            

            written = 0
            angle = int(_id.split('_')[1][5:])
            prob_trs = 0
            shift = 0

            if angle <= 25:
                prob_trs = prob_trs_nadir
                shift = shift_nadir
            elif angle <= 40:
                prob_trs = prob_trs_off_nadir
                shift = shift_off_nadir
            else:
                prob_trs = prob_trs_high_nadir
                shift = shift_high_nadir

            pred_data = image_center_cropped[:, :, :]
            labels = wsh(pred_data[:, :, 0] / 255., prob_trs,
                         1 - pred_data[:, :, 1] / 255.,
                         pred_data[:, :, 0] / 255., shift)
            label_numbers = list(np.unique(labels))
            all_dfs = []
            for label in label_numbers:
                if label != 0:
                    submask = (labels == label).astype(np.uint8)
                    shapes = rasterio.features.shapes(submask.astype(np.int16), submask > 0)

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
                    all_dfs.append(df.copy())


            if len(all_dfs) > 0:
                df_poly = pd.concat(all_dfs)
                df_poly = df_poly.reset_index(drop=True)
                df_poly = df_poly[df_poly.area_size > MIN_POLYGON_AREA]
                df_poly = df_poly.reset_index(drop=True)
                df_poly = df_poly[df_poly.area_size > MIN_POLYGON_AREA].sort_values(
                    by='area_size', ascending=False)
                df_poly.loc[:, 'wkt'] = df_poly.poly.apply(lambda x: shapely.wkt.dumps(
                    x, rounding_precision=0))
                df_poly.loc[:, 'bid'] = list(range(1, len(df_poly) + 1))
                df_poly.loc[:, 'area_ratio'] = df_poly.area_size / df_poly.area_size.max()
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        fid,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    fid,
                    -1,
                    "POLYGON EMPTY"))

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    f.close()