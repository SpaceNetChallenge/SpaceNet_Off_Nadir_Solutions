"""
XD_XD's solution for the SpaceNet Off-Nadir Building Detection Challenge

** Usage

 $ python main.py <sub-command> [options]

    sub-commands:
        - train
        - inference
        - filecheck
        - check

"""
import warnings
from pathlib import Path
import tempfile
import csv
import os
import datetime
import json
import shutil
import time
import sys

import scipy.sparse as ss
import numpy as np
import pandas as pd
import attr
import click
import tqdm
import cv2
import rasterio
import skimage.measure
from sklearn.utils import Bunch

from torch import nn
import torch
from torch.optim import Adam
from torchvision.models import vgg16
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from albumentations.torch.functional import img_to_tensor
from albumentations import (
    Normalize, Compose, HorizontalFlip, RandomRotate90, RandomCrop, CenterCrop)

import spacenetutilities.labeltools.coreLabelTools as cLT
from spacenetutilities import geoTools as gT
from shapely.geometry import shape
from shapely.wkt import dumps
import geopandas as gpd


warnings.simplefilter(action='ignore', category=FutureWarning)


class conv_relu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class decoder_block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(decoder_block, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_relu(in_channels, middle_channels),
            conv_relu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class unet_vgg16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super().__init__()
        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        self.center = decoder_block(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = decoder_block(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = decoder_block(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = decoder_block(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = decoder_block(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = conv_relu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        return x_out


def get_image(imageid, basepath='/wdata/dataset', rgbdir='train_rgb'):
    fn = f'{basepath}/{rgbdir}/Pan-Sharpen_{imageid}.tif'
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AtlantaDataset(Dataset):
    def __init__(self, image_ids, aug=None, basepath='/wdata/dataset'):
        self.image_ids = image_ids
        self.aug = aug
        self.basepath = basepath

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im = get_image(imageid, basepath=self.basepath, rgbdir='train_rgb')
        assert im is not None

        locid = '_'.join(imageid.split('_')[-2:])

        mask = cv2.imread(
            f'{self.basepath}/masks/mask_{locid}.tif',
            cv2.IMREAD_GRAYSCALE)
        assert mask is not None

        augmented = self.aug(image=im, mask=mask)

        mask_ = (augmented['mask'] > 0).astype(np.uint8)
        mask_ = torch.from_numpy(np.expand_dims(mask_, 0)).float()
        label_ = torch.from_numpy(np.expand_dims(augmented['mask'], 0)).float()

        return (
            img_to_tensor(augmented['image']), mask_, label_, imageid)


class AtlantaTestDataset(Dataset):
    def __init__(self, image_ids, aug=None, basepath='/wdata/dataset'):
        self.image_ids = image_ids
        self.aug = aug
        self.basepath = basepath

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im = get_image(imageid, basepath=self.basepath, rgbdir='test_rgb')
        assert im is not None

        augmented = self.aug(image=im)
        return img_to_tensor(augmented['image']), imageid


@click.group()
def cli():
    pass


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
def check(inputs):
    systemcheck_train()
    # TODO: check training images


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='/wdata',
              help="working directory")
def preproctrain(inputs, working_dir):
    """
    * Making 8bit rgb train images
    """
    # preproc images
    Path(f'{working_dir}/dataset/train_rgb').mkdir(parents=True,
                                                   exist_ok=True)
    catalog_paths = list(sorted(Path(inputs).glob('./Atlanta_nadir*')))
    assert len(catalog_paths) > 0
    print('Found {} catalog directories'.format(len(catalog_paths)))
    for catalog_dir in tqdm.tqdm(catalog_paths, total=len(catalog_paths)):
        src_imgs = list(sorted(catalog_dir.glob('./Pan-Sharpen/Pan-*.tif')))
        for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
            dst = f'{working_dir}/dataset/train_rgb/{src.name}'
            if not Path(dst).exists():
                pan_to_bgr(str(src), dst)

    # prerpoc masks
    (Path(working_dir) / Path('dataset/masks')).mkdir(parents=True,
                                                      exist_ok=True)
    geojson_dir = Path(inputs) / Path('geojson/spacenet-buildings')
    mask_dir = Path(working_dir) / Path('dataset/masks')
    ref_catalog_name = list(Path(inputs).glob(
        './Atlanta_nadir*/Pan-Sharpen'))[0].parent.name
    for geojson_fn in geojson_dir.glob('./spacenet-buildings_*.geojson'):
        masks_from_geojson(mask_dir, inputs, ref_catalog_name, geojson_fn)


def masks_from_geojson(mask_dir, inputs, ref_name, geojson_fn):
    chip_id = geojson_fn.name.lstrip('spacenet-buildings_').rstrip('.geojson')
    mask_fn = mask_dir / f'mask_{chip_id}.tif'
    if mask_fn.exists():
        return

    ref_fn = str(Path(inputs) / Path(
        f'{ref_name}/Pan-Sharpen/Pan-Sharpen_{ref_name}_{chip_id}.tif'))
    cLT.createRasterFromGeoJson(str(geojson_fn), ref_fn, str(mask_fn))


def read_cv_splits(inputs):
    fn = '/root/working/cv.txt'
    if not Path(fn).exists():
        train_imageids = list(sorted(
            Path(inputs).glob('./*/Pan-Sharpen/Pan-Sharpen_*.tif')))

        # split 4 folds
        df_fold = pd.DataFrame({
            'filename': train_imageids,
            'catid': [path.parent.parent.name for path in train_imageids],
        })
        df_fold.loc[:, 'fold_id'] = np.random.randint(0, 4, len(df_fold))
        df_fold.loc[:, 'ImageId'] = df_fold.filename.apply(
            lambda x: x.name[len('Pan-Sharpen_'):-4])

        df_fold[[
            'ImageId', 'filename', 'catid', 'fold_id',
        ]].to_csv(fn, index=False)

    return pd.read_csv(fn)


@cli.command()  # noqa: C901
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='./working',
              help="working directory")
@click.option('--fold_id', '-f', default=0, help='fold id')
def train(inputs, working_dir, fold_id):
    start_epoch, step = 0, 0

    # TopCoder
    num_workers, batch_size = 8, 4 * 8
    gpus = [0, 1, 2, 3]

    # My machine
    # num_workers, batch_size = 8, 2 * 3
    # gpus = [0, 1]

    patience, n_epochs = 8, 150
    lr, min_lr, lr_update_rate = 1e-4, 5e-5, 0.5
    training_timelimit = 60 * 60 * 24 * 2  # 2 days
    st_time = time.time()

    model = unet_vgg16(pretrained=True)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    train_transformer = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomCrop(512, 512, p=1.0),
        Normalize(),
    ], p=1.0)

    val_transformer = Compose([
        CenterCrop(512, 512, p=1.0),
        Normalize(),
    ], p=1.0)

    # train/val loadrs
    df_cvfolds = read_cv_splits(inputs)
    trn_loader, val_loader = make_train_val_loader(
        train_transformer, val_transformer, df_cvfolds, fold_id,
        batch_size, num_workers)

    # train
    criterion = binary_loss(jaccard_weight=0.25)
    optimizer = Adam(model.parameters(), lr=lr)

    report_epoch = 10

    model_name = f'v12_f{fold_id}'
    fh = open_log(model_name)

    # vers for early stopping
    best_score = 0
    not_improved_count = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tl = trn_loader  # alias
        trn_metrics = Metrics()

        try:
            tq = tqdm.tqdm(total=(len(tl) * trn_loader.batch_size))
            tq.set_description(f'Ep{epoch:>3d}')
            for i, (inputs, targets, labels, names) in enumerate(trn_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()

                # Increment step counter
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)

                # Update eval metrics
                trn_metrics.loss.append(loss.item())
                trn_metrics.bce.append(criterion._stash_bce_loss.item())
                trn_metrics.jaccard.append(criterion._stash_jaccard.item())

                if i > 0 and i % report_epoch == 0:
                    report_metrics = Bunch(
                        epoch=epoch,
                        step=step,
                        trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                        trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                        trn_jaccard=np.mean(
                            trn_metrics.jaccard[-report_epoch:]),
                    )
                    write_event(fh, **report_metrics)
                    tq.set_postfix(
                        loss=f'{report_metrics.trn_loss:.5f}',
                        bce=f'{report_metrics.trn_bce:.5f}',
                        jaccard=f'{report_metrics.trn_jaccard:.5f}')

            # End of epoch
            report_metrics = Bunch(
                epoch=epoch,
                step=step,
                trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                trn_jaccard=np.mean(trn_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_metrics)
            tq.set_postfix(
                loss=f'{report_metrics.trn_loss:.5f}',
                bce=f'{report_metrics.trn_bce:.5f}',
                jaccard=f'{report_metrics.trn_jaccard:.5f}')
            tq.close()
            save(model, epoch, step, model_name)

            # Run validation
            val_metrics = validation(model,
                                     criterion,
                                     val_loader,
                                     epoch,
                                     step,
                                     fh)
            report_val_metrics = Bunch(
                epoch=epoch,
                step=step,
                val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_val_metrics)

            if time.time() - st_time > training_timelimit:
                tq.close()
                break

            if best_score < report_val_metrics.val_jaccard:
                best_score = report_val_metrics.val_jaccard
                not_improved_count = 0
                copy_best(model, epoch, model_name, step)
            else:
                not_improved_count += 1

            if not_improved_count >= patience:
                # Update learning rate and optimizer

                lr *= lr_update_rate
                # Stop criterion
                if lr < min_lr:
                    tq.close()
                    break

                not_improved_count = 0

                # Load best weight
                del model
                model = unet_vgg16(pretrained=False)
                path = f'/root/working/models/{model_name}/{model_name}_best'
                cp = torch.load(path)
                model = nn.DataParallel(model).cuda()
                epoch = cp['epoch']
                model.load_state_dict(cp['model'])
                model = model.module
                model = nn.DataParallel(model, device_ids=gpus).cuda()

                # Init optimizer
                optimizer = Adam(model.parameters(), lr=lr)

        except KeyboardInterrupt:
            save(model, epoch, step, model_name)
            tq.close()
            fh.close()
            sys.exit(1)
        except Exception as e:
            raise e
            break

    fh.close()


def validation(model, criterion, val_loader,
               epoch, step, fh):
    report_epoch = 10
    val_metrics = Metrics()

    with torch.no_grad():
        model.eval()

        vl = val_loader

        tq = tqdm.tqdm(total=(len(vl) * val_loader.batch_size))
        tq.set_description(f'(val) Ep{epoch:>3d}')
        for i, (inputs, targets, labels, names) in enumerate(val_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            tq.update(inputs.size(0))

            val_metrics.loss.append(loss.item())
            val_metrics.bce.append(criterion._stash_bce_loss.item())
            val_metrics.jaccard.append(criterion._stash_jaccard.item())

            if i > 0 and i % report_epoch == 0:
                report_metrics = Bunch(
                    epoch=epoch,
                    step=step,
                    val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                    val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                    val_jaccard=np.mean(
                        val_metrics.jaccard[-report_epoch:]),
                )
                tq.set_postfix(
                    loss=f'{report_metrics.val_loss:.5f}',
                    bce=f'{report_metrics.val_bce:.5f}',
                    jaccard=f'{report_metrics.val_jaccard:.5f}')

        # End of epoch
        report_metrics = Bunch(
            epoch=epoch,
            step=step,
            val_loss=np.mean(val_metrics.loss[-report_epoch:]),
            val_bce=np.mean(val_metrics.bce[-report_epoch:]),
            val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
        )
        tq.set_postfix(
            loss=f'{report_metrics.val_loss:.5f}',
            bce=f'{report_metrics.val_bce:.5f}',
            jaccard=f'{report_metrics.val_jaccard:.5f}')
        tq.close()

    return val_metrics


@attr.s
class Metrics(object):
    loss = attr.ib(default=[])
    bce = attr.ib(default=[])
    jaccard = attr.ib(default=[])


class binary_loss(object):
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self._stash_bce_loss = 0
        self._stash_jaccard = 0

    def __call__(self, outputs, targets):
        eps = 1e-15

        self._stash_bce_loss = self.nll_loss(outputs, targets)
        loss = (1 - self.jaccard_weight) * self._stash_bce_loss

        jaccard_target = (targets == 1).float()
        jaccard_output = torch.sigmoid(outputs)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()

        jaccard_score = (
            (intersection + eps) / (union - intersection + eps))
        self._stash_jaccard = jaccard_score
        loss += self.jaccard_weight * (1. - jaccard_score)

        return loss


def save(model, epoch, step, model_name):
    path = f'/wdata/models/{model_name}/{model_name}_ep{epoch}_{step}'
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, path)


def copy_best(model, epoch, model_name, step):
    path = f'/wdata/models/{model_name}/{model_name}_ep{epoch}_{step}'
    best_path = f'/root/working/models/{model_name}/{model_name}_best'
    shutil.copy(path, best_path)


def write_event(log, **data):
    data['dt'] = datetime.datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def open_log(model_name):
    time_str = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
    path = f'/wdata/models/{model_name}/{model_name}.{time_str}.log'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, 'at', encoding='utf8')
    return fh


def make_train_val_loader(train_transformer,
                          val_transformer,
                          df_cvfolds,
                          fold_id,
                          batch_size,
                          num_workers):
    trn_dataset = AtlantaDataset(
        df_cvfolds[df_cvfolds.fold_id != fold_id].ImageId.tolist(),
        aug=train_transformer)
    trn_loader = DataLoader(
        trn_dataset,
        sampler=RandomSampler(trn_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    val_dataset = AtlantaDataset(
        df_cvfolds[df_cvfolds.fold_id == fold_id].ImageId.tolist(),
        aug=val_transformer)
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())
    return trn_loader, val_loader


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='/wdata',
              help="working directory")
@click.option('--output', '-o', default='out.txt',
              help="output filename")
def inference(inputs, working_dir, output):
    print('Collect filenames...')
    test_collection = []
    catalog_paths = list(sorted(Path(inputs).glob('./Atlanta_nadir*')))
    assert len(catalog_paths) > 0

    print(f'Found {len(catalog_paths)} catalog directories')
    for catalog_dir in catalog_paths:
        src_imgs = list(sorted(catalog_dir.glob('./Pan-Sharpen/Pan-*.tif')))
        for src in src_imgs:
            test_collection.append(src.name)
    print(f'Found {len(test_collection)} test images.')
    assert len(test_collection) > 0

    print(f'Check preprocessed 8bit rgb images')
    for src_img in test_collection:
        assert (
            Path(working_dir) / Path(f'dataset/test_rgb/{src_img}')).exists()

    model_names = [
        'v12_f0_best',
        'v12_f1_best',
        'v12_f2_best',
    ]
    for model_name in model_names:
        inference_by_model(model_name, test_collection)

    # merge prediction masks and write submission file
    output_fn = str(Path(working_dir) / output)
    make_sub(model_names, test_collection, output_fn)


def make_sub(model_names, test_collection, output_fn):  # noqa: C901
    chip_summary_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        tq = tqdm.tqdm(total=(len(test_collection)))
        tq.set_description(f'(avgfolds)')
        for name in test_collection:
            tq.update(1)
            y_pred_avg = np.zeros((900, 900), dtype=np.float32)

            imageid = name.lstrip('Pan-Sharpen_').rstrip('.tif')
            for model_name in model_names:
                # Prediction mask
                prefix = '_'.join(model_name.split('_')[:2])
                pred_mask_dir = f'/wdata/models/{prefix}/test_{model_name}/'
                y_pred = np.array(ss.load_npz(
                    str(Path(pred_mask_dir) / Path(f'{imageid}.npz'))
                ).todense() / 255.0)
                y_pred_avg += y_pred
            y_pred_avg /= len(model_names)

            # Remove small objects
            y_pred = (y_pred_avg > 0.5)
            y_pred_label = skimage.measure.label(y_pred)

            nadir_angle = int(imageid.split('_')[1].lstrip('nadir'))

            min_area_thresh = 200
            if nadir_angle <= 25:
                min_area_thresh = 150
            if nadir_angle <= 40 and nadir_angle > 25:
                min_area_thresh = 200
            if nadir_angle > 40:
                min_area_thresh = 250

            for lbl_idx in np.unique(y_pred_label):
                if (y_pred_label == lbl_idx).sum() < min_area_thresh:
                    y_pred_label[y_pred_label == lbl_idx] = 0

            # to_summary
            simplification_threshold = 0
            preds_test = (y_pred_label > 0).astype('uint8')
            pred_geojson_path = str(Path(tempdir) / Path(f'{name}.json'))

            catid = '_'.join(imageid.split('_')[:-2])
            geotiff_path = f'/data/test/{catid}/Pan-Sharpen/'
            im_fname = f'Pan-Sharpen_{imageid}.tif'

            try:
                raw_test_im = rasterio.open(
                    os.path.join(geotiff_path, im_fname))
                shapes = cLT.polygonize(
                    preds_test,
                    raw_test_im.profile['transform'])
                geom_list = []
                raster_val_list = []
                for s in shapes:
                    geom_list.append(shape(s[0]).simplify(
                        tolerance=simplification_threshold,
                        preserve_topology=False))
                    raster_val_list.append(s[1])
                feature_gdf = gpd.GeoDataFrame({
                    'geometry': geom_list,
                    'rasterVal': raster_val_list})
                feature_gdf.crs = raw_test_im.profile['crs']
                feature_gdf['conf'] = 1
                gT.exporttogeojson(pred_geojson_path, feature_gdf)
            except ValueError:
                # print(f'Warning: Empty prediction array for {name}')
                pass

            chip_summary = {
                'chipName': im_fname,
                'geoVectorName': pred_geojson_path,
                'imageId': imageid,
                'geotiffPath': geotiff_path,
            }
            chip_summary_list.append(chip_summary)

        tq.close()
        __createCSVSummaryFile(chip_summary_list, output_fn, pixPrecision=2)


def __createCSVSummaryFile(chipSummaryList, outputFileName, pixPrecision=2):
    with open(outputFileName, 'w') as csvfile:
        writerTotal = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writerTotal.writerow([
            'ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])

        # TODO: Add description=createCSVSummaryFile
        for chipSummary in tqdm.tqdm(chipSummaryList,
                                     total=len(chipSummaryList),
                                     desc='createCSVSummaryFile'):
            chipName = chipSummary['chipName']
            geoVectorName = chipSummary['geoVectorName']
            rasterChipDirectory = chipSummary['geotiffPath']
            imageId = chipSummary['imageId']

            buildingList = gT.geoJsonToPixDF(
                geoVectorName,
                rasterName=os.path.join(rasterChipDirectory, chipName),
                affineObject=[],
                gdal_geomTransform=[],
                pixPrecision=pixPrecision)
            buildingList = gT.explodeGeoPandasFrame(buildingList)

            if len(buildingList) > 0:
                for idx, building in buildingList.iterrows():
                    tmpGeom = dumps(building.geometry,
                                    rounding_precision=pixPrecision)
                    writerTotal.writerow([imageId, idx, tmpGeom, 1])
            else:
                imageId = chipSummary['imageId']
                writerTotal.writerow([imageId, -1,
                                      'POLYGON EMPTY', 1])


def inference_by_model(model_name, filenames,
                       batch_size=2,
                       num_workers=0,
                       fullsize_mode=False):
    # TODO: Optimize parameters for p2.xlarge
    print(f'Inrefernce by {model_name}')
    prefix = '_'.join(model_name.split('_')[:2])
    model_checkpoint_file = f'/root/working/models/{prefix}/{model_name}'

    pred_mask_dir = f'/wdata/models/{prefix}/test_{model_name}/'
    Path(pred_mask_dir).mkdir(parents=True, exist_ok=True)

    model = unet_vgg16(pretrained=False)
    cp = torch.load(model_checkpoint_file)
    if 'module.final.weight' in cp['model']:
        model = nn.DataParallel(model).cuda()
        epoch = cp['epoch']
        model.load_state_dict(cp['model'])
        model = model.module
        model = model.cuda()
    else:
        epoch = cp['epoch']
        model.load_state_dict(cp['model'])
        model = model.cuda()

    image_ids = [
        Path(path).name.lstrip('Pan-Sharpen_').rstrip('.tif')
        for path in Path('/wdata/dataset/test_rgb/').glob(
            'Pan-Sharpen*.tif')]

    tst_transformer = Compose([
        Normalize(),
    ], p=1.0)
    tst_dataset = AtlantaTestDataset(image_ids, aug=tst_transformer)
    tst_loader = DataLoader(
        tst_dataset,
        sampler=SequentialSampler(tst_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        tq = tqdm.tqdm(total=(len(tst_loader) * tst_loader.batch_size))
        tq.set_description(f'(test) Ep{epoch:>3d}')
        for X, names in tst_loader:
            tq.update(X.size(0))

            # TODO
            if fullsize_mode:
                pass
            else:
                pass

            for j, name in enumerate(names):
                # Image level inference
                # 900 -> 512 crop
                X_ = torch.stack([
                    X[j, :, :512, :512],
                    X[j, :, -512:, :512],
                    X[j, :, :512, -512:],
                    X[j, :, -512:, -512:],
                ])

                y_pred = np.zeros((900, 900), dtype=np.float32)
                y_pred_weight = np.zeros((900, 900), dtype=np.uint8)
                inputs = X_.cuda()
                outputs = model(inputs)
                y_pred_sigmoid = np.clip(torch.sigmoid(
                    torch.squeeze(outputs)
                ).detach().cpu().numpy(), 0.0, 1.0)

                y_pred[:512, :512] += y_pred_sigmoid[0]
                y_pred_weight[:512, :512] += 1
                y_pred[-512:, :512] += y_pred_sigmoid[1]
                y_pred_weight[-512:, :512] += 1
                y_pred[:512, -512:] += y_pred_sigmoid[2]
                y_pred_weight[:512, -512:] += 1
                y_pred[-512:, -512:] += y_pred_sigmoid[3]
                y_pred_weight[-512:, -512:] += 1
                y_pred = y_pred / y_pred_weight

                # Save quanlized values
                y_pred_mat = ss.csr_matrix(
                    np.round(y_pred * 255).astype(np.uint8))
                ss.save_npz(
                    str(Path(pred_mask_dir) / Path(f'{name}.npz')),
                    y_pred_mat)
        tq.close()


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='/wdata',
              help="working directory")
def preproctest(inputs, working_dir):
    """
    * Making 8bit rgb test images
    """
    (Path(working_dir) / Path('dataset/test_rgb')).mkdir(parents=True,
                                                         exist_ok=True)

    # rgb images
    catalog_paths = list(sorted(Path(inputs).glob('./Atlanta_nadir*')))
    assert len(catalog_paths) > 0
    print('Found {} catalog directories'.format(len(catalog_paths)))
    for catalog_dir in tqdm.tqdm(catalog_paths, total=len(catalog_paths)):
        src_imgs = list(sorted(catalog_dir.glob('./Pan-Sharpen/Pan-*.tif')))
        for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
            dst = f'{working_dir}/dataset/test_rgb/{src.name}'
            if not Path(dst).exists():
                pan_to_bgr(str(src), dst)

    # TODO: Add assertion check with md5sum
    # 3126a99e11d8b014630638f63f892c2c
    # working/dataset/test/Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_733151_3735939.tif


def pan_to_bgr(src, dst, thresh=3000):
    with rasterio.open(src, 'r') as reader:
        img = np.empty((reader.height,
                        reader.width,
                        reader.count))
        for band in range(reader.count):
            img[:, :, band] = reader.read(band+1)
    img = np.clip(img[:, :, :3], None, thresh)
    img = np.floor_divide(img, thresh/255).astype('uint8')
    cv2.imwrite(dst, img)


@cli.command()
@click.option('--inputs', '-i', default='./test',
              help="input directory")
@click.option('--working_dir', '-w', default='/wdata',
              help="working directory")
def filecheck(inputs, working_dir):
    # check test images generated by sp4 baseline code
    filecheck_inference_models(working_dir)
    systemcheck_inference()
    # filecheck_inference_images(working_dir)

    # check train images generated by sp4 baseline code
    # check train masks generated by sp4 baseline code
    # print("Something is wrong. Contact with the author.")


def filecheck_inference_models(working_dir):
    checklist = [
        '/root/working/models/v12_f0/v12_f0_best',
        '/root/working/models/v12_f1/v12_f1_best',
        '/root/working/models/v12_f2/v12_f2_best',
    ]

    is_ok = True
    for path in checklist:
        is_ok &= __filecheck(Path(path))

    is_warn = True
    cp = torch.load('/root/working/models/v12_f0/v12_f0_best')
    is_warn &= helper_assertion_check("Check v12_f0_best.step == 80206",
                                      cp['step'] == 80206)
    cp = torch.load('/root/working/models/v12_f1/v12_f1_best')
    is_warn &= helper_assertion_check("Check v12_f1_best.step == 92874",
                                      cp['step'] == 92874)
    cp = torch.load('/root/working/models/v12_f2/v12_f2_best')
    is_warn &= helper_assertion_check("Check v12_f2_best.step == 95034",
                                      cp['step'] == 95034)


def filecheck_inference_images(working_dir):
    # inputs: dataset directory
    checklist = [
        "dataset/test_rgb/",
    ]

    is_ok = True
    for path_fmt in checklist:
        path = Path(working_dir) / Path(path_fmt)
        is_ok &= __filecheck(path)


def __filecheck(path, max_length=80):
    print(path, end='')
    if len(str(path)) > max_length - 6:
        print('\t', end='')
    else:
        space_size = max_length - 6 - len(str(path))
        print(space_size * ' ', end='')

    if path.exists():
        print('[ \x1b[6;32;40m' + 'OK' + '\x1b[0m ]')
        return True
    else:
        print('[ \x1b[6;31;40m' + 'NG' + '\x1b[0m ]')
        return False


def systemcheck_inference():
    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())


def systemcheck_train():
    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())
    assert helper_assertion_check("Check CUDA device count == 4",
                                  torch.cuda.device_count() == 4)


def helper_assertion_check(msg, res, max_length=80):
    print(msg, end='')
    if len(msg) > max_length - 6:
        print('\t', end='')
    else:
        space_size = max_length - 6 - len(msg)
        print(space_size * ' ', end='')

    if res:
        print('[ \x1b[6;32;40m' + 'OK' + '\x1b[0m ]')
        return True
    else:
        print('[ \x1b[6;31;40m' + 'NG' + '\x1b[0m ]')
        return False


if __name__ == "__main__":
    cli()
