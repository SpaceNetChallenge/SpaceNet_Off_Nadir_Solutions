import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from adamw import AdamW
from losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import Dpn92_9ch_Unet

from imgaug import augmenters as iaa

from utils import preprocess_inputs, parse_img_id, dice

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

masks_folder = '/wdata/masks'
occluded_masks_dir = '/wdata/masks_occluded'
train_dir = '/wdata/train_png'
train_dir2 = '/wdata/train_png_5_3_0'
train_dir3 = '/wdata/train_png_pan_6_7'
models_folder = '/wdata/weights'

df = pd.read_csv('train_folds.csv')

input_shape = (448, 448)

def shift_image(img, shift_pnt):
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return res

def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return result

def gauss_noise(img, var=30):
    row, col, ch = img.shape
    mean = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')

def clahe(img, clipLimit=2.0, tileGridSize=(5,5)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output

def _blend(img1, img2, alpha):
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')

_alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))
def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)

def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)

def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)

def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)

class TrainData(Dataset):
    def __init__(self, image_ids, epoch_size):
        super().__init__()
        self.image_ids = image_ids
        self.epoch_size = epoch_size
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        fn = self.image_ids[idx] + '.png'

        img = cv2.imread(path.join(train_dir, fn), cv2.IMREAD_COLOR)
        img2 = cv2.imread(path.join(train_dir2, fn), cv2.IMREAD_COLOR)
        if img2 is None:
            print('Error!', fn)
        img3 = cv2.imread(path.join(train_dir3, fn), cv2.IMREAD_COLOR)
        msk = cv2.imread(path.join(masks_folder, fn), cv2.IMREAD_COLOR)
        occluded_msk = cv2.imread(path.join(occluded_masks_dir, fn), cv2.IMREAD_UNCHANGED)
        
        if random.random() > 0.6:
            shift_pnt = (random.randint(-400, 400), random.randint(-400, 400))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            img3 = shift_image(img3, shift_pnt)
            msk = shift_image(msk, shift_pnt)
            occluded_msk = shift_image(occluded_msk, shift_pnt)

        if random.random() > 0.96:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-150, 150), img.shape[1] // 2 + random.randint(-150, 150))
            scale = 1
            angle = random.randint(0, 8) - 4
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                img3 = rotate_image(img3, angle, scale, rot_pnt)
                msk = rotate_image(msk, angle, scale, rot_pnt)
                occluded_msk = rotate_image(occluded_msk, angle, scale, rot_pnt)
                

        crop_size = input_shape[0]
        if random.random() > 0.8:
            crop_size = random.randint(int(input_shape[0] / 1.2), int(input_shape[0] / 0.8))

        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        img3 = img3[y0:y0+crop_size, x0:x0+crop_size, :]
        msk = msk[y0:y0+crop_size, x0:x0+crop_size, :]
        occluded_msk = occluded_msk[y0:y0+crop_size, x0:x0+crop_size, ...]

        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            img3 = cv2.resize(img3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, input_shape, interpolation=cv2.INTER_LINEAR)
            occluded_msk = cv2.resize(occluded_msk, input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.5:
            if random.random() > 0.91:
                img = clahe(img)
                img2 = clahe(img2)
                img3 = clahe(img3)
            elif random.random() > 0.91:
                img = gauss_noise(img)
                img2 = gauss_noise(img2)
                img3 = gauss_noise(img3)
            elif random.random() > 0.91:
                img = cv2.blur(img, (3, 3))
                img2 = cv2.blur(img2, (3, 3))
                img3 = cv2.blur(img3, (3, 3))
        else:        
            if random.random() > 0.91:
                img = saturation(img, 0.9 + random.random() * 0.2)
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
                img3 = saturation(img3, 0.9 + random.random() * 0.2)
            elif random.random() > 0.91:
                img = brightness(img, 0.9 + random.random() * 0.2)
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
                img3 = brightness(img3, 0.9 + random.random() * 0.2)
            elif random.random() > 0.91:
                img = contrast(img, 0.9 + random.random() * 0.2)
                img2 = contrast(img2, 0.9 + random.random() * 0.2)
                img3 = contrast(img3, 0.9 + random.random() * 0.2)

        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)
            img2 = el_det.augment_image(img2)
            img3 = el_det.augment_image(img3)

        msk = (msk > 127) * 1
        occluded_msk = (occluded_msk > 127) * 1
        occluded_msk = occluded_msk[..., np.newaxis]

        msk = np.concatenate([msk, occluded_msk], axis=2)

        img = np.concatenate([img, img2, img3], axis=2)

        img = preprocess_inputs(img)

        nadir, cat_inp, coord_inp = parse_img_id(fn)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()
        nadir = torch.from_numpy(np.asarray([nadir / 60.0]).copy()).float()
        cat_inp = torch.from_numpy(cat_inp.copy()).float()
        coord_inp = torch.from_numpy(coord_inp.copy()).float()
        sample = {"img": img, "mask": msk, 'nadir': nadir, 'cat_inp':cat_inp, 'coord_inp': coord_inp, 'img_name': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_ids):
        super().__init__()
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx] + '.png'

        img = cv2.imread(path.join(train_dir, fn), cv2.IMREAD_COLOR)
        img2 = cv2.imread(path.join(train_dir2, fn), cv2.IMREAD_COLOR)
        img3 = cv2.imread(path.join(train_dir3, fn), cv2.IMREAD_COLOR)
        msk = cv2.imread(path.join(masks_folder, fn), cv2.IMREAD_COLOR)

        msk = (msk > 127) * 1
        msk = msk[..., :2]

        img = np.concatenate([img, img2, img3], axis=2)
        
        img = img[98:-98, 98:-98, ...]
        msk = msk[98:-98, 98:-98, ...]

        img = preprocess_inputs(img)

        nadir, cat_inp, coord_inp = parse_img_id(fn)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()
        nadir = torch.from_numpy(np.asarray([nadir / 60.0]).copy()).float()
        cat_inp = torch.from_numpy(cat_inp.copy()).float()
        coord_inp = torch.from_numpy(coord_inp.copy()).float()
        sample = {"img": img, "mask": msk, 'nadir': nadir, 'cat_inp':cat_inp, 'coord_inp': coord_inp, 'img_name': fn}
        return sample

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(net, data_loader):
    dices = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            mask = sample["mask"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            cat_inp = sample["cat_inp"].cuda(non_blocking=True)
            coord_inp = sample["coord_inp"].cuda(non_blocking=True)
            nadir = sample["nadir"].cuda(non_blocking=True)
            out, nadir_pred = model(imgs, nadir, cat_inp, coord_inp)
            probs = torch.sigmoid(out)
            pred = probs.cpu().numpy() > 0.5
            for j in range(mask.shape[0]):
                dices.append(dice(mask[j, 0], pred[j, 0]))

    d = np.mean(dices)
    print("Val Dice: {0}".format(d))
    return d

def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)
    d = np.mean(d)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name))

        best_score = d
    print("dice: {}\tdice_best: {}".format(d, best_score))
    return best_score

def train_epoch(current_epoch, loss_function, l1_loss, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    losses4 = AverageMeter()
    nadir_losses = AverageMeter()
    dices = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        cat_inp = sample["cat_inp"].cuda(non_blocking=True)
        coord_inp = sample["coord_inp"].cuda(non_blocking=True)
        masks = sample["mask"].cuda(non_blocking=True)
        nadir = sample["nadir"].cuda(non_blocking=True)
        out, nadir_pred = model(imgs, nadir, cat_inp, coord_inp)

        loss1 = loss_function(out[:, 0, ...], masks[:, 0, ...])
        loss2 = loss_function(out[:, 1, ...], masks[:, 1, ...])
        loss3 = loss_function(out[:, 2, ...], masks[:, 2, ...])
        loss4 = loss_function(out[:, 3, ...], masks[:, 3, ...])
        nadir_loss = l1_loss(nadir_pred, nadir)
        loss = loss1 + 0.4 * loss2 + 0.05 * loss3 + 0.005 * loss4 + 0.002 * nadir_loss #  

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, masks[:, 0, ...])

        losses.update(loss1.item(), imgs.size(0))
        losses2.update(loss2.item(), imgs.size(0))
        losses3.update(loss3.item(), imgs.size(0))
        losses4.update(loss4.item(), imgs.size(0))
        nadir_losses.update(nadir_loss.item(), imgs.size(0))
        dices.update(dice_sc, imgs.size(0))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss2 {loss2.val:.4f} ({loss2.avg:.4f}); Loss3 {loss3.val:.4f} ({loss3.avg:.4f}); Loss4 {loss4.val:.4f} ({loss4.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f}); Nadir {nadir_loss.val:.4f} ({nadir_loss.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss2=losses2, loss3=losses3, loss4=losses4, dice=dices, nadir_loss=nadir_losses))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Loss2 {loss2.avg:.4f}; Loss3 {loss3.avg:.4f}; Loss4 {loss4.avg:.4f}; Dice {dice.avg:.4f}; Nadir {nadir_loss.avg:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss2=losses2, loss3=losses3, loss4=losses4, dice=dices, nadir_loss=nadir_losses))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    
    fold = int(sys.argv[1])

    #hardcoded for training env:
    vis_dev = str(fold)
    if fold > 3:
        vis_dev = str(fold - 4)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    cudnn.benchmark = True
    
    all_files = sorted(listdir(train_dir))

    batch_size = 5
    val_batch = 2

    best_snapshot_name = 'dpn92_9ch_fold_{0}_best_0'.format(fold)
    last_snapshot_name = 'dpn92_9ch_fold_{0}_last_0'.format(fold)

    np.random.seed(34)
    random.seed(34)

    train_files = df[df['fold'] != fold]['id'].values
    val_files = df[df['fold'] == fold]['id'].values

    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // val_batch

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_files, len(train_files))
    val_train = ValData(val_files)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_train, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

    model = nn.DataParallel(Dpn92_9ch_Unet()).cuda()

    params = model.parameters()

    optimizer = AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 12, 22], gamma=0.5)
    loss_function = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=True).cuda()
    
    l1_loss = torch.nn.SmoothL1Loss().cuda()

    best_score = 0
    for epoch in range(25):
        train_epoch(epoch, loss_function, l1_loss, model, optimizer, scheduler, train_data_loader)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, path.join(models_folder, last_snapshot_name))
        torch.cuda.empty_cache()
        best_score = evaluate_val(val_data_loader, best_score, model, best_snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))