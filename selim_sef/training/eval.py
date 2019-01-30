import os
import warnings

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from training.metric import dice

warnings.simplefilter("ignore")


def postprocess(mask, min_size=10):
    if np.sum(mask) < min_size:
        return np.zeros_like(mask)
    else:
        return mask


pred_dir = "../spacenet_preds_d161"


def validate(net, data_loader):
    os.makedirs(pred_dir, exist_ok=True)
    body_dices = []
    border_dices = []
    sep_dices = []
    three_channels = False
    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = Variable(sample["img"])
            mask = sample["mask"].numpy()
            img_names = sample["img_name"]
            imgs = imgs.cuda()
            logits = net((imgs, sample["angle"]))
            pred = torch.sigmoid(logits).cpu().numpy()
            three_channels = mask.shape[-3] == 3
            for i in range(mask.shape[0]):
                pred_mask = np.zeros((928, 928, 3))
                pred_mask[...,0] = pred[i][0, ...]
                pred_mask[...,1] = pred[i][1, ...]
                pred_mask[...,2] = pred[i][2, ...]
                path = os.path.join(pred_dir, img_names[i][:-4].split("/")[-1] + ".png")
                cv2.imwrite(path, pred_mask * 255)
            pred = 1 * (pred > 0.5)
            for i in range(mask.shape[0]):
                body_dices.append(dice(mask[i, 0, ...], postprocess(pred[i, 0, ...], 15)))
                border_dices.append(dice(mask[i, 1, ...], postprocess(pred[i, 1, ...], 5)))
                if three_channels:
                    sep_dices.append(dice(mask[i, 2, ...], postprocess(pred[i, 2, ...], 5)))


    print("\nDice = {}".format(np.mean(body_dices)))
    print("Border Dice = {}".format(np.mean(border_dices)))
    if three_channels:
        print("Sep Dice = {}".format(np.mean(sep_dices)))
    score = np.mean(body_dices)

    return score
