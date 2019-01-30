import argparse
import os
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import zoo
from dataset.dense_data import TestDenseData
from inference.tta import transforms1, transforms_ms
from tools.config import load_config
from training.utils import get_model_params, create_transforms

warnings.simplefilter("ignore")


def predict_tta(model, batch, apply_sigmoid, transforms):
    batch = (batch[0].cpu().numpy(), batch[1])
    ret = []
    for cls in transforms:
        ret.append(cls(apply_sigmoid)(model, batch))
    out = np.moveaxis(np.mean(ret, axis=0), 1, -1)
    return out

def get_nadir(f):
    return  int(f.split("_")[2][5:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PyTorch Segmentation Predictor")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--data-path', type=str, help='Path to test images')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, help='Path for predicted masks')
    arg('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = load_config(args.config)
    num_classes = conf["segnetwork"]['seg_classes']

    train_transforms, val_transforms = create_transforms(conf["input"])

    dirs = os.listdir(args.data_path)

    params = get_model_params({**conf["segnetwork"], **conf["network"]})
    model = zoo.__dict__[conf['arch']](**params)

    model = torch.nn.DataParallel(model).cuda()
    weights = args.resume
    print("=> loading checkpoint '{}'".format(weights))
    checkpoint = torch.load(weights)
    miou_best = checkpoint['best_miou']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))
    os.makedirs(args.output_dir, exist_ok=True)

    model.eval()
    for dir in dirs:
        data_test = TestDenseData(data_path=os.path.join(args.data_path, dir), transform=val_transforms)
        test_data_loader = DataLoader(data_test, batch_size=2, num_workers=8, shuffle=False, pin_memory=False)

        with torch.no_grad():
            for sample in tqdm(test_data_loader):
                images = sample["img"]
                img_names = sample["img_name"]
                nadir = get_nadir(os.path.basename(sample["img_name"][0]))
                if nadir < 30:
                    ttas = transforms_ms
                else:
                    ttas = transforms1
                preds = predict_tta(model, (images, sample["angle"]), apply_sigmoid=True, transforms=ttas)
                preds = preds[:, 14:-14, 14:-14, :]
                for i in range(preds.shape[0]):
                    pred = preds[i]
                    mask = pred * 255
                    cv2.imwrite(
                        os.path.join(args.output_dir, "{}.png".format(os.path.basename(sample["img_name"][i])[:-4])),
                        mask)
