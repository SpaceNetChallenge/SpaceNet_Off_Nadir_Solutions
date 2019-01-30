import argparse
import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import zoo
from dataset.dense_data import DenseData
from tools.config import load_config
from training import utils
from training.eval import validate
from training.losses import dice_round, ComboLoss
from training.meters import AverageMeter

torch.backends.cudnn.benchmark = True


def get_model_name(model, num_classes, snapshot_prefix, dataset_name):
    snapshot_name = "{}_{}_{}".format(model, dataset_name, num_classes)
    if snapshot_prefix is not None:
        snapshot_name = "{}_{}".format(snapshot_prefix, snapshot_name)
    return snapshot_name
parser = argparse.ArgumentParser("PyTorch Segmentation Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--csv', type=str, help='path to csv folds')
arg('--nadir', type=str, help='Nadir type')
arg('--fold', type=int, default=0, help='fold number')
arg('--data-path', type=str, default='data/mappilary', help='Path to dataset folder')
arg('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
arg('--test-epoch', type=int, default=1, help='Test epoch')
arg('--workers', type=int, default=8, help='number of cpu threads to use')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default=None, help='Path for snapshot')
arg('--snapshot-prefix', type=str, default=None, help='Prefix of snapshot nams')
arg('--test-init', type=bool, default=False, help='Test initialization')
arg('--fp16', help='Test initialization', action='store_true')


args = parser.parse_args()

if args.fp16:
    from apex import amp
    amp_handle = amp.init()


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conf = load_config(args.config)
    num_classes = conf["segnetwork"]['seg_classes']
    snapshot_name = get_model_name(conf['arch'] + "_" + conf["segnetwork"]['backbone_arch'], num_classes,
                                   args.nadir, "fold_{}".format(args.fold))

    params = utils.get_model_params({**conf["segnetwork"], **conf["network"]})
    model = zoo.__dict__[conf['arch']](**params)
    model = DataParallel(model.cuda()).cuda()

    loss_function = ComboLoss(**conf["loss"]).cuda()
    optimizer, scheduler = utils.create_optimizer(conf["optimizer"], model)
    miou_best = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            miou_best = checkpoint['best_miou']
            state_dict = checkpoint['state_dict']
            if conf['optimizer'].get('zero_decoder', False):
                for key in state_dict.copy().keys():
                    if not key.startswith("module.encoder"):
                        del state_dict[key]
            for key in state_dict.keys():
                print(key)

            model.load_state_dict(state_dict, strict=False)
            if not conf['optimizer'].get('start_zero', False):
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if conf['optimizer'].get('start_zero', False):
        start_epoch = 0
        miou_best = 0
    train_transforms, val_transforms = utils.create_transforms(conf["input"])
    cudnn.benchmark = True
    batch_size = conf["optimizer"]["batch_size"]

    data_train = DenseData(csv_path=args.csv, data_path=args.data_path, mode='train', fold=args.fold, transform=train_transforms, nadir=args.nadir)
    data_val = DenseData(csv_path=args.csv, data_path=args.data_path, mode='val', fold=args.fold, transform=val_transforms, nadir=args.nadir)
    val_data_loader = DataLoader(data_val, batch_size=2, num_workers=args.workers, shuffle=False, pin_memory=False)
    current_epoch = start_epoch
    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=False, drop_last=True)

    for epoch in range(start_epoch, conf["optimizer"]["schedule"]["epochs"]):
        if epoch < 1 and conf['optimizer'].get('freeze_first_epoch', False):
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = False
            for p in model.module.encoder_stages[0].parameters():
                p.requires_grad = True
        else:
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = True



        train_epoch(args, conf, current_epoch, loss_function, model, optimizer, scheduler, train_data_loader)
        model = model.eval()
        torch.save({
            'epoch': current_epoch + 1,
            'arch': conf["segnetwork"]["backbone_arch"],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': miou_best,
        }, args.output_dir + snapshot_name + "_last")
        if args.test_epoch is not None and current_epoch % args.test_epoch == 0 and (current_epoch > 0 or args.test_epoch == 1):
            miou_best = evaluate_val(args, val_data_loader, miou_best, model, snapshot_name, conf, current_epoch,
                                     optimizer)

        current_epoch += 1


def evaluate_val(args, data_val, miou_best, model, snapshot_name, conf, current_epoch, optimizer):
    print("Test phase")
    model = model.eval()
    ious = validate(model, data_loader=data_val)
    miou = np.mean(ious)

    if miou > miou_best:
        if args.output_dir is not None:
            torch.save({
                'epoch': current_epoch + 1,
                'arch': conf["segnetwork"]["backbone_arch"],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': miou,
            }, args.output_dir + snapshot_name + "_best")
        miou_best = miou
    torch.save({
        'epoch': current_epoch + 1,
        'arch': conf["segnetwork"]["backbone_arch"],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou': miou,
    }, args.output_dir + snapshot_name + "_last")
    print("miou: {}\tmiou_best: {}".format(miou, miou_best))
    return miou_best


def train_epoch(args, conf, current_epoch, loss_function, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    dices = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * len(train_data_loader))
        imgs = sample["img"].cuda(async=True)
        angle = sample["angle"].cuda(async=True)
        masks = sample["mask"].cuda(async=True)
        num_classes = conf['segnetwork']['seg_classes']
        masks = masks[:, :num_classes]
        out = model((imgs, angle))
        loss = loss_function(out, masks)
        with torch.no_grad():
            dice = 1 - dice_round(torch.sigmoid(out[:,0,...]), masks[:,0,...])
        losses.update(loss.item(), imgs.size(0))
        dices.update(dice, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))
        optimizer.zero_grad()
        if not args.fp16:
            loss.backward()
        else:
            with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                 scaled_loss.backward()
        if conf["optimizer"]["clip"] != 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf["optimizer"]["clip"])
        optimizer.step()


if __name__ == '__main__':
    main()
