import json

DEFAULTS = {
    "arch": "fpn_resnext",

    "segnetwork": {
        "backbone_arch": "resnext101",
        "seg_classes": 2,
        "ignore_index": 255,
    },
    "network": {

    },
    "optimizer": {
        "batch_size": 256,
        "freeze_first_epoch": False,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 0,
        "clip": 1.,
        "learning_rate": 0.1,
        "classifier_lr": -1.,  # If -1 use same learning rate as the rest of the network
        "nesterov": True,
        "schedule": {
            "type": "constant",  # supported: constant, step, multistep, exponential, linear, poly
            "mode": "epoch",  # supported: epoch, step
            "epochs": 10,
            "params": {}
        }
    },
    "input": {
        "scale_train": -1,  # If -1 do not scale
        "random_vh_shift": 0,
        "crop_train": 224,
        "color_jitter_train": False,
        "lighting_train": False,
        "random_crop": [202, 202],
        "crop_size_range": [1., 1.],
        "rescale_prob": 0.0,
        "mask_downscale_factor": 1,
        "padding_block": 0,
        "padding_mode": 'reflect',
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config
