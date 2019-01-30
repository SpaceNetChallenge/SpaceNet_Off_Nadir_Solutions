import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import Compose

from dataset.dense_transform import Normalize, Rotate90, VFlip, Pad, RandomRotate
from dataset.dense_transform import RandomCropAndScale, HFlip, ToTensor, ColorJitterImage, LightingImage
from tools.adamw import AdamW
from tools.clr import CyclicLR
from tools.lr_policy import PolyLR


def get_model_params(network_config):
    """Convert a configuration to actual model parameters

    Parameters
    ----------
    network_config : dict
        Dictionary containing the configuration options for the network.

    Returns
    -------
    model_params : dict
        Dictionary containing the actual parameters to be passed to the `net_*` functions in `models`.
    """
    model_params = {}
    model_params["seg_classes"] = network_config["seg_classes"]
    model_params["backbone_arch"] = network_config["backbone_arch"]
    return model_params


def create_optimizer(optimizer_config, model, master_params=None):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config["classifier_lr"] != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append(v)
            else:
                classifier_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        if master_params:
            params = master_params
        else:
            params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler


def create_transforms(input_config):
    """Create transforms from configuration

    Parameters
    ----------
    input_config : dict
        Dictionary containing the configuration options for input pre-processing.

    Returns
    -------
    train_transforms : list
        List of transforms to be applied to the input during training.
    val_transforms : list
        List of transforms to be applied to the input during validation.
    """

    train_transforms = []
    if input_config.get('random_rotate', None):
        train_transforms.append(RandomRotate(input_config['random_rotate']['angle'], input_config['random_rotate']['prob']))

    if input_config.get('random_crop', None):
        train_transforms.append(RandomCropAndScale(input_config['random_crop'][0], input_config['random_crop'][1], scale_range=input_config['crop_size_range'], rescale_prob=input_config['rescale_prob'], prob=1))

    train_transforms += [
        # HFlip(),
        # Rotate90(),
        # VFlip(),
        ToTensor(),
    ]
    if input_config.get("color_jitter_train", False):
        train_transforms.append(ColorJitterImage())

    val_transforms = []
    val_transforms += [
        Pad(),
        ToTensor(),
    ]

    return Compose(train_transforms), Compose(val_transforms)
