from models.qubvel_segmentation_models.unet.model import Unet
from models.qubvel_segmentation_models.fpn.model import FPN


def make_model(network, freeze_encoder=1, predict_flag=0):
    if predict_flag:
        weights = None
    else:
        weights = 'imagenet'
    
    if network == 'inceptionresnet_unet_borders':
        return  Unet(input_shape=(None, None, 3),
                    activation='sigmoid',
                    backbone_name='inceptionresnetv2',
                    classes=2,
                    encoder_weights=weights,
                    freeze_encoder=bool(freeze_encoder))
    
    elif network == 'inceptionresnet_fpn_borders':
        return  FPN(input_shape=(None, None, 3),
                    activation='sigmoid',
                    backbone_name='inceptionresnetv2',
                    classes=2,
                    encoder_weights=weights,
                    freeze_encoder=bool(freeze_encoder))
    else:
        raise ValueError('unknown network ' + network)

