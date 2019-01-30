import sys
import numpy as np
np.random.seed(10)
sys.path.insert(0,'./Mask_RCNN/')
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model_mod_mpan as modellib_mpan
from mrcnn import model_mod_rgb as modellib_rgb


class SpacenetConfig(Config):
    NAME = ''
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # background + buildings
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320
    IMAGE_RESIZE_MODE = "pad64"
    IMAGESHAPE = np.array([320,320,3])
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    RPN_ANCHOR_SCALES = (8,16,32,64,128)
    RPN_ANCHOR_STRIDE = 1
    RPN_ANCHOR_RATIOS = [.5,1.,2.]
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = .33
    RPN_NMS_THRESHOLD = .7
    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_NMS_THRESHOLD = 0.5
    BACKBONE_STRIDES = [2,4, 8, 16, 32]
    DILATION = [5]
    STEPS_PER_EPOCH = 0
    VALIDATION_STEPS = 0
    USE_MINI_MASK = True
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    MAX_GT_INSTANCES = 53
    DETECTION_MAX_INSTANCES = 53
    WEIGHT_DECAY = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    POSITIVE_TYPE = 'shuffle'

image_per_gpu=2  #p3.2xlarge 
class SpacenetConfigIRGB_u16(SpacenetConfig):
    IMAGES_PER_GPU = image_per_gpu #for aws p3.2xlarge
    NUM_CLASSES = 1 + 1  # background + buildings
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = np.array([512,512,4])
    IMAGE_CHANNEL_COUNT = 4
    MEAN_PIXEL = np.asarray([1450.23,  782.24,  650.68, 469.42])
    MAX_GT_INSTANCES =130
    DETECTION_MAX_INSTANCES=130
    USE_MINI_MASK = True


class SpacenetConfigRGB_u8(SpacenetConfig):
    IMAGES_PER_GPU = image_per_gpu#for aws p3.2xlarge
    NUM_CLASSES = 1 + 1  # background + buildings
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = np.array([512,512,3])
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MAX_GT_INSTANCES =130
    DETECTION_MAX_INSTANCES=130
    USE_MINI_MASK = True


class SpacenetConfigMPAN_u16(SpacenetConfig):
    IMAGES_PER_GPU = image_per_gpu #for aws p3.2xlarge
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = np.array([512, 512, 3])
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([397.31020501, 364.65232488, 223.70400872])
    MAX_GT_INSTANCES = 130
    DETECTION_MAX_INSTANCES = 130
    USE_MINI_MASK = True


class SpacenetConfigMPAN_u8(SpacenetConfig):
    IMAGES_PER_GPU = image_per_gpu #for aws p3.2xlarge
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = np.array([512, 512, 3])
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([85.8, 82.0, 55.4])
    MAX_GT_INSTANCES = 130
    DETECTION_MAX_INSTANCES = 130
    USE_MINI_MASK = True
