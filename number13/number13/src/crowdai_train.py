import sys
import os
import gc
import numpy as np
np.random.seed(10)
sys.path.insert(0,'./Mask_RCNN/')
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model_mod_rgb as modellib
from pycocotools.coco import COCO
from imgaug import augmenters as iaa
import glob
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore')
ROOT_DIR = os.getcwd()
from skimage import io

DEBUG = False
if DEBUG:
  TRAIN_STEP = 2
  VAL_STEP = 1

class CrowdAiConfig(Config):
    # Give the configuration a recognizable name
    NAME = "crowdai_"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320
    IMAGE_RESIZE_MODE = "pad64"
    #IMAGE_MIN_SCALE = 1
    IMAGESHAPE = np.array([320,320,3])
    IMAGE_CHANNEL_COUNT = 3
    # Image mean (RGB)
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
    MAX_GT_INSTANCES = 34
    DETECTION_MAX_INSTANCES = 34
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    POSITIVE_TYPE = 'shuffle'


class CrowdAiDataset(utils.Dataset):
    def load_data(self, coco,imgIds,dataDir ):
        self.coco=coco
        self.dataDir=dataDir
        self.imgIds=imgIds
        self.add_class("buildings", 1, "b")
        for item in imgIds:
            self.add_image("buildings", image_id=item, path=None)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img = self.coco.loadImgs(self.imgIds[image_id])[0]
        ids='{}/{}/{}'.format(self.dataDir,'images',img['file_name'])
        image = io.imread(ids)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        print(info)
        if info["source"] == "buildings":
            return info["buildings"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        annIds = self.coco.getAnnIds(imgIds=self.imgIds[image_id], catIds=[], iscrowd=None)
        if type(annIds)==int:
            annIds = [annIds]
        anns = self.coco.loadAnns(annIds)
        anns = [item for item in anns if item['area']!=0]
        masks=np.stack([self.coco.annToMask(item) for item in anns],axis=-1)
        _, _, num_masks = masks.shape
        class_ids = np.asarray([1] * num_masks)
        return masks, class_ids.astype(np.int32)


def get_dataset(Dir, fltarea=None):
    annFile = '{}/annotation.json'.format(Dir)
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['buildings'])
    imgIds = coco.getImgIds(catIds=catIds)

    def filter_ids(ids,fltarea):
        newIds=[]
        for id in ids:
            annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
            anns = coco.loadAnns(annIds)
            anns = [item for item in anns if item['area'] != 0]
            if len(anns)!=0 and fltarea!=None:
                if fltarea=='small':
                    anns = [item for item in anns if item['area'] < 256]
                    if len(anns)!=0:
                        newIds.append(id)
                if fltarea=='large':
                    annsbool = [(item['area'] >256) for item in anns]
                    if all(annsbool):
                        newIds.append(id)
            if len(anns) != 0 and fltarea == None:
                newIds.append(id)
        return newIds
    #if 'train' in Dir:
    print ('removing area==0 and filtering images with {} mask'.format(fltarea))
    imgIds = filter_ids(imgIds,fltarea=fltarea)
    dataset = CrowdAiDataset()
    dataset.load_data(coco, imgIds, Dir)
    dataset.prepare()
    del coco,imgIds
    gc.collect()
    return dataset


def train(init_with="coco", weights=None, fine=0, last_epoch=0, epochs=1,gpus=1):
    class TrainConfig(CrowdAiConfig):
           GPU_COUNT = gpus
    config=TrainConfig()
    config.DILATION=[5]
    print (config.display())
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
    if init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        print ('Loading coco model')
        model.load_weights(weights, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                      "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        if weights is None:
            weight_path=model.find_last()[1]
        else:
            weight_path = weights
        print ('loaded weight ..',weights)
        # Load the last model you trained and continue training
        model.load_weights(weight_path, by_name=True)
    dataset_val = get_dataset(''.join([HOME_DIR,'/val']),fltarea=None)
    dataset_train = get_dataset(''.join([HOME_DIR,'/train']),fltarea=None)
    print('train samples::{}'.format(dataset_train.num_images), 'valid samples::{}'.format(dataset_val.num_images))
    config.STEPS_PER_EPOCH=int(dataset_train.num_images/ (config.IMAGES_PER_GPU*config.GPU_COUNT))
    config.VALIDATION_STEPS = int(dataset_val.num_images/ (config.IMAGES_PER_GPU*config.GPU_COUNT))
    print ('STEPS_PER_EPOCH', config.STEPS_PER_EPOCH)
    print ('VALIDATION_STEPS', config.VALIDATION_STEPS)
    if DEBUG:
        config.STEPS_PER_EPOCH = TRAIN_STEP
        config.VALIDATION_STEPS = VAL_STEP
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.3)
    ])
    model.epoch = last_epoch
    model.train(dataset_train, dataset_val,
                  learning_rate=config.LEARNING_RATE,
                  epochs=epochs,
                  layers='all',augmentation=augmentation)
    
    if fine>0:
       print ('current lr:', config.LEARNING_RATE)
       print ('new lr :', config.LEARNING_RATE / 10)
       model.train(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE/10 ,
                      epochs=model.epoch+fine,
                      layers="all",augmentation=augmentation)

    del model,dataset_train,dataset_val
    gc.collect()


def copy_final_model(): # Just so that spacenet finds the final required model model
    from config import MODEL_DIR
    out_dir = os.path.join(MODEL_DIR,'external_pretrained')
    final_model = glob.glob(''.join([out_dir,'/*/*.h5'])) 
    final_model = [item for item in final_model if int(item.split('/')[-1].split('_')[-1].replace('.h5',''))==8][0]
    shutil.copy2(final_model, os.path.join(out_dir,'crowdai_final.h5'))


if __name__=='__main__':
    print ('training mask_rcnn')
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--weights', required=True,
                        help='start with weights provide coco if coco')  # change required to true
    parser.add_argument('--home_dir', required=True, help='train val  dir')
    parser.add_argument('--model_output', required=False, help='Store models in which dir?')
    parser.add_argument('--gpus', required=False, default=1, type=int, help='num gpus to use')
    parser.add_argument('--is_coco', action = 'store_true', default=True)
    parser.add_argument('--fine',required=False,type=int, default=4)
    parser.add_argument('--from_epoch',required=False, default=0 ,type=int)
    parser.add_argument('--to_epoch',required=False,default=4, type=int)

    args = parser.parse_args()
    weights = args.weights
    HOME_DIR = args.home_dir
    MODEL_DIR = args.model_output
    gpus = args.gpus
    is_coco= args.is_coco
    from_epoch = args.from_epoch
    to_epoch = args.to_epoch
    gpus = args.gpus
    fine = args.fine
    if is_coco:
        init_with = 'coco'
    else:
        init_with='last'
    train(init_with=init_with, weights=weights,fine=fine,last_epoch=from_epoch, epochs=to_epoch,gpus = gpus)
    copy_final_model() # so that spacenet finds it
    #python3 crowdai_train.py --weights ../crowdai_data/mask_rcnn_coco.h5 --home_dir ../crowdai_data 
    #--model_output ../spacenet_models/external_pretrained/ --gpus 1 
