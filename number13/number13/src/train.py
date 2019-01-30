import sys
import numpy as np
np.random.seed(10)
import tifffile
sys.path.insert(0,'./Mask_RCNN/')
from mrcnn import utils
import glob
from pycocotools.coco import COCO
from skimage import io
from imgaug import augmenters as iaa
import pickle
import os
import gc
from keras import backend as K
import util
import argparse
from config import MODEL_IRGB_MAIN, MODEL_MPAN_MAIN, MODEL_RGB_U8, MODEL_MPAN_U8,MODEL_DIR
from config import TRAIN_DATA_DIR_MPAN, TRAIN_DATA_DIR_IRGB
from config import NADIR, OFF_NADIR, VERY_NADIR
from models import modellib_mpan, modellib_rgb
from models import SpacenetConfigRGB_u8, SpacenetConfigMPAN_u8
from models import SpacenetConfigMPAN_u16, SpacenetConfigIRGB_u16
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s %(message)s', filename='train.log', level=logging.DEBUG)
val_set = pickle.load(open('val_set.pkl','rb'))
annotation_prefix='Atlanta_nadir7_catid_1030010003D22F00'
im_extension = '.tif'
#Limit train and validation data to check if all models are produced by training steps and kept appropriately in the Directories as expected
DEBUG = False
if DEBUG:
    TRAIN_STEP =2
    VALIDATION_STEPS =1


class SpacenetDataset(utils.Dataset):
    def load_data(self, coco,imgIds, dataDir, group='rgb',is_uint16 = False):
        # Add classes
        self.coco=coco
        self.dataDir=dataDir
        self.imgIds=imgIds
        self.is_uint16=is_uint16
        self.group = group
        self.add_class("buildings", 1, "b")
        for i,item in enumerate(imgIds):
            subdir ='_'.join(item.split('_')[0:-3])
            self.add_image("buildings", image_id=i, path=''.join([dataDir,subdir,'/',item]))

    def load_image(self, image_id):
        info = self.image_info[image_id]
        if not self.is_uint16:
            image = tifffile.imread(info['path'])
            if self.group=='rgb':
                if image.shape[-1]>3:
                    image=image[:,:,1::]
            image = util.stretch_8bit(image,lower_percent=2,higher_percent=8)

        else:
            image = tifffile.imread(info['path'])

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
        imgIds = self.imgIds[image_id].split('_')[-3::]
        imgIds = '_'.join(imgIds)
        imgIds_to7 = '_'.join([annotation_prefix, imgIds]).replace(im_extension, '')
        #fix for pycocotools differing in how it gets annotaion in py35 and py36
        if sys.version_info[1] == 6:
           imgIds_to7 = [imgIds_to7]
        annIds = self.coco.getAnnIds(imgIds=imgIds_to7, catIds=[], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        anns = [item for item in anns if item['area']!=0]
        masks=np.stack([self.coco.annToMask(item) for item in anns],axis=-1)
        _, _, num_masks = masks.shape
        class_ids = np.asarray([1] * num_masks)
        return masks, class_ids.astype(np.int32)


def get_train_val_split(dir, angles, is_uint16=False):
    def split(images):
        images = [item.split('/')[-1] for item in images]
        val_imgids = []
        for item in images:
            if '_'.join(item.split('_')[-3::]).replace(im_extension, '') in val_set:
                val_imgids.append(item)
        train_imgids = np.setdiff1d(images, val_imgids)
        return train_imgids,val_imgids
    images=[]
    if angles[0]=='all':
        for sub_dir in glob.glob(''.join([dir,'/*/'])):
            images.extend(glob.glob(''.join([sub_dir, '/*',im_extension])))
        images = [item.split('/')[-1] for item in images]
        train_imgids, val_imgids = split(images)
        return train_imgids, val_imgids
    elif angles[0]=='nadir':
        nadir_type = NADIR
    elif angles[0]=='off':
        nadir_type = OFF_NADIR
    elif angles[0]=='very':
        nadir_type = VERY_NADIR
    else:
        nadir_type = angles
    for angle in nadir_type:
        dir_path=''.join([dir,'/',angle,'/'])
        images.extend(glob.glob(''.join([dir_path, '/*',im_extension])))
    train_imgids, val_imgids = split(images)
    return train_imgids, val_imgids


def get_datasets(annFile, angles, is_uint16=False, group = 'rgb'):
    coco = COCO(annFile)
    if group =='mpan':
        datadir = TRAIN_DATA_DIR_MPAN
    if group == 'rgb':
        datadir = TRAIN_DATA_DIR_IRGB
    train_imgids, val_imgids = get_train_val_split(datadir, angles,is_uint16=is_uint16)
    if DEBUG:
       val_imgids = train_imgids
    logging.info('Using datadir.. {}'.format(datadir))
    dataset_train = SpacenetDataset()
    dataset_train.load_data(coco, train_imgids, datadir, group=group, is_uint16=is_uint16)
    dataset_train.prepare()
    dataset_val = SpacenetDataset()
    dataset_val.load_data(coco, val_imgids, datadir, group=group, is_uint16=is_uint16)
    dataset_val.prepare()
    return dataset_train,dataset_val,coco


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth=True
    K.set_session(K.tf.Session(config=cfg))


def train(annFile, angle, previous_weights, group='rgb',is_uint16=False, exclude=None, epochs=1,
          do_fine=0, init_epoch=0, gpus=1):
    gc.enable()
    if group=='rgb':
        if is_uint16:
            class TrainConfig(SpacenetConfigIRGB_u16):
                GPU_COUNT = gpus
                NAME = angle[0]+'_'
            config=TrainConfig()
            MODEL_DIR = MODEL_IRGB_MAIN
        else:
            class TrainConfig(SpacenetConfigRGB_u8):
                GPU_COUNT = gpus
                NAME = angle[0] + '_'
            config = TrainConfig()
            MODEL_DIR = MODEL_RGB_U8
        model = modellib_rgb.MaskRCNN(mode="training", config=config,
                                       model_dir=MODEL_DIR)
    if group=='mpan':
        if is_uint16:
            class TrainConfig(SpacenetConfigMPAN_u16):
                GPU_COUNT = gpus
                NAME = angle[0]+'_'
            config=TrainConfig()
            MODEL_DIR = MODEL_MPAN_MAIN
        else:
            class TrainConfig(SpacenetConfigMPAN_u8):
                GPU_COUNT = gpus
                NAME = angle[0]+'_'
            config=TrainConfig()
            MODEL_DIR = MODEL_MPAN_U8
        model = modellib_mpan.MaskRCNN(mode="training", config=config,
                              model_dir = MODEL_DIR)

    logging.info('train using weights {} for angle {} is_uin16={}'.format(previous_weights, angle,is_uint16))
    model.load_weights(previous_weights,exclude=exclude, by_name=True)
    dataset_train, dataset_val, _ = get_datasets(annFile, angle, is_uint16=is_uint16, group=group)
    config.STEPS_PER_EPOCH = int(dataset_train.num_images / (config.IMAGES_PER_GPU*config.GPU_COUNT))
    config.VALIDATION_STEPS = int(dataset_val.num_images / (config.IMAGES_PER_GPU*config.GPU_COUNT))
    logging.info('train samples::{}..valid samples::{}'.format(dataset_train.num_images,dataset_val.num_images))
    logging.info('STEPS_PER_EPOCH::{}..VALIDATION_STEPS::{}'.format(config.STEPS_PER_EPOCH,config.VALIDATION_STEPS))
    if DEBUG:
        config.STEPS_PER_EPOCH = TRAIN_STEP
        config.VALIDATION_STEPS = VALIDATION_STEPS
    augmentation = iaa.Fliplr(0.5)
    model.epoch = init_epoch
    if epochs!=0:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    layers='all', augmentation=augmentation)
    if do_fine!=0:
        learning_rate = config.LEARNING_RATE / 10
        model.epoch = epochs+init_epoch
        model.train(dataset_train, dataset_val,
                    learning_rate=learning_rate,
                    epochs=model.epoch+do_fine,
                    layers="all", augmentation=augmentation)

    path_to_saved_weights = model.find_last()[1]
    logging.info('finished training {} for {} epochs'.format(angle,model.epoch))
    #limit_mem()
    K.clear_session()
    del model, dataset_train, dataset_val
    gc.collect()
    return path_to_saved_weights


def train_cascade(annFile, initial_weight, angles, train_type, group='rgb', is_uint16=False,epochs=1 , gpus=1):

    if train_type=='fine_tune_angle':
        try:
            if 'very' in angles:
                for angle in VERY_NADIR: #should be sorted in increasing angle
                    initial_weight = train(annFile, [angle], initial_weight, group=group, is_uint16=is_uint16,epochs=epochs,
                                                     do_fine=1,gpus=gpus)
            if 'off' in angles:
                for angle in OFF_NADIR: #should be sorted in increasing angle
                    initial_weight  = train(annFile, [angle], initial_weight, group=group, is_uint16=is_uint16,epochs=epochs,
                                                     do_fine=1,gpus=gpus)
            if 'nadir' in angles:
                for angle in NADIR: #should be sorted in increasing angle
                    initial_weight = train(annFile, [angle], initial_weight, group=group, is_uint16=is_uint16,epochs=epochs,
                                                   do_fine=1,gpus=gpus)
        except:
            logging.error('Error in cascade train while doing {}'.format(angles))
    else:
        for angle in angles:
            _  = train(annFile, [angle], initial_weight, group=group, is_uint16=is_uint16,epochs=epochs,
                                   do_fine=0, gpus=gpus)
    return initial_weight


def train_extra(annfile,group,model_dir):
    if group=='mpan':
        angles=VERY_NADIR
    if group=='rgb':
        angles=np.concatenate([NADIR,OFF_NADIR]).tolist()

    def get_weights_angle_specific(use_epoch, model_dir_old):
        weights = {}
        for item in glob.glob(''.join([model_dir_old, '*/'])):
            key = item.split('/')[-2]
            # strip time stamps
            key = '_'.join(key.split('_')[0:-1])
            ww = glob.glob(''.join([item, '/*.h5']))
            if len(ww) == 1:
                weights[key] = ww[0]
            else:
                for w in ww:
                    if int(w.split('/')[-1].split('_')[-1].replace('.h5', '').replace('_', '')) == use_epoch:
                        weights[key] = w
        return weights
    weights = get_weights_angle_specific(use_epoch=2,  model_dir_old=model_dir)
    for angle in angles:
        wn = weights[angle.lower()]
        train(annfile, [angle], wn, group=group, is_uint16=True, epochs=0, init_epoch=2, do_fine=2, gpus=gpus)
    print('done {} training ...'.format(group))


def train_all(pretrained_weights,gpus=1):
    annfile_mpan = ''.join([TRAIN_DATA_DIR_MPAN, '/', annotation_prefix, '_annotation.json'])
    annfile_irgb = ''.join([TRAIN_DATA_DIR_IRGB, '/', annotation_prefix, '_annotation.json'])
    #train_initial_rgb_nadir7-> from crowdai_pretrained train on nadir 7 only for 4 epoch at lr=0.001
    #and 1 epoch at lr=0.0001 (uint8_rgb_nadir7)
    weights = train(annfile_irgb,['Atlanta_nadir7_catid_1030010003D22F00'], pretrained_weights, group='rgb',
                    is_uint16=False, epochs=4, do_fine=1, gpus=gpus)
    #train_rgb_uint8->from (uint8_nadir_7) train one epoch at 0.001 for whole datatset (uint8_rgb_all)
    weights = train(annfile_irgb,['all'], weights, group='rgb', is_uint16=False, epochs=1, do_fine=0, gpus=gpus)
    #train_mpan_uint8-> # from(uint8 rgb_all) train one epoch on all off ranges(uint8_mpan_off)

    weights = train(annfile_mpan,['off'], weights, group='mpan', is_uint16=False, epochs=1, do_fine=0, gpus=gpus)
    #train_mpan_uint16-> from (uint8_mpan_off) train uint16_mpan_off
    weights_mpan_off16 = train(annfile_mpan,['off'], weights, group='mpan', is_uint16=True, epochs=1, do_fine=0, gpus=gpus)
    #train_mpan_uint16-> from (uint16_mpan_off) train uint16_mpan_nadir, uint16_mpan_very
    weights_mpan_nadir16= train(annfile_mpan,['nadir'], weights_mpan_off16, group='mpan', is_uint16=True, epochs=1, do_fine=0, gpus=gpus)
    weights_mpan_very16= train(annfile_mpan,['very'], weights_mpan_off16, group='mpan', is_uint16=True, epochs=1, do_fine=0, gpus=gpus)
    #train_mpan_uint16-> from (uint16_mpan_very) fine tune angles in very cascaded .i.e previous angle weights loaded for next angle
    _ = train_cascade(annfile_mpan, weights_mpan_very16,angles=['very'], train_type='fine_tune_angle',group='mpan',is_uint16=True,epochs=1, gpus=gpus)
    # Further fine tune each angle for two more epochs at fine larning rate of 0.0001
    train_extra(annfile_mpan, 'mpan', MODEL_MPAN_MAIN)
    #train_irgb_uint16-> from (uint16_mpan_nadir) train (uint16_irgb_nadir),uint16_irgb_off
    #going to 4 chan now so exclude conv1 for the first time
    weights_irgb_nadir16 = train(annfile_irgb,['nadir'], weights_mpan_nadir16, group='rgb', exclude=['conv1'], is_uint16=True, epochs=1, do_fine=0, gpus=gpus)
    weights_irgb_off16 = train(annfile_irgb,['off'], weights_irgb_nadir16, group='rgb', is_uint16=True, epochs=1, do_fine=0, gpus=gpus)
    #train_irgb_uint16-> from uint16_irgb_nadir and uint16_irgb_off fine tune all anles in nadir and off
    _ = train_cascade(annfile_irgb, weights_irgb_nadir16, angles=['nadir'], train_type='fine_tune_angle', group='rgb',
                      is_uint16=True, epochs=1, gpus=gpus)
    # Further fine tune each angle for two more epochs at fine larning rate of 0.0001
    _ = train_cascade(annfile_irgb, weights_irgb_off16, angles=['off'], train_type='fine_tune_angle', group='rgb',
                      is_uint16=True, epochs=1, gpus=gpus)
    # Further fine tune each angle for two more epochs at fine larning rate of 0.0001
    train_extra(annfile_irgb, 'rgb', MODEL_IRGB_MAIN)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train')
    #put img patches data in wdata as /wdata/irgbdata and /wdata/mpandata and models in ../spacenet_models
    parser.add_argument('--bands', required=False, default='all',
                        help='whcih bands to use rgb or mpan(764 urban)'
                             'one of [rgb mapn]')
    parser.add_argument('--angles', required=False, default=['nadir'], type=str, nargs='+',
                        help='Specify which nadir range to train one of [all ,nadir ,off, very]'
                             'or list of specific angle')
    parser.add_argument('--train_type',required=False, default='fine_tune_angle', type=str,help='cascaded fine_tune_angle or just normal train')
    parser.add_argument('--weights', required=True, help='Start training from this weight')
    parser.add_argument('--is_uint16',action='store_true',default=False, help='unit8 or uint16')
    parser.add_argument('--gpus',required=False, default=1,type=int, help='num gpus to use')

    args = parser.parse_args()
    angles = args.angles
    bands = args.bands
    initial_weight = args.weights
    gpus = args.gpus
    train_type = args.train_type
    is_uint16=args.is_uint16
    assert (train_type in ['fine_tune_angle','train_normal']) #all means either rgb or mpan everythin is both
  
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.isdir(MODEL_IRGB_MAIN):
        os.mkdir(MODEL_IRGB_MAIN)
    if not os.path.isdir(MODEL_MPAN_MAIN):
        os.mkdir(MODEL_MPAN_MAIN)
    if not os.path.isdir(MODEL_MPAN_U8):
        os.mkdir(MODEL_MPAN_U8)
    if not os.path.isdir(MODEL_RGB_U8):
        os.mkdir(MODEL_RGB_U8)

    if bands=='rgb':
       annfile_irgb = ''.join([TRAIN_DATA_DIR_IRGB, '/', annotation_prefix, '_annotation.json'])
       if train_type=='fine_tune_angle':
             _ = train_cascade(annfile_irgb,initial_weight,angles=angles, train_type='fine_tune_angle',group=bands,is_uint16=is_uint16,epochs=1,                                   gpus=gpus) 
       else:
             train(annfile_irgb,angles,initial_weight,group=bands,is_uint16=is_uint16,epochs=1,do_fine=0,init_epoch=0,gpus=gpus) 
    elif bands=='mpan':
       annfile_mpan = ''.join([TRAIN_DATA_DIR_MPAN, '/', annotation_prefix, '_annotation.json']) 
       if train_type=='fine_tune_angles':
             _ = train_cascade(annfile_mpan,initial_weight,angles=angles, train_type='fine_tune_angle',group=bands, is_uint16=is_uint16,epochs=1,                                   gpus=gpus) 
       else: 
             train(annfile_mpan,angles,initial_weight,group=bands, is_uint16=is_uint16,epochs=1,do_fine=0,init_epoch=0,gpus=gpus) 
    else:
        train_all(initial_weight,gpus=gpus)
        print ('All Done..')
