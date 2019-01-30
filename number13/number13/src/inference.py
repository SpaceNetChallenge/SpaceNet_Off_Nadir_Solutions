import argparse
import os
from tqdm import tqdm
import pandas as pd
import glob
import numpy as np
from prediction import predict_mpan512, predict_rgb512
from models import modellib_mpan, modellib_rgb
from models import SpacenetConfigIRGB_u16 as SpacenetConfigRGB
from models import SpacenetConfigMPAN_u16 as SpacenetConfigMPAN
from util import strip_tail
import logging
from config import NADIR, OFF_NADIR, VERY_NADIR, SPACENET_COLUMNS
from config import MODEL_MPAN_MAIN as MODEL_MPAN
from config import MODEL_IRGB_MAIN as  MODEL_RGB
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s %(message)s', filename='inference.log', level=logging.DEBUG)


def get_weights_angle_specific(use_epoch=2, group='rgb'):
    if group=='rgb':
        MODEL_DIR = MODEL_RGB
    if group=='mpan':
        MODEL_DIR = MODEL_MPAN
    weights = {}
    for item in glob.glob(''.join([MODEL_DIR, '*/'])):
        key = item.split('/')[-2]
        #strip time stamps
        key = '_'.join(key.split('_')[0:-1])
        ww = glob.glob(''.join([item, '/*.h5']))
        if len(ww) == 1:
            weights[key] = ww[0]
        else:
            for w in ww:
                if int(w.split('/')[-1].split('_')[-1].replace('.h5', '').replace('_','')) == use_epoch:
                    weights[key] = w
    return weights


def infer_nadir_angle(input_dir, group='rgb', infer_angles='all', nms_thres = 0.3):
    # infer angles takes all, nadir,off very for nadir range specific inference or list of anglse for some subset
    if group=='rgb':
        class InferenceConfig(SpacenetConfigRGB):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.0
            DETECTION_NMS_THRESHOLD =nms_thres
            DETECTION_MAX_INSTANCES = 130
        inference_config=InferenceConfig()
        model = modellib_rgb.MaskRCNN(mode="inference", model_dir=MODEL_RGB,
                                  config=inference_config)
    if group=='mpan':
        class InferenceConfig(SpacenetConfigMPAN):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.0
            DETECTION_NMS_THRESHOLD = nms_thres
            DETECTION_MAX_INSTANCES = 130
        inference_config = InferenceConfig()
        model = modellib_mpan.MaskRCNN(mode="inference", model_dir=MODEL_MPAN,
                              config=inference_config)

    #print (inference_config.display())

    weights = get_weights_angle_specific(use_epoch=4,group=group)
    for w in weights:
        logging.info('For {}: Use {}'.format(w,weights[w]))
    assert all([weights[k] != [] for k in weights])
    all_angles = glob.glob(''.join([input_dir,'/*']))
    all_angles = [item.split('/')[-1]  for item in all_angles]
    nadir = [item for item in all_angles if int(item.split('_')[1].replace('nadir','')) in range(7,26)]
    nadir_nums = [int(item.split('_')[1].replace('nadir','')) for item in nadir]
    off = [item for item in all_angles if int(item.split('_')[1].replace('nadir','')) in range(26,41)]
    off_nums = [int(item.split('_')[1].replace('nadir','')) for item in off]
    very = [item for item in all_angles if int(item.split('_')[1].replace('nadir','')) in range(41,54)]
    very_nums = [int(item.split('_')[1].replace('nadir','')) for item in very]

    print ('Num in nadir', len(nadir))
    print ('Num in off',len(off))
    print ('Num in very', len(very))

    annotations = []

    if infer_angles!='all':
        if infer_angles=='nadir':
            all_angles = nadir
        elif infer_angles=='off':
            all_angles = off
        elif infer_angles=='very':
            all_angles = very
        else:
            all_angles = [item for item in all_angles if item in infer_angles]

    for angle in tqdm(all_angles):
        subdir  = ''.join([input_dir,'/',angle,'/'])
        angle_num = int(angle.split('_')[1].replace('nadir',''))
        print ('Doing ',angle,angle_num)
        if angle.lower() in weights.keys():  # If  angle number and catid matches fine
            model.load_weights(weights[angle.lower()],by_name = True)
        else:
            #use the nadir angle number
            ww=[k for k in weights.keys() if k not in ['all','nadir','off','very'] if int(k.split('_')[1].replace('nadir',''))==angle_num]
            #incase of angles such as 10 and 53 there were multiple in public test set          
            #also maybe angle is 11 or 26 which are not present in public test set so use whole range specific weights
            if len(ww)==1:
               w=ww[0]
            else:
               #use the whole range specific weights:
               #print ('has multiple nadir nums and or angle num never in public train or test set') 
               if angle_num in nadir_nums:
                  w=weights['nadir']
               elif angle_num in off_nums:
                  w=weights['off']
               else:
                  w=weights['very']
               #print ('using range specific weights {} for {} '.format(w,angle_num))
               model.load_weights(w, by_name = True )

        if angle_num in nadir_nums:
            conf_thres = 0.90
        elif angle_num in off_nums:
            conf_thres = 0.87
        else:
            conf_thres = 0.84
        if group=='rgb':
            anns = predict_rgb512(subdir, model, conf_thres = conf_thres)
            annotations.extend(anns)
        if group=='mpan':
            anns = predict_mpan512(subdir,model, conf_thres = conf_thres)
            annotations.extend(anns)
        logging.info('{}: {} anns_num:{} with nms:{} conf_thres:{} '.format(group, angle, len(anns), nms_thres, conf_thres))

    return annotations


def ensure_model_available():
       wrgb = get_weights_angle_specific(use_epoch=4, group='rgb')
       angles= np.concatenate([NADIR,OFF_NADIR])
       for item in angles:
          try:
            found = int(wrgb[item.lower()].split('/')[-1].split('_')[-1].replace('.h5',''))==4
            print ('Found model weights for {}'.format(item))
          except: 
            print ('Expected  model weight not found for.. '.format(item))
       wmpan = get_weights_angle_specific(use_epoch=4, group='mpan')
       angles= VERY_NADIR
       for item in angles:
          try:
            found = int(wmpan[item.lower()].split('/')[-1].split('_')[-1].replace('.h5',''))==4
            print ('Found model weights for {}'.format(item))
          except:
            print ('Expected  model weight not found for '.format(item))


def infer_all(input_dir):
    ensure_model_available()	
    print ('Predicting nadir ...')
    anns_nadir = infer_nadir_angle(input_dir, group='rgb', infer_angles='nadir', nms_thres=0.3)
    print ('Predicting off nadir..')
    anns_off = infer_nadir_angle(input_dir, group='rgb', infer_angles='off', nms_thres=0.3)
    print ('Predicting very nadir...')
    anns_very = infer_nadir_angle(input_dir, group='mpan', infer_angles='very', nms_thres=0.3)
    anns=anns_nadir
    anns.extend(anns_off)
    anns.extend(anns_very)
    return anns


def save_results(result,save_as):
    df = pd.DataFrame(result,columns=SPACENET_COLUMNS)
    df.to_csv(save_as,index = False)
    print ('saved results..to.', save_as)


if __name__=='__main__':
    #test.sh <data-folder> <output_path>
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--indir', required=True, help='Dir to read images from')
    parser.add_argument('--output', required=True, help='file name to store results')
    parser.add_argument('--bands', required=False, default='ens',help='whcih bands to use rgb or mpan(764 urban) or ensemble both'
                                                                      'one of [rgb mapn ens]')
    parser.add_argument('--angles' ,required=False, default='all', help='Specify which nadir range to predict one of [all ,nadir ,off, very]'
                                                                          'or list of specific angle')
    args = parser.parse_args()
    input_dir = args.indir
    output = args.output
    angles = args.angles
    bands = args.bands
    assert bands in ['ens','rgb','mpan']
    #Insure models directory is there..'
    if not os.path.isdir(MODEL_RGB):
       logging.info('No RGB Model Dir {}',format(MODEL_RGB))
    if not os.path.isdir(MODEL_MPAN):
       logging.info('NO MPAN  model dir {}'.format(MODEL_MPAN))
    print ('Predicting from :',input_dir, 'Writing to:',output)
    if angles not in ['all','nadir', 'off','very']:
        if angles in NADIR or angles in OFF_NADIR or angles in VERY_NADIR:
            print ('Doing Single angle...')
            angles=[angles]
    print ('Doing..', angles)
    if bands=='rgb':
        print  ('Using..IRGB Pan sharpened bands')
        rgb_result = infer_nadir_angle(input_dir,group='rgb',infer_angles=angles,nms_thres=0.3)
        save_results(rgb_result,output)
    elif bands=='mpan':
        print ('Using ..Ir2 Ir1 Red Multi spectral pansharpened ..denoted as mpan')
        mpan_result = infer_nadir_angle(input_dir, group='mpan', infer_angles= angles,nms_thres=0.3)
        save_results(mpan_result, output)
    else:
        print ('Predicting IRGB for nadir and off angles denoted as rgb')
        print('Predicting Multi-spectral(7,6,5) for very nadir angle denoted as mpan')
        ens_result = infer_all(input_dir)
        save_results(ens_result, output)
    print ('done...')
