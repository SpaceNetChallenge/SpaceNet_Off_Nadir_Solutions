import json
from cocoeval import COCOeval
#from pycocotools import cocoeval as COCOeval
from config import NADIR, OFF_NADIR, VERY_NADIR, SPACENET_COLUMNS
from config import MODEL_MPAN, MODEL_RGB
from models import modellib, SpacenetConfigRGB, SpacenetConfigMPAN
from train_multi import get_datasets
from prediction import get_predictions_coco,get_predictions_spacenet, get_predictions_spacenet_withaug
import argparse
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd

try:
   unicode
except:
   unicode=str

annotation_prefix='Atlanta_nadir7_catid_1030010003D22F00'


#required since the ground truth used are for nadir7 only
def fix_gt(gt,angle):
    anns = []
    for i in gt.getAnnIds():
        val = gt.anns[i]
        val['image_id'] = val['image_id'].replace(annotation_prefix, angle)
        anns.append(val)
    #anns = np.asarray(anns)
    imgs = []
    for k in gt.imgs.keys():
        val = gt.imgs[k]
        val['file_name'] = val['file_name'].replace(annotation_prefix, angle)
        val['id'] = val['id'].replace(annotation_prefix, angle)
        imgs.append(val)
    #imgs = np.asarray(imgs)
    #gt.loadAnns([anns])
    #gt.loadImgs([imgs])
    #gt.dataset.update()
    #gt.createIndex()
    fin={}
    fin['info']=[]
    fin['categories']=[{'id':100,'name':'building','supercategory':'building'}]
    fin['images']=imgs
    fin['annotations']=anns
    json.dump(fin,open('tmp_gt.json','w'))
    gt= COCO('tmp_gt.json')
    return gt


#evaluate single angle coco style
def eval_spacenet(annFile, weight_path, angles, is_uint16=False, group='rgb', subname='tmp.json',conf_thres=0.88,is_spacenet=False):
    
    if group=='rgb':
        class InferenceConfig(SpacenetConfigRGB):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.0
            DETECTION_NMS_THRESHOLD =  0.3
            DETECTION_MAX_INSTANCES = 130

        MODEL_DIR = MODEL_RGB
    if group=='mpan':
        class InferenceConfig(SpacenetConfigMPAN):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.0
            DETECTION_NMS_THRESHOLD = 0.3
            DETECTION_MAX_INSTANCES = 130
        MODEL_DIR = MODEL_MPAN

    inference_config = InferenceConfig()
    
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=inference_config)
    model.load_weights(weight_path, by_name=True)
    print ('Loaded weights::',weight_path)
    _,dataset_val,ground_truth_annotations = get_datasets(annFile, angles, is_uint16=is_uint16, group = group)
    print ('Evaluating' ,angles[0])
    if not is_spacenet:
        ground_truth_annotations = fix_gt(ground_truth_annotations,angles[0])
        get_predictions_coco(dataset_val, model, subname=subname,conf_thres=conf_thres)
        ground_truth_annotations=COCO('tmp_gt.json')
        submission_file = json.loads(open(subname).read())
        results = ground_truth_annotations.loadRes(submission_file)
        cocoEval = COCOeval(ground_truth_annotations, results, 'segm')
        subimg=np.unique([item['image_id'] for item in submission_file])
        print ('testing num images::', len(subimg))
        cocoEval.params.imgIds = subimg
        cocoEval.evaluate()
        cocoEval.accumulate()
        print ('__________all_______')
        average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="all", maxDets=100)
        average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="all", maxDets=100)
        f1_all = 2 * (average_precision * average_recall) / (average_precision + average_recall)

        print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
        print ('__________large________')
        average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="large", maxDets=100)
        average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="large", maxDets=100)
        print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
        print ('___________medium_______')
        average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="medium", maxDets=100)
        average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="medium", maxDets=100)
        print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
        print ('___________small_______')
        average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="small", maxDets=100)
        average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="small", maxDets=100)
        print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
        print ('____________________________________________________________________________________________________')
        print('f1-score', f1_all)

    else:
        spacenet_columns = ['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence']
        shifts = {0: (0, 0)}
        final_anns = []
        for i in tqdm(dataset_val.image_ids):
            images = [dataset_val.load_image(i)]
            image_id = dataset_val.imgIds[i]
            anns = get_predictions_spacenet(image_id.replace('.jpg','').replace('.tif',''), images, model, shifts,conf_thres=conf_thres)
            final_anns.extend(anns)
        final_anns = [item[0:-1] for item in final_anns]
        df = pd.DataFrame(final_anns, columns=spacenet_columns)
        df.to_csv(subname, index=False)
        print('done.....')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='local validation ')
    parser.add_argument('--annfile', required=True, help='COCO style annotation file')
    parser.add_argument('--bands', required=False, default='ens',
                        help='whcih bands to use rgb or mpan(764 urban)'
                             'one of [rgb mapn]')
    parser.add_argument('--weights', required=True, help=' weight to load')
    parser.add_argument('--angles', required=True, type=str,
                        help='Specify which nadir angle to test..just one')
    parser.add_argument('--conf_thres', type=float, required=True)
    parser.add_argument('--subname',required=False, default='tmp.json')
    parser.add_argument('--is_spacenet',action='store_true',default=False)

    args = parser.parse_args()
    annfile = args.annfile
    angles = [args.angles]
    bands = args.bands
    weights = args.weights
    conf_thres = args.conf_thres
    subname = args.subname
    is_spacenet=args.is_spacenet
    eval_spacenet(annfile,weights, angles, is_uint16=True, group = bands,subname=subname,conf_thres=conf_thres,is_spacenet=is_spacenet)
