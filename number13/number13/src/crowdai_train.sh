if [ -d /wdata/crowdai_data ];
then
        echo "external data already unziped"
else
        echo "extracting external crowdai dataset" ; unzip -qq ../crowdai_data.zip -d /wdata/
fi
echo 'Removing pretrained crowdai weights as well'
rm /wdata/spacenet_models/external_pretrained/crowdai_final.h5
echo 'crowdai train'
#get the matterport mask-rcnn model weights trained on coco from matterport 
if [ -f /wdata/crowdai_data/mask_rcnn_coco.h5 ];
then
   echo 'Matterports Mask-rcnn model weights trained on coco present'
else
   wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -P /wdata/crowdai_data/
fi
python3 crowdai_train.py --weights /wdata/crowdai_data/mask_rcnn_coco.h5 --home_dir /wdata/crowdai_data --model_output /wdata/spacenet_models/external_pretrained/ --gpus 4 

