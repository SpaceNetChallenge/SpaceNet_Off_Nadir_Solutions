#!/bin/bash
echo "training...."
echo "removing old models"
rm -rf /wdata/spacenet_models/rgb8_models
rm -rf /wdata/spacenet_models/mpan8_models
rm -rf /wdata/spacenet_models/irgb16_models
rm -rf /wdata/spacenet_models/mpan16_models


if [ -d /wdata/mpandata ] & [ -d /wdata/irgbdata ]; 
then  
   echo "Patches data already present.."
else
   echo "Creating patches  data first which takes  360 Gb "
   python3 create_patches_all.py --raw_dir $1 --summary_dir "$1/summaryData/" --step_size 512   
   echo "done.generating patches"
fi

#./crowdai_train.sh  #UNCOMMENT if extrnal data training again is required. 
echo 'train spacenet off-nadir from crowdai pretrained weights'
if [ -f /wdata/spacenet_models/external_pretrained/crowdai_final.h5 ];
then
  python3 train.py --weights "/wdata/spacenet_models/external_pretrained/crowdai_final.h5" --gpus 4 #set the num gpus
else
  echo 'Connot find crowdai_final.h5 pretrained model weights ..extract from archived model or train on crowdai data'
fi

