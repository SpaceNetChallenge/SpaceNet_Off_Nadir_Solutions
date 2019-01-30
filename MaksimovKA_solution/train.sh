#!/usr/bin/env bash
rm -rf /wdata/*
mkdir /wdata/models_weights
mkdir /wdata/models_logs
mkdir /wdata/backbones_weights

cp -r /project/default_backbones_weights/* /wdata/backbones_weights

python /project/prepare_data/convert_data.py --training_data $1

# below line commented because I code for folds give different results from launch to lauch
# and the folds split was generated only once, when I started the research
# so instead of this I copy the first one, but you can algorithm in create_folds.py 
# python /project/prepare_data/create_folds.py
cp /project/default_folds_split/folds_split.csv /wdata

python /project/train/train_segmentation.py \
--gpu "0,1,2,3" \
--multi_gpu \
--fold "0,1,2,3,4" \
--num_workers 20 \
--network inceptionresnet_fpn_borders \
--loss_function double_head \
--optimizer adam \
--learning_rate 0.001 \
--decay 0.0001 \
--batch_size 64 \
--crop_size 320 \
--epochs 200 \
--augment 1 \
--add_contours 1 \
--preprocessing_function 1 \
--freeze_encoder 0

python /project/train/train_segmentation.py \
--gpu "0,1,2,3" \
--multi_gpu \
--fold "0,1,2,3,4" \
--num_workers 20 \
--network inceptionresnet_unet_borders \
--loss_function double_head \
--optimizer adam \
--learning_rate 0.001 \
--decay 0.0001 \
--batch_size 64 \
--crop_size 320 \
--epochs 200 \
--augment 1 \
--add_contours 1 \
--preprocessing_function 1 \
--freeze_encoder 0


