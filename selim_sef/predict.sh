#!/usr/bin/env bash
#DATA_PATH=/media/selim/sota/datasets/spacenet/SpaceNet-Off-Nadir_Test/SpaceNet-Off-Nadir_Test_Public/
DATA_PATH=$1

PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/sc50.json --output-dir predictions/masks/sc50 --resume weights/all_scse_unet_seresnext50_fold_0_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161 --resume weights/all_densenet_unet_densenet161_fold_0_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121 --resume weights/all_densenet_unet_densenet121_fold_0_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/r34.json --output-dir predictions/masks/r34 --resume weights/all_resnet_unet_resnet34_fold_0_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/r101.json --output-dir predictions/masks/r101 --resume weights/all_scse_unet_seresnext101_fold_0_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161_1 --resume weights/all_densenet_unet_densenet161_fold_1_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121_1 --resume weights/all_densenet_unet_densenet121_fold_1_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161_2 --resume weights/all_densenet_unet_densenet161_fold_2_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/sc50.json --output-dir predictions/masks/sc50_1 --resume weights/all_scse_unet_seresnext50_fold_1_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121_2 --resume weights/all_densenet_unet_densenet121_fold_2_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/r34.json --output-dir predictions/masks/r34_1 --resume weights/all_resnet_unet_resnet34_fold_1_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161_3 --resume weights/all_densenet_unet_densenet161_fold_3_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121_3 --resume weights/all_densenet_unet_densenet121_fold_3_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161_4 --resume weights/all_densenet_unet_densenet161_fold_4_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/sc50.json --output-dir predictions/masks/sc50_2 --resume weights/all_scse_unet_seresnext50_fold_2_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121_4 --resume weights/all_densenet_unet_densenet121_fold_4_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/r34.json --output-dir predictions/masks/r34_2 --resume weights/all_resnet_unet_resnet34_fold_2_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/sc50.json --output-dir predictions/masks/sc50_3 --resume weights/all_scse_unet_seresnext50_fold_3_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d121.json --output-dir predictions/masks/d121_5 --resume weights/all_densenet_unet_densenet121_fold_5_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/d161.json --output-dir predictions/masks/d161_5 --resume weights/all_densenet_unet_densenet161_fold_5_3_best --data-path $DATA_PATH
PYTHONPATH=. python3 inference/predict.py --gpu 0 --config configs/r34.json --output-dir predictions/masks/r34_3 --resume weights/all_resnet_unet_resnet34_fold_3_3_best --data-path $DATA_PATH
