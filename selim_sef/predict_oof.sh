#!/usr/bin/env bash
DATA_PATH=$1

PYTHONPATH=. python inference/predict_oof.py --gpu 0 --folds_csv folds16.csv --fold 0 --config configs/sc50.json --output-dir oof/masks/sc50_0 --resume weights/all_scse_unet_seresnext50_fold_0_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 1 --folds_csv folds16.csv --fold 0 --config configs/d161.json --output-dir oof/masks/d161_0 --resume weights/all_densenet_unet_densenet161_fold_0_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 2 --folds_csv folds16.csv --fold 0 --config configs/d121.json --output-dir oof/masks/d121_0 --resume weights/all_densenet_unet_densenet121_fold_0_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 3 --folds_csv folds16.csv --fold 0 --config configs/r34.json --output-dir oof/masks/r34_0 --resume weights/all_resnet_unet_resnet34_fold_0_3_best --data-path $DATA_PATH &

wait

PYTHONPATH=. python inference/predict_oof.py --gpu 0 --folds_csv folds16.csv --fold 1 --config configs/sc50.json --output-dir oof/masks/sc50_1 --resume weights/all_scse_unet_seresnext50_fold_1_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 1 --folds_csv folds16.csv --fold 1 --config configs/d161.json --output-dir oof/masks/d161_1 --resume weights/all_densenet_unet_densenet161_fold_1_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 2 --folds_csv folds16.csv --fold 1 --config configs/d121.json --output-dir oof/masks/d121_1 --resume weights/all_densenet_unet_densenet121_fold_1_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 3 --folds_csv folds16.csv --fold 1 --config configs/r34.json --output-dir oof/masks/r34_1 --resume weights/all_resnet_unet_resnet34_fold_1_3_best --data-path $DATA_PATH &

wait
PYTHONPATH=. python inference/predict_oof.py --gpu 0 --folds_csv folds16.csv --fold 2 --config configs/sc50.json --output-dir oof/masks/sc50_2 --resume weights/all_scse_unet_seresnext50_fold_2_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 1 --folds_csv folds16.csv --fold 2 --config configs/d161.json --output-dir oof/masks/d161_2 --resume weights/all_densenet_unet_densenet161_fold_2_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 2 --folds_csv folds16.csv --fold 2 --config configs/d121.json --output-dir oof/masks/d121_2 --resume weights/all_densenet_unet_densenet121_fold_2_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 3 --folds_csv folds16.csv --fold 2 --config configs/r34.json --output-dir oof/masks/r34_2 --resume weights/all_resnet_unet_resnet34_fold_2_3_best --data-path $DATA_PATH &

wait
PYTHONPATH=. python inference/predict_oof.py --gpu 0 --folds_csv folds16.csv --fold 3 --config configs/sc50.json --output-dir oof/masks/sc50_3 --resume weights/all_scse_unet_seresnext50_fold_3_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 1 --folds_csv folds16.csv --fold 3 --config configs/d161.json --output-dir oof/masks/d161_3 --resume weights/all_densenet_unet_densenet161_fold_3_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 2 --folds_csv folds16.csv --fold 3 --config configs/d121.json --output-dir oof/masks/d121_3 --resume weights/all_densenet_unet_densenet121_fold_3_3_best --data-path $DATA_PATH &
PYTHONPATH=. python inference/predict_oof.py --gpu 3 --folds_csv folds16.csv --fold 3 --config configs/r34.json --output-dir oof/masks/r34_3 --resume weights/all_resnet_unet_resnet34_fold_3_3_best --data-path $DATA_PATH &



