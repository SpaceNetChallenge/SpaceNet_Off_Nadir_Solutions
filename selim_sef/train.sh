#!/usr/bin/env bash
PYTHONPATH=. python3 tools/mask_from_geo.py --train-dir $1
PYTHONPATH=. python3 tools/mask_utils.py

sh train_all.sh $1

sh predict_oof.sh $1

python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_0 d161_0 r34_0 sc50_0
python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_1 d161_1 r34_1 sc50_1
python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_2 d161_2 r34_2 sc50_2
python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_3 d161_3 r34_3 sc50_3

python3 train_classifier.py