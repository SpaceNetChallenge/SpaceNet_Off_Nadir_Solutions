#!/usr/bin/env bash
DATA_PATH=$1

python train.py --gpu 0 --fold 0 --config configs/r34.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/r34_0.log &
python train.py --gpu 1 --fold 1 --config configs/r34.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/r34_1.log &
python train.py --gpu 2 --fold 2 --config configs/r34.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/r34_2.log &
python train.py --gpu 3 --fold 3 --config configs/r34.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/r34_3.log &
wait

python train.py --gpu 0 --fold 0 --config configs/d121.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d121_0.log &
python train.py --gpu 1 --fold 1 --config configs/d121.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d121_1.log &
python train.py --gpu 2 --fold 2 --config configs/d121.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d121_2.log &
python train.py --gpu 3 --fold 3 --config configs/d121.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d121_3.log &
wait

python train.py --gpu 0 --fold 0 --config configs/d161.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d161_0.log &
python train.py --gpu 1 --fold 1 --config configs/d161.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d161_1.log &
python train.py --gpu 2 --fold 2 --config configs/d161.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d161_2.log &
python train.py --gpu 3 --fold 3 --config configs/d161.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d161_3.log &
wait

python train.py --gpu 0 --fold 0 --config configs/sc50.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/sc50_0.log &
python train.py --gpu 1 --fold 1 --config configs/sc50.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/sc50_1.log &
python train.py --gpu 2 --fold 2 --config configs/sc50.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/sc50_2.log &
python train.py --gpu 3 --fold 3 --config configs/sc50.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/sc50_3.log &
wait

python train.py --gpu 0 --fold 5 --config configs/d121.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d121_5_0.log &
python train.py --gpu 1 --fold 5 --config configs/d161.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/d161_5.log &
python train.py --gpu 2 --fold 0 --config configs/r101.json --csv folds16.csv --nadir all --test-epoch 2 --output-dir weights/ --data-path $DATA_PATH > logs/r101_0.log &
wait
