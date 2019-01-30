#!/usr/bin/env bash

python /project/predict/predict_segmentation.py --test_folder $1 --submit_output_file $2 --gpu "0"