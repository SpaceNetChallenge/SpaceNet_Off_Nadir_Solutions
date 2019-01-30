#!/usr/bin/env bash

sh predict.sh $1

python3 ensemble.py

python3 predict_trees.py

python3 generate_polygons.py --output-path $2