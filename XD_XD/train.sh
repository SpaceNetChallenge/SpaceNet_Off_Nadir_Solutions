#!/bin/bash
# Usage: train.sh
source activate sp4 &&\
    python main.py check \
        -i /data/training \
    &&\
    python main.py preproctrain \
        -i /data/training \
        -w /wdata \
    &&\
    python -W ignore main.py train \
        -f 0 \
        -i /data/training \
        -w /wdata \
    &&\
    python -W ignore main.py train \
        -f 1 \
        -i /data/training \
        -w /wdata \
    &&\
    python -W ignore main.py train \
        -f 2 \
        -i /data/training \
        -w /wdata
