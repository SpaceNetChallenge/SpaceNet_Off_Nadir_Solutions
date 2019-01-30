#!/bin/bash
# Usage: test.sh <output_filename>
# i.e.)
#        $ bash test.sh out.txt
#
# ---------------------
source activate sp4 &&\
    CUDA_VISIBLE_DEVICES=0 python main.py filecheck \
        -i /data/test \
        -w /wdata \
    &&\
    python main.py preproctest \
        -i /data/test \
        -w /wdata \
    &&\
    CUDA_VISIBLE_DEVICES=0 python -W ignore main.py inference \
        -i /data/test \
        -w /wdata \
        -o $@
