#!/bin/bash
echo "testing...."
if [ -d /wdata/spacenet_models/irgb16_models ] & [ -d /wdata/spacenet_models/mpan16_models ]; 
then
    python3 inference.py --indir $1 --output $2
else
    if  [ -f ../spacenet_models.zip ];  
    then
      echo "extracting models.."
      unzip ../spacenet_models.zip -d /wdata/
      python3 inference.py --indir $1 --output $2
    else
       echo "No model dir ...train again?"
    fi
fi


