# MaksimovKA SpaceNet 4 Off-Nadir Buildings Solution Description

## Overview
Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.

##	Introduction
_Tell us a bit about yourself, and why you have decided to participate in the contest._
-	MaksimovKA nickname on topcoder
-	3rd place
-	I worked a lot with sattelite data as Computer Vision engineer and decided to try my skill on public statelite images challanges.

##	Solution Development
_How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?_
-	The task was to make binary instance segmentation (building footprints localisation). I  treated this task as semantic segmenation task and used some well known tricks in postprocessing t convert the results to instance segmentation result.
-	The main idea of the approach is to use Unet like architecture and FPN architecture for semantic segmentation but with changes that was used by winning solution in Data Science bowl 2018 – I used two channels as target – one channel for bulding mask and one channel for building contours.
-	I tried different combination of input channels but finally used only RGB with my custom data normalization using mean and std per channel for training dataset.
-	I used Unet and FPN from this great repository – https://github.com/qubvel/segmentation_models. I tired all encoders that are available in the repo and choosed the best one using 5-fold validation split.
-	Also I ahve experimented with different types of data augmentation to prevent overfitting.

##	Final Approach
_Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:_
-	Split data in 5 folds
-	Create binary masks with 2 channels – building body and building contour (using opencv findCountours function)
-	Train for 200 epochs starting with high learning rate 1e-3 using ADAM optimizer with 0.5 decay after non changing loss after 3 epochs and also with earlystopping for 20 epochs. As loos I used 0.5*bce + 0.5*dice for each of output target channels.
-	Batch size was 64 using 4 nvidia GPUs of random crop of 320x320 pixels, predict was done on full image padded to 928 pixels (becauce network is fully convolutional).
-	I trained Unet and FPN with inceptionresnetv2 encoder in both Unet and FPN on 5 folds – and I got 10 models (5 folds Unet and 5 folds FPN) that I used for my ensemble – I just averaged output probabilities for output target.
-	Used postprocessing to convert result to instances, see code for details.

## Open Source Resources, Frameworks and Libraries
_Please specify the name of the open source resource along with a URL to where it’s housed and it’s license type:_
-	segmentation-models (https://github.com/qubvel/segmentation_models/), MIT
-	tqdm ( https://pypi.python.org/pypi/tqdm), MPLv2, MIT
-	numpy ( https://pypi.python.org/pypi/numpy), BSD
-	opencv-python ( https://pypi.python.org/pypi/opencv-python), MIT
-	matplotlib ( https://pypi.python.org/pypi/matplotlib), BSD
-	scipy ( https://pypi.python.org/pypi/scipy), BSD
-	scikit-image ( https://pypi.python.org/pypi/scikit-image), Modified BSD
-	scikit-learn ( https://pypi.python.org/pypi/scikit-learn), BSD
-	GDAL ( https://anaconda.org/conda-forge/gdal), MIT
-	Pandas ( https://pypi.python.org/pypi/pandas), BSD
-	keras (https://github.com/keras-team/keras), MIT
-	tensorflow ( www.tensorflow.org), Apache
-	Shapely (https://github.com/Toblerity/Shapely)
-	albumentations (https://github.com/albu/albumentations), MIT

## Potential Algorithm Improvements
_Please specify any potential improvements that can be made to the algorithm:_
-	Use deeper encoder like SeNet154 for example.
-	Use advance postprocessing methods.


## Algorithm Limitations
_Please specify any potential limitations with the algorithm:_
-	It should not generalize to new kinds of data (big difference in weather conditions, zoom, etc); it is limitation for all machine learning algorithms.


## Deployment Guide
_Please provide the exact steps required to build and deploy the code:_
Steps are the same that was described here - https://docs.google.com/document/d/1-J2S6Dm87237Zy3NoXemlj6MVffald02n8mQ_SUVuqc/edit
