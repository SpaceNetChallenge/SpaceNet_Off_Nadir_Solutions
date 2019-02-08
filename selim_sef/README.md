# selim_sef SpaceNet 4 Off-Nadir Buildings Solution Description


## Overview
Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.

## Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.
-	Handle: selim_sef
-	Placement you achieved in the MM:
-	About you: During the day I’m a computer vision engineer.  About two years ago I started participating in machine learning challenges. Quickly I became interested in Deep Learning for Computer Vision tasks. Which recently has become my main job.
-	Why you participated in the MM: 1. quite challenging task 2. open source dataset 3. to apply the skills obtained at previous competitions and to try different network architectures

## Solution Development
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?
-	I solved the task using the same approach as in the winning solution of Data Science Bowl 2018 https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741. Which basically has encoder decoder network like UNet and watershed postprocessing but instead of  predicting just binary masks the CNN predicted 3 masks: (body mask, separations between buildings, building contours). This helped to separate buildings much better than the simple watershed approach. Also additional step is used to remove false positive predictions. For that Grading Boosted Trees (LightGBM) is trained on masks morphological features produced from out of fold mask predictions.
-	At first I used heavy aumentations which inlcuded flips and rotations which turned out to be absolutely damaging for model performance.
-	To help models understand nadir angle and azimuth angle I made one hot encoding of catalogs and passed that as an additional channel just before decoder.
-	I used encoders pretrained on ImageNet and just initialized with He initialization additional input channels. Using pretrained encoders allows network to converge faster and produce better results even if it had less input channels originally.

## Final Approach
Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:
-	For semantic segmentation I used different variation of UNet architectures with Densenet161, Densenet121, Se-Resnext50 and Resnet34 encoders. For SE-Resnext networks I also used SCSE attention module in the decoder part.
-	I trained neworks using novel AdamW optimizer
-	As a loss function I used loss=focal+(1–soft dice). Using both focal (or bce) and soft dice in the loss is crucial to achieve good results in binary semantic segmentation and to get better results with ensembling.
-	OHE for catalogs – crucial to have better predictions on very off nadir angles
-	Augmentations: color jittering/ random sized crops, without flips
-	Preprocessing: I scaled image pixel values between 0 and 1
-	The final solution has an ensemble of models to produce binary masks. The masks produced by these models are averaged and then processed with watershed and GBT algorithm to remove false positives.
-	The best encoder was Densenet161. Even one fold produced results for 2nd place on public leaderboard. An ensemble of models had just marginal improvement (1 percent) compared to single fold of Densenet161.

4.	Open Source Resources, Frameworks and Libraries
Please specify the name of the open source resource along with a URL to where it’s housed and it’s license type:
-	Docker, https://www.docker.com (Apache License 2.0)
-	Nvidia-docker, https://github.com/NVIDIA/nvidia-docker, ( BSD 3-clause)
-	Python 3, https://www.python.org/, ( PSFL (Python Software Foundation License))
-	Numpy, http://www.numpy.org/, (BSD)
-	Tqdm, https://github.com/noamraph/tqdm, ( The MIT License)
-	Anaconda, https://www.continuum.io/Anaconda-Overview,( New BSD License)
-	OpenCV, https://opencv.org/ (BSD)
-	Pytorch https://pytorch.org/ (BSD)

5.	Potential Algorithm Improvements
Please specify any potential improvements that can be made to the algorithm:
-	As I found out issue with augmentations quite late I did not have time to tune all the parameters.
-	Watershed and GBT postprocessing thresholds should have been tuned for different angles
-	Most likely it was possible to use azimuth and nadir angle to “shift” predictions and get them closer to ground truth.
-	I did not use MUL images which may potentially improve results

6.	Algorithm Limitations
Please specify any potential limitations with the algorithm:
-	The current approach doesn’t work perfectly on very off nadir images


7.	Deployment Guide
Please provide the exact steps required to build and deploy the code:
1.	In this contest, a Dockerized version of the solution was required, which should run out of the box
