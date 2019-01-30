__author__ Saket Kunwar
__handle__ number13

This is the docker version of my solution to the Spacenet off-nadir building detection challenge.
The code should be executed using nvidia-docker. A Mask-Rcnn model(Keras Matterport) with a modified Resnet-101 backbone is used. 
For each angle, there is a seperate single model as I found fine-tuning each angle gave the best result, so 27 models, one 
for each anngle plus 3 that are trained on whole ranges, ie nadir, off-nadir, very_nadir. The ziped model files, 
which  is 6.9 Gb is unzipped first time inference is run. For inference a single p2.xlarge will take approximately 1 hour 45 minutes 
on the given public test set. I found (InfraRed 1, Red, Green ,Blue) to be best suited for Nadir and Off-Nadir, while
a simple pan-sharpened images from (InfraRed 2, InfraRed 1, Red) to be better for Very-Nadir.
So during inference the right bands and model is loaded. There is no other ensembles.

The training part on this dataset starts from a pre-trained Mask-RCNN model, trained on
CrowdAI dataset. The CrowdAI dataset was in turn trained from MS-COCO weights from Matterport
which is also publicly available. My model configuration for both CrowdAI and Spacenet dataset are similar 
except for image dimension where in crowdai dataset it is 300x300 and on this dataset I created patches 
of 512x512 with  4 channel for (InfraRed, Red, Green, Blue) and 512x512 with 3 channel for (InfraRed2, InfraRed1, Red). 
The pacthes are created and stored in /wdata/irgbdata (203 Gb) and /wdata/mpandata (152 Gb).
The CrowdAI dataset is uint8 and RGB only and already comes in 300x300 dimension, and it was according to crowdai 
created from spacenet catalog. 

I started training on this dataset from a uint8 converted image patches. Later I found uint16
to yield better results and trained on it by loading the weights already trained on uint8. The final
model all use uint16 patches.

The training steps is summarized below:

At each steps model weight from the previous step is used.

1. Train on Nadir 7 only for 4 epoch at learning rate of 0.001 and 1 epcoh at learning rate of 0.0001 (uint8 , RGB, 3 channel)<- load crowdai pretrained weights
2. Train on All Dataset for 1 epoch (uint8, RGB, 3 channel) <- load weight from step 1
3. Train on OFF-NADIR angle ranges only ( uint8, (7,6 4) urban bands, 3 channel) denoted as mpan in code <- load weights from step 2
4. Train on OFF-NADIR angle range only (uint16, (7,6 4) urban bands, 3 channel) <- load weights from step 3
5. Train on NADIR angle range only (uint16, (7,6 4) urban bands, 3 channel) <- load weights from step 4
6. Train on VERY-NADIR angle range only (uint16, (7,6 4) urban bands, 3 channel) <- load weights from step 4
7. Fine-tune VERY-NADIR where each angle is trained seperately(only data from its folder) with model weight from previous angle loaded
     (i..e for nadir44 load weight trained on nadir42, for nadir46 load weight trained for 44)
8. Fine tune each VERY-NADIR angle at lr=0.0001 for two more epochs
9. Train NADIR angle ranges (uint16, IRGB, 4 channel)<- load weights from 4
10 Train OFF-NADIR angle ranges (uint16, IRGB, 4 channel)<- load weights from 9
11. Fine-tune NADIR where each angle is trained seperately(only data from its folder), similar to step 7
12. Fine tune each NADIR angle at lr=0.0001 for two more epochs, similar to step 8
13. Fine-tune OFF_NADIR where each angle is trained seperately(only data from its folder), similar to step 7
14. Fine tune each OFF_NADIR angle at lr=0.0001 for two more epochs, similar to step 8
 
This training procedure is the one followed in the given code for maximum reproducability. Although
ideally better results might be obtained by using uint16 throughout. All epoch are saved which takes (33 Gb) of space.
There is a slight overhead as tensorflow also saves the event files. 

Only some sparts of the above training were done using multi-gpu, p3.8xlarge (aws)  which uses 4 Nvidia v100 gpu (16mb gpu ram).
Compared to Nvidia Titan xp,  the training time should be similar. With my resnet-101 changes
batch size of 2 is used for spacenet dataset. So training on a  4-gpu system I estimate that total training time on the spacenet off-nadir dataset ,
should take 2 days. The code for training on the CrowdAI dataset is provided as (crowdai_train.py or crowdai_train.sh) seperately. 
As mentioned I used pretrained-weights trained on it  and not the dataset directly. So when (train.sh) is executed,  weights trained on 
the crowdai data is loaded, all spacenet models are deleted. Training on the crowdai dataset on a 4 multi-gpu  system, 
will also take roughly 2 days, totalling 4 days. The train.sh script has one line of code which when uncommented will also remove the crowdai pretrained 
weights as well, and re-train on the crowdai dataset starting from Matterports coco weights, producing the crowdai_final.h5 weights.
I can provide a docker with that line uncommented if the organizers want to reproduce the crowdai model weights also.

Most of the training and evaluation was done and monitored using coco style annotation and coco style metrics. The spacenet visualizer was used
to visualize and validate F1-score, used for this compitition. The ground truth in summaryData was first converted to coco style annotation. 
Only annotation for Atlanta_nadir7 was created as rest have the same annotation. During training and my local evaluation
annotation specific 'image_ids' are infered from this one annotation file rather than creating seperate annotation for each angle.

Summary of expected execution time and disk usage:

	Inference on public test set = 1 hour 45 minutes on p2.xlarge (aws)
	Train only on spacenet off-nadir dataset = 2 days (4 Gpu with 16Gb gpu ram minimum)
	Total patches data stored in /wdata = 360 Gb
	Final spacenet models for inference (stored in /wdata) = 7 Gb
	Crowdai dataset (stored in /wdata = 5 Gb
	Models weights saved during Training /wdata (all epochs) = 35 Gb




