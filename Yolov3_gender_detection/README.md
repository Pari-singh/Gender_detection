## Introduction

This model is built on Pytorch. Dockerfile contains all the required libraries for this code. To run this repository using Dockerfile, follow the steps:
1. git clone the repository and move to Yolov3_gender_detection
2. Build the image by typing `docker build --build-arg IMAGE_CUDA_VERSION=10.0 -t <image_name> .`
3. Create the image container with apex module by typing 
       `docker run -it --runtime=nvidia --ipc=host -p 8088:8088 -p 6006:6006 <image_name>`

## Pretrained model

Initialize with Imagenet weights, can work for any input size, only when training from 1st layer. When using transfer learning, need to resize the image size based on the pretrained model, in case of coco it is 416x416 for yolov3 or use coco weights and unfreeze the yolov3 (after the feature extractor) and train on any sized input. All the weights given are with 416x416, but can be used as baseline to finetune the model.

<< weights in server (/projects/dataset/trained\_weights/SuperMarket/pytorch\_weights) for pretrained yolov3, imagenet, person, gender_finetune, gender >>

## Training on Custom data


### Dataset format
Convert your data to coco format: Your data should mirror the directory structure created by data/get_coco_dataset.sh, with images and labels in separate folders, and one label file per image. <br />
Each image's label file must be locatable by simply replacing /images/\*.jpg with /labels/\*.txt in its pathname. An example image and label pair would be:

`../coco/images/val2014/COCO_val2014_000000013992.jpg  # image` <br />
`../coco/labels/val2014/COCO_val2014_000000013992.txt  # label`

* One file per image (if no objects in image, no label file is required).
* One row per object.
* Each row is class x_center y_center width height format.
* Box coordinates must be in normalized xywh format (from 0 - 1).
* Classes are zero-indexed (start from 0).

### Files creation

1. Create train, val and test\*.txt files: Create data/train.txt where each row contains a path to an image, the same image label must also exist in a corresponding /labels folder for each image that has targets.

2. Create new \*.names file listing all of the names for the classes in our dataset. For eg data/coco_gender.names is made for SuperMarket data

3. Create a .data file in data folder where lines 2 and 3 point to the new text file for training and validation. Line 1 contains the new class count. Line 4 points to the \*.names file. 

4. Update \*.cfg file (optional). Each YOLO layer has 255 outputs: n+5 outputs per anchor [4 box coordinates + 1 object confidence + n class confidences], times 3 anchors. To change it according to the class count, reduce this to [4 + 1 + n] * 3 = 15 + 3\*n outputs, where n is your class count. This modification should be made to the output filter preceding each of the 3 YOLO layers. Also modify classes=2 to classes=n in each YOLO layer, where n is your class count.

**Note: Its easier to convert any data format to first csv and then changing into any other format. Refer to [modresnet](https://gitlab-hq.wavesemi.com/WaveAI/man_vs_woman/tree/master/modResv1.5Yolov3) for converting SuperMarket type data to csv format. <br />
As an example I ran my entire data processing in [preprocessing notebook](https://gitlab-hq.wavesemi.com/WaveAI/man_vs_woman/blob/master/Yolov3_gender_detection/Processing_data.ipynb). 

### Train
1. Update hyperparameters such as LR, LR scheduler, optimizer, augmentation settings, multi_scale settings, etc in train.py for your particular task.

2. Run python3 train.py to train using custom data by passing the .data file as argument --data and .cfg file as argument --cfg and weights of pre-trained models. <br />
For eg. `export CUDA_VISIBLE_DEVICES=0,1; python train.py --data data/coco-gender.data --cfg cfg/yolov3_gender.cfg --weights weights/last.pt`.

## Transfer Learning

1. Download the official imagenet weights and coco weights for the model from here and here resp.

2. Update the config file as mentioned in the Training custom data section under files creation.

3. Run train.py by passing --transfer argument with weights of the model using --weights argument and the new dataset's config file using --config.


## Predict/ Test model (mAP)

Run the test.py file passing the .data file of the dataset created using --data argument, --weights and --cfg arguments as created for the dataset. <br />
For eg. python test.py --weights weights/last.pt --cfg cfg/yolo.v3_gender.cfg --data data/coco_gender.data

This gives us the accuracy of the model in terms of Precision, Recall, mAP and F1 score. Pass the file containing test images under eval variable in .data file.


## Inference

To run inference on images using pre-trained weights, follow the steps:
1. Create a folder inside data directory called samples and upload all the images on which inferencing is to be done inside this folder.

2. Next run detect.py using the pre-trained models by passing the arguments. <br />
For eg. `python detect.py --cfg cfg/yolov3_gender.cfg --data data/coco_gender.data --weights weights/backup150.pt`. Remember model weights trained will be stored in weights folder on their own.


## Debugging

* **train**:  Make sure to run the container with ip:host as that instantiates Nvidia's apex library for AMP. <br />
If the image container doesn't contain apex by default, clone the [repository](https://github.com/NVIDIA/apex) and place all the files in the repository in the working directory.

 - [--epochs]: Number of epochs
 - [--batch-size]: Number of batch size. Although effective batch size = batch-size\*accumulate
 - [--accumulate]: Number of batches to be accumulated before optimizing
 - [--cfg']: path to config file created
 - [--data']: path to the data file created for the data
 - [--img-size']: size to convert the maximum size of the image to, keeping the aspect ratio same.
 - [--multi-scale]: to train at faster speed. Only for square-image training (images are padded to letterbox sized).
 - [--rect']: if image size is rectangle, pass this along with multi-scale turned off (ie with single-scale).
 - [--resume']: To resume training
 - [--transfer']: to use transfer learning using coco weights (imagenet is set by default).
 - [--num-workers]: number of Pytorch DataLoader workers
 
 
* **test**: 
 - [--batch-size]: batch size for testing
 - [--cfg]: config file path
 - [--data]: data file path containing eval as test data
 - [--weights]: pretrained model weights
 - [--iou-thres]: threshold for iou score
 - [--img_size]: size to convert the maximum size of the image to, keeping the aspect ratio same.
 
* **detect**:
 - [--cfg]: config file path
 - [--data]: data file path
 - [--weights]: trained weights to infer on
 - [--images]: folder containing the images
 - [--img-size]: size to convert the maximum size of the image to, keeping the aspect ratio same.
 - [--output]: ouput path to store the result
 - [--half]: half precision, using FP16, useful while video object detection
 - [--webcam]: detection for realtime video object detection

