## Introduction

This model is built on Tensorflow-gpu 2.0 beta version. Dockerfile contains all the required libraries for this code. To run this repository using Dockerfile, follow the steps:
1. git clone the repository and move to Yolov3_gender_detection
2. Build the image by typing `docker build --build-arg IMAGE_CUDA_VERSION=10.0 -t <image_name> .`
3. Create the image container by typing 
       `docker run -it --runtime=nvidia -p 8088:8088 -p 6006:6006 <image_name>`
4. Weights will stored in checkpoints folder and tensorboard logs will be saved in tensorboard directory.

<< trained weights in /project/dataset/trained\_weights/SuperMarket/mod\_res50v1.5yolov3, of various experimented training with square images, full dataset, several images etc. >>
       
## Dataset in TFRecord format

To run the model efficiently, TF2.0 presents Dataset API which loads the data in binary form at once. This needs the data to be converted to TFRecord format. To convert any dataset format to TFRecord, follow the steps:
1. Convert the dataset format into a csv file with columns `filename, xmin, ymin, xmax, ymax, class`, where `filename` is the image full name with extension and `class` is the string name of the class.
2. Split the data into train, val and test set into folders named train.csv, val.csv and test.csv respectively.
3. Convert each of the csv files into TFRecord using csv_tfrecord.py and passing the image directory as an argument. <br />
 eg. python csv_tfrecord.py --csv_input=../Wave_annotation_set3/train_label.csv --output_path=train.tfrecord --image_dir=../Wave_annotation_set3/train
TFRecord files will be created for each of the csv files separately in the mentioned output path.
4. Create a file called class.names containing class names in the order of index

## Training

1. Get the anchor sizes for our custom images by changing the filename to train.csv in kmeans.py file and run the python file. <br />
2. Replace the values of yolo_anchors variable inside yolov3_tf3/models.py file with the values obtained above by running kmeans.py.
3. Run train.py by passing TFRecord files of train and val set and export the visible CUDA devices in order to train on GPUs. <br />
 Eg. run `export CUDA_VISIBLE_DEVICES=0,1; python train.py --train_dataset train.tfrecord --val_dataset val.tfrecord --classes class.names`
   - [--train_dataset]: path to training dataset
   - [--val_dataset]: path to validation dataset
   - [--weights]: path to the weight (.tf format) file
   - [--classes]: path to class.names file
   - [--mode]: incase of using the pre-trained weights, we need the model architecture to be same. Thus this is useful incase of re-training with the models obtained earlier.
   - [--transfer]: incase of utilizing transfer learning, we need the model architecture to be same. Thus this is useful incase of re-training with the models obtained earlier.
   - [--height]: height of the image
   - [--width]: width of the image
   - [--epochs]: number of epochs to train
   - [--batch_size]: batch size per GPU
   - [--num_gpus]: number of available GPUs, passed with CUDA visible devices
   - [--class_num]: number of classes
   
## Detection

`python detect.py --weights ./checkpoints/yolov3-tiny.tf --image ./data/street.jpg`
  - [--classes]: path to classes file
  - [--image]: path to input image
  - [--output]: path to output image
  - [--weights]: path to weights file
  - [--num_classes]: number of classes in the model
  
