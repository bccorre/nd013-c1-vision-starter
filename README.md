# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
```
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.


### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data/waymo/training_and_validation
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
This project goal is to train and evaluate an object detection pre-treined model. It consists of three different steps:
- data exploratory analysis: which consists of first insights on the data, and also some conclusion on how to split the training, validation and tests sets.
- data split script: which is a python script which will take the data and split into three groups (training, validation and testing) using the conclusions from previous step
- training and validation part: which shows many different experiments to try to improve the performance of the model in the testing set. The first experiment is used as a reference, and subsequent ones different approaches to try to improve the reference performance.

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.
For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.

### Dataset
#### Dataset analysis
All the data set analysis data is included in the jupyter notebook, with all details and conclusion on how to make the data splits.

Exploratory Data Analysis.ipynb

The data splits is created by runnning the create_splits.py script.

create_splits.py --data-dir "/home/workspace/data/waymo/training_data_set"

The script will create three folders in /home/data folder: train, test and val folders. That will then be used for training and validation.

The create_split script also uses a file of selected records with more diversity of objects, or with objects of all classes. This file is generated by the exploratory python code present in the jupyter notebook.

The splits of the records are done following the proportion: 75% for training, 15% for validation and the other 10% for test. The files selected in by the data analysis are also split following this proportion, and they are excluded from the list with whole files.

Once the create_split is executed, symbolic links are created on the train, eval, and test folders to the original files.

#### Cross validation
A spit of 75% training, 15% validation and 10% testing of files is used. This is selected to have the most proportion of training files, but also to avoid overfitting the data.

### Training
In order to train the model the following script must be run in the workspace folder, pointing to the model_dir used (reference or experiment_01 or 02) also pointing to the right configuration pipeline file.

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

Once the training is finished the following command must be run in order to evaluate the model:

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

#### Reference experiment
This experiment the model from the link below is used:

http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 

The training, eval, and test datasets are left as is. 

Also, the configuration file for training is the one generated base on ./pipeline.config and placed at ./experiments/reference.

After training and evaluation, the results are as follows:
```
![alt Reference results](experiments/reference/ref_results.png "tensor flow board with training and eval losses")
```
The results show that the model converges to a local minima, and reaches loss values of 8.9 in the evaluation set and 8.942 in the training set. 

One of the reasons is that the learning rate reaches zero after 2400 generations, not allowing the optimization get out of local minima. 

Another reason is that the training set need to be increased, this would help to improve the diversity of images which will change the local minimas, and makes the model reach the global minima and not overfit.

#### Improve on the reference
##### Experiment 01

Adding the following augmentations to the configuration file:

- random_rgb_to_gray: this could be useful to not rely on colored images in order to detect object, especially in cases where only a few color dominate the training set.
- random_adjust_brightness: to introduce more images by changing the brightness.
- random_square_crop_by_scale: to crop the existing images and try to zoom in to identify small objects, and also less objects per image, increasing accuracy of the model.
```
![alt Experiment 1 results](experiments/experiment_01/exp_01_res.png "tensor flow board with training and eval losses")
```
Given the results above, the local minima stagnation is more pronounced. The model seems to be in better minima and jump to a new one twice.

Although, it does converge to losses values better than the reference. For the training set it reaches 5.94, and to the evaluation set it reaches 5.94. This show that the split are good enough.

This means that a new optimizer with a new learning rate decay strategy would improve the convergency.

#####  Experiment 02

In this experiment the learning rate decay strategy is changed to exponential decay, in order to have very small increments in the end but allowing the model to explore (and not stay in one local minima) in the beginning.

Also the Adam optimizer is chosen in order to experiment and see if other methods can generalize more.

```
![alt Experiment 2 results](experiments/experiment_02/exp2_res_loss.png "tensor flow board with training and eval losses")

![alt Experiment 2 results Learning rate](experiments/experiment_02/exp2_res_learning_rate.png "tensor flow board with learning rate over time")
```

In this experiment the training and eval losses are aroun 0.9, less than experiment 1, giving better results than previous experiment.

The generated model is used to create a the testing video.
