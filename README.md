# keras-kinetics-i3d for sign language
Keras implementation (including pretrained weights) of Inflated 3d Inception architecture reported in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d). The keras implementation was got from this [repository](https://github.com/dlpbc/keras-kinetics-i3d).

## Scripts description

This repository aims to use the Keras implementation of I3D to train a sign language recognition system. The **i3d_inception.py** file contains the implementation of the CNN architecture used to train the model.

**evaluate_sample.py** contains a script to test the model inference in a video sample.

**read_dataset.py** is the auxiliar code used to read the data from TFRecords using the TFData API. It can also be used to show some samples (frames) of data.

**augmentation.py** contains the math operations to transform the images during the training time.

**train.py** is the code used to build the archtecture and start the model training process.

## Data Preparation
The initial dataset used in the I3D training was the AUTSL. The videos was downloaded from this [link](https://chalearnlap.cvc.uab.cat/dataset/40/description/). After downloading the data, it's necessary  to convert the videos into TFRecord format. To do so, the script in this [repository](https://github.com/AlvaroCavalcante/video2tfrecord/tree/optimized-video2record) was used, in the branch **optimized-video2record**.

After transforming all the videos into TFRecords, we can start the model training. To do so, we used the following [colab notebook](https://colab.research.google.com/drive/157WZf6oUq36OPz327t028Z-JJzOuuM2s?usp=sharing).