# Project Title: Fruit classification

## Project Description

A repository created for a DTU's Machine Learning Operations course project.

### Goal

The goal of the project is to create a well funcioning Machine Learning pipeline for classifying fruit types based on 2D images. Following proper ML operation principles and best practices like reproduceability, continous integration and utilizing cloud computing is a key objective of the project. The complete pipeline will also be implemented with monitoring for the maintainance purposes. Future updates of the project will mainly focus on scaling the application in order to secure the longevity of it.

### Data

The dataset is named Fruits-360 and can be downloaded from https://www.kaggle.com/datasets/moltean/fruits/data. It is a dataset with 90380 colourfull images of 131 fruits and vegetables with image size: 100x100 pixels. The data is grouped by folders with corresponding fruit name. Within the folder there is a number of images of the corresponding fruit. One of the tasks within the project is a properly process and strucute the data into images and lables and also group the dataset into train set and validation set.

### Framework

The project is built within a Pytorch ecosystem. Both the data preparation, a model, training and predictions are built on top of the Pytorch library. The models are taken from the [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) package.

### Model

ResNet is one of the most chosen model types for the image classification tasks due to its ability to address the challenges of training very deep neural networks. ResNet architectures, especially variants like ResNet50 and ResNet101, have demonstrated state-of-the-art performance on various benchmark datasets for image classification tasks. On top of that ResNet has shown good generalization capabilities across a wide range of image classification tasks, making them versatile for different applications.

For this project a ResNet model implementation from [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) was implemented in the pipline.