# Fingerprint Image Classification using Deep Learning

## Overview
This project implements an end-to-end deep learning pipeline for classifying fingerprint images into predefined blood group labels using convolutional neural networks (CNNs).
The primary objective of this project is to demonstrate practical application of computer vision and deep learning techniques such as transfer learning, model comparison, and performance evaluation.

> Disclaimer: This project is intended strictly for academic and machine learning experimentation.
It does not claim any biological or medical validity regarding blood group determination from fingerprints.

---

## Dataset
- Publicly available fingerprint image dataset sourced from Roboflow
- Images are organized into blood group classes (A, B, AB, O)
- Dataset is split into training and validation sets

## Problem Statement
Given a fingerprint image, the task is to classify it into one of the predefined blood group labels based on patterns present in the dataset images. This is treated as a supervised image classification problem using deep learning.

## Approach
1. Image preprocessing and normalization
2. Dataset loading using PyTorch `ImageFolder` and `DataLoader`
3. Transfer learning using pretrained CNN architectures
4. Training and fine-tuning of multiple deep learning models
5. Evaluation and comparison using standard classification metrics

## Models Used
- **ResNet18** – primary CNN model for classification
- **MobileNetV2** – lightweight and efficient architecture

All models are initialized with ImageNet pretrained weights and fine-tuned on the fingerprint dataset.

## Tech Stack
- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Libraries & Tools:**
  - torchvision
  - NumPy
  - OpenCV
  - Matplotlib
  - scikit-learn
- **Techniques:**
  - Convolutional Neural Networks (CNNs)
  - Transfer Learning
  - Image Preprocessing
  - Model Comparison

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results
The trained models were evaluated on unseen validation data.
Performance was analyzed using classification metrics and confusion matrices to compare the effectiveness of different CNN architectures.

## Future Improvements
- Increase dataset size for better generalization
- Apply advanced data augmentation techniques
- Experiment with additional CNN architectures
- Deploy the trained model using a simple web interface
