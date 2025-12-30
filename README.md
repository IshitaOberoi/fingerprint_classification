# Fingerprint Image Classification using Deep Learning

## Overview
This project implements an end-to-end deep learning pipeline for fingerprint image classification
using convolutional neural networks (CNNs). The goal is to explore the application of transfer learning,
model comparison, and robust evaluation techniques on a real-world image dataset.

The project focuses on building a clean, modular ML pipeline that includes custom dataset handling,
training multiple CNN architectures, evaluation using standard metrics, and saving trained models
for reuse.

> Disclaimer  
> This project is intended strictly for academic and machine learning experimentation.  
> It does not claim any biological or medical validity regarding blood group determination from fingerprints.
---

## Dataset
- Publicly available fingerprint image dataset sourced from Roboflow
- Images are organized into blood group classes (A, B, AB, O)
- Split into train, validation, and test sets
- Dataset exhibits class imbalance, which is handled during evaluation

## Problem Statement
Given a fingerprint image, the task is to classify it into one of the predefined blood group labels based on patterns present in the dataset images. This is treated as a supervised image classification problem using deep learning.

## Approach
1. Custom dataset creation using PyTorch `Dataset` to load images and labels from CSV files
2. Image preprocessing and normalization
3. Transfer learning using pretrained CNN architectures
4. Training and comparison of multiple deep learning models
5. Evaluation using precision, recall, F1-score, accuracy
6. Saving trained model weights for inference and reuse

## Models Used
- **ResNet18**
  - Deeper architecture
  - Achieved higher classification performance
- **MobileNetV2**
  - Lightweight and efficient model
  - Used for performance–efficiency comparison

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
  - Evaluation Metrics

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Results
- ResNet18 achieved the best overall performance on the test set
- MobileNetV2 provided competitive results with reduced model complexity
- The results demonstrate the effectiveness of transfer learning on limited datasets

## Future Improvements
- Increase dataset size for better generalization
- Apply advanced data augmentation techniques
- Experiment with additional CNN architectures
- Add confusion matrix visualizations
- Deploy the trained model using a web interface (Streamlit / FastAPI)
