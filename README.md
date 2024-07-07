# Visual Recognition by Satellite Imagery

## Project Overview

This project focuses on the classification of land cover types using satellite imagery, leveraging recent advancements in deep learning. The primary objective is to evaluate two automated algorithms for land cover categorization using deep learning techniques, particularly Convolutional Neural Network (CNN) architecture for multi-class semantic segmentation.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
  - [Research Paper 1: U-Net](#research-paper-1-u-net)
  - [Research Paper 2: Improved CNN](#research-paper-2-improved-cnn)
- [Dataset](#dataset)
- [Analysis and Inference](#analysis-and-inference)
- [Contributors](#contributors)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction

Satellite picture categorization is a complex task involving remote sensing, computer vision, and machine learning. The unpredictability of satellite data and the absence of a single labeled high-resolution dataset make this task particularly challenging. Proper land cover forecasting is crucial for monitoring environmental changes and human settlement expansion. This study aims to improve visual recognition and evaluate the efficiency and accuracy of land cover categorization systems, contributing to ongoing research in deep learning for automated satellite image processing.

## Problem Statement

Recent advancements in deep learning have revolutionized the field of land cover classification using satellite imagery. This project evaluates two recent approaches for land cover classification using deep learning, focusing on CNN architecture. By assessing the accuracy and efficiency of these approaches, we aim to contribute to the ongoing research in automated analysis of satellite imagery.

## Methodology

### Research Paper 1: U-Net

**Paper**: "Land Cover Classification with U-Net: Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net" by Srimannarayana Baratam

**Model Architecture**:
- U-net is a fully-convolutional architecture used for semantic picture segmentation.
- Contractive route: Repeating two 3x3 convolutions followed by ReLU and 2x2 Max-pooling.
- Expansive route: Up-sampling followed by 2x2 convolution, concatenation with the contracting path, further up-sampling.
- Final layer: 1x1 convolution to reduce 64 components to the necessary 7 classes.

**Dataset**: Kaggle competition on Land Cover Classification (DeepGlobe 2018)
- Satellite imagery with pixel-wise land cover annotations.
- 21 land cover classes.

### Research Paper 2: Improved CNN

**Model Architecture**:
- Augmented CNN with handmade texture characteristics.
- Two convolutional layers, max-pooling, dropout, feature fusion, fully connected dense layers, Softmax layer with cross-entropy loss function.

**Dataset**: SAT-6 Dataset
- High-resolution aerial imagery with 21 land cover categories.

## Dataset

**UC Merced Land Use Dataset**:
- High-resolution aerial imagery.
- 21 distinct land cover classes.
- Images typically have a resolution of 256x256 pixels.
- Stored in RGB format.

## Analysis and Inference

- **U-Net**: Identified a larger scope of the satellite image with pixel-wise upsampled segmentation.
- **DeepSat V2**: Provided high probability classification for magnified images but struggled with coexistence of different geographies in one image.
- Both models have unique strengths and can be used simultaneously in image analysis tools for different tasks.

## Contributors

- **Pratham Singhal** (2021082)
- **Yash Yadav** (2021117)
- **Manshaa Kapoor** (2021540)

