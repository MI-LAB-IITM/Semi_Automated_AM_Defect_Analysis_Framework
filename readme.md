# Semi-Automated Defect Analysis in Additive Manufacturing  

## Overview

This repository implements a two-stage semi-automated framework for quantifying defects in materials microstructures using an example dataset of additive manufacturing (AM). The pipeline includes a defect detection model trained using active learning, a microstructure-aware defect classification model, and, finally, mapping identified defects to AM processing parameters.

The framework is validated on AM systems and is designed to:

- Reduce manual annotation effort  
- Improve training efficiency through informed coreset selection  
- Enable interactive model predicted label refinement with human-in-the-loop  
- Classify defects, quantify them and map them to associated processing parameters  

This repository accompanies the manuscript:

> Efficient Semi-Automated Material Microstructure Analysis Using Deep Learning: A Case Study in Additive Manufacturing

---

# Scientific Contribution

This work introduces:

### 1. SMILE — Sampling via Maximin Latin-hypercube sampling from Embedding
A novel core-set selection algorithm for active learning that uses:

- t-SNE 2D feature embedding, K-means clustering for structured partitioning and Latin Hypercube Sampling (maximin criterion)  

SMILE promotes diversity and feature-space coverage. It is designed for data scarce, expert annotated materials imaging problems, where efficient sample selection is critical for reducing labeling effort and improving model generalization

### 2. Active Learning Framework for Microstructure Segmentation
An iterative annotation–training workflow integrating:

- U-Net semantic segmentation (Check out the Defect detection folder)  
- Core-set (SMILE) selection (Check out the Coreset (SMILE) folder for the algorithm)
- Interactive expert correction using CVAT (Check out the Deployment (Docker, CVAT, Nuclio) folder for implementation)

### 3. Microstructure-Informed Defect Classification

A dedicated classification stage that:

- Extracts defect-centered patches from segmented regions and incorporates etched microstructural context
- Uses a CNN with ImageNet-pretrained initialization ( Check out Defect classification folder)
- Applies focal loss to address class imbalance

Defects are classified as: Porosity and Lack of fusion

This enables microstructure aware differentiation of defect types beyond purely morphological features.  

### 4. End-to-End Defect Quantification
The pipeline connects:

Segmentation → Classification → Process–Defect map

# Repository Structure

```
.
├── Coreset Algorithm (SMILE)/            # Active learning coreset selection algorithm
│   └── SMILE.py    
│
├── Defect Detection/                     # Unet based semantic segmentation
│   └── Segmentation_training.ipynb       # Notebook for training & evaluation
│   └── instance_seg_utils.py             # Segmentation helper functions
│   └── seg_model.pth                     # Final trained segmentation model
│   └── SMILE_dataset.pkl                 # SMILE-coreset selected images
│
├── Defect Classification/                # CNN-based defect type prediction
│   └── Classification_training.ipynb     # Notebook for training & evaluation
│   └── Classification_test_dataset.pkl
│   └── class_model.pth                   # Final trained classification model
│   └── Classification_training_dataset.pkl 
│
├── Deployment (Docker, CVAT, Nuclio)/
│   └── Dockerfile                         # Container build
│   └── docker-compose.yml                 # Local CVAT setup
│   └── readme.md                          # check out to reproduce CVAT + Nuclio inside Docker
│   Nuclio/
│   └── function.yaml                      # Nuclio function configuration
│   └── model_handler.py                   # Model loading and inference logic
│   └── main.py    
│
├── requirements.txt                       # Environment Requirements
└── README.md
```


