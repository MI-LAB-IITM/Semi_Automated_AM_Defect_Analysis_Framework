# Semi-Automated Defect Analysis in Additive Manufacturing  

## Overview

This repository implements a two-stage semi-automated framework for quantitative defect analysis in additively manufactured (AM) materials. The pipeline integrates defect detection using active learning, defect classification, and mapping defects to processing parameters.

The framework is validated on AM systems and is designed to:

- Reduce manual annotation effort  
- Improve training efficiency through informed coreset selection  
- Enable interactive model refinement with human-in-the-lop  
- Classify and Quantitatively map defects to processing parameters  

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

- U-Net semantic segmentation  
- Uncertainty-based sampling  
- Core-set selection  
- Interactive expert correction using CVAT

### 3. Microstructure-Informed Defect Classification

A dedicated classification stage that:

- Extracts defect-centered patches from segmented regions
- Incorporates etched microstructural context
- Uses a CNN with ImageNet-pretrained initialization
- Applies focal loss to address class imbalance

Defects are classified as: Porosity and Lack of fusion

This enables microstructure aware differentiation of defect types beyond purely morphological cues.  

### 4. End-to-End Defect Quantification
The pipeline connects:

Segmentation → Classification → Process–Defect map

# Repository Structure

```
.
├── Coreset Algorithm (SMILE)/ # Active learning coreset selection algorithm
│   └── SMILE.py    
│
├── Defect Detection/          # Unet based semantic segmentation
│   └── Segmentation_training.ipynb
│   └── instance_seg_utils.py
│   └── seg_model.pth
│   └── SMILE_dataset.pkl       
│
├── Defect Classification/     # CNN-based defect type prediction
│   └── Classification_training.ipynb
│   └── Classification_test_dataset.pkl 
│   └── class_model.pth
│   └── Classification_training_dataset.pkl 
│
├── Deployment (Docker, CVAT, Nuclio)/
│   ├── Dockerfile               # Container build
│   └── docker-compose.yml       # Local CVAT setup
│   Nuclio/
│   ├── function.yaml        # Nuclio function configuration
│   ├── model_handler.py     # Model loading and inference logic
│   ├── main.py    
│   └── requirements.txt     # Python dependencies for Nuclio

├── requirements.txt
└── README.md
```


