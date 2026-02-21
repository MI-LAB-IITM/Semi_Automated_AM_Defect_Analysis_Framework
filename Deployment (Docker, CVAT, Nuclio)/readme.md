# Deployment Guide: CVAT + Nuclio Integration

This directory contains the configuration files required to deploy the trained
defect segmentation model as a Nuclio function and integrate it with CVAT for
interactive annotation correction.

This deployment setup is OPTIONAL. The experimental results reported in the
paper (segmentation performance, active learning evaluation, and defect
classification) can be reproduced without using CVAT or Nuclio. This folder is
provided for users who wish to replicate the complete human-in-the-loop
workflow described in the manuscript.

## Demo

A short demonstration of the model inference and CVAT-based
annotation correction workflow is available here:

[Video link]


---------------------------------------------------------------------

Overview

During the defect detection stage, an active learning loop was implemented:

1. A U-Net segmentation model was trained on a small labeled subset.
2. The trained model generated predictions for unlabeled images.
3. Predicted masks were reviewed and corrected by a human expert using CVAT.
4. Corrected masks were added to the training set.
5. The model was retrained on the expanded dataset.

To streamline this workflow, the trained model was deployed as a Nuclio
function so that CVAT could give directly the annotations.

---------------------------------------------------------------------

## Directory Structure

```
deployment/
│
├── nuclio/
│   ├── function.yaml        # Nuclio function configuration
│   ├── model_handler.py     # Model loading and inference logic
│   └── requirements.txt     # Python dependencies for Nuclio runtime
│
├── Dockerfile               # Container build
└── docker-compose.yml       # Local CVAT setup
└── readme.md
```

---------------------------------------------------------------------

Requirements

The following tools must be installed:

- Docker
- Nuclio CLI (nuctl)
- CVAT

CVAT can be installed following the official instructions:
https://github.com/opencv/cvat

---------------------------------------------------------------------

Step 1: Start CVAT

If using Docker-based installation:

    docker-compose up -d

After startup, CVAT will be available at:

    http://localhost:8080

---------------------------------------------------------------------

Step 2: Deploy the Nuclio Function

From the root directory of this repository, run: nuctl deploy \

This command builds the inference container, installs dependencies,
and registers the model as a callable function within CVAT.

---------------------------------------------------------------------

Step 3: Use the Model in CVAT

1. Open CVAT in your browser.
2. Create a new segmentation task.
3. Navigate to the "Models" section.
4. Select the deployed model.
5. Run automatic annotation.
6. Review and correct predicted masks.
7. Export corrected masks in PNG format.

---------------------------------------------------------------------

Model Weights

Ensure that pretrained model weights are available at the path specified
in model_handler.py.

---------------------------------------------------------------------

Misc Notes

- No modifications to the CVAT source code are required.
- The core machine learning experiments can be reproduced without
  deploying this module.
- This configuration is provided to enable full replication of the
  interactive annotation workflow described in the paper.

---------------------------------------------------------------------

Scope

This deployment setup supports:

- Binary semantic segmentation (defect vs background)
- Mask output compatible with CVAT

It does not modify CVAT internals or provide custom CVAT plugins.
