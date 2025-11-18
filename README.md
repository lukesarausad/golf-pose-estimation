# Golf Pose Estimation with RTMPose

Real-time golf swing analysis using RTMPose for 2D pose estimation.

## Project Overview
This project implements RTMPose for golf swing analysis as part of CSE 455 final project at University of Washington.

### Team Members
- Luke Sarausad
- Bassil Rashid

## Features
- Real-time pose estimation on golf swing videos
- Swing phase classification
- Biomechanics analysis (angles, club speed)
- Visual pose overlay

## Project Structure
```
golf_pose_rtm/
├── data/              # Dataset directory (not tracked)
├── models/            # Model implementations
├── utils/             # Utility functions
├── train.py           # Training script
├── inference.py       # Inference script
└── config.py          # Configuration
```

## Setup

### Requirements
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

### Dataset
Download COCO 2017 dataset from [cocodataset.org](https://cocodataset.org)

## Usage

### Training
```bash
python train.py
```

### Inference
```bash
python inference.py --video path/to/golf_video.mp4
```

## Implementation Details
Based on RTMPose paper: "RTMPose: Real-Time Multi-Person Pose Estimation"

Key components:
- Backbone: ResNet50 (simplified from CSPNeXt)
- Head: SimCC coordinate representation
- Loss: KL Divergence

## Progress
- [x] Project setup
- [ ] Model implementation
- [ ] Training pipeline
- [ ] Golf-specific analysis
- [ ] Results and evaluation

## References
- RTMPose paper
- COCO Keypoint Detection dataset