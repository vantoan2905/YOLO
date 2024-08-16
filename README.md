

# Car Detection Using YOLO

## Overview

This project focuses on real-time car detection in traffic images using the YOLOv8 model. The goal is to accurately detect and classify vehicles in various traffic conditions, aiding in traffic analysis and management.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Project Description

In this project, we utilized YOLOv8, a state-of-the-art deep learning model, for real-time car detection. The project involved:
- Data collection and annotation of traffic images.
- Preprocessing and cleaning the dataset.
- Fine-tuning YOLOv8 models to optimize for accuracy and speed.
- Evaluating the model performance on a variety of traffic scenarios.

## Features

- **Real-time detection**: Detects cars in real-time with high accuracy.
- **Scalable**: Can be extended to detect other types of vehicles or objects.
- **Optimized**: Fine-tuned models for balance between detection speed and accuracy.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vantoan2905/YOLO.git
   cd car-detection-yolo
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the YOLOv8 environment:

   ```bash
   pip install ultralytics
   ```

## Usage

1. **To detect cars in a video or image:**

   ```bash
   python detect.py --source <path_to_image_or_video> --weights <path_to_yolov8_weights>
   ```

2. **To train the model:**

   ```bash
   python train.py --data <path_to_dataset> --epochs <number_of_epochs> --weights <path_to_pretrained_weights>
   ```

## Dataset

The dataset consists of traffic images annotated with bounding boxes around cars. The dataset was cleaned and augmented to improve the model's performance in various traffic conditions.

- **Dataset Format**: The dataset is in YOLO format, with annotations in `.txt` files corresponding to each image.
- **Source**: Collected from various traffic cameras and publicly available datasets.

## Model Training

- **Model Used**: YOLOv8.
- **Training**: The model was fine-tuned on our dataset using transfer learning.
- **Parameters**: The model was trained for `XX` epochs with a learning rate of `XX`.

## Results

- **Accuracy**: Achieved an accuracy of `XX%` on the test set.
- **Sample Outputs**: [Link to sample outputs or screenshots]

## Future Work

- **Multi-class detection**: Extend the model to detect multiple types of vehicles.
- **Improved accuracy**: Further fine-tuning and dataset augmentation.
- **Deployment**: Implement the model in a real-world traffic monitoring system.

## Contributors

- **Nguyễn Toản** - [LinkedIn](https:www.linkedin.com/in/vantoan14090)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

