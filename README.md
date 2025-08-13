# Valohai-NVIDIA-TAO

This repository contains a complete training pipeline that integrates the **NVIDIA TAO Toolkit** with **Valohai**, using the **KITTI dataset** for object detection with the DetectNet\_v2 model.

## Overview

This project demonstrates how to:

* Preprocess KITTI dataset images and labels into TFRecord format
* Configure and train a DetectNet\_v2 model using NVIDIA TAO Toolkit
* Evaluate the trained model performance
* Run the entire workflow using a Valohai pipeline

## Project Structure

* `load_data.py`: Downloads and preprocesses KITTI data (images, labels, specs)
* `train.py`: Launches DetectNet\_v2 training with NVIDIA TAO
* `evaluate.py`: Evaluates the model using TAO Toolkit
* `valohai.yaml`: Defines the pipeline and steps for Valohai execution
* `requirements.txt`: Contains the Python packages required for preprocessing and training orchestration

## Important Note
Before running the project. Make sure you add your NGC_API_KEY in Valohai project registry
* image pattern: `nvcr.io/*`
* username: `$oauthtoken`
* password: `YOUR_NGC_API_KEY`

## Spec Files used in TAO configurations
* [TFRecords spec file](https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/tlt_specs/detectnet_v2_tfrecords_kitti_trainval.txt)
* [Training spec file](https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/tlt_specs/detectnet_v2_train_resnet18_kitti.txt)


## Pipeline Steps

The pipeline automates the full model development workflow:

### 1. Load & Preprocess Dataset

* Downloads KITTI object detection dataset (images and labels)
* Extracts, parses, and optionally subsets the dataset
* Converts data to **TFRecord** format compatible with TAO Toolkit
* Outputs zipped datasets for Valohai input versioning

Parameters:

* `subset`: Number of images to include
* `num_plot_images`: Visualizes a few samples during preprocessing

### 2. Train Model with TAO Toolkit

* Trains a **DetectNet\_v2** model using NVIDIA's TAO Toolkit Docker container
* Uses the TFRecords and spec files created in the previous step
* Saves the resulting `.hdf5` model file and logs
* Outputs training progress
<img width="945" height="386" alt="image" src="https://github.com/user-attachments/assets/1659657d-516d-4384-99db-79431f054a80" />



Configurable parameters:

* `epochs`
* `batch_size_per_gpu`
* `use_batch_norm`
* `val_split`

Check [Training spec file](https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/tlt_specs/detectnet_v2_train_resnet18_kitti.txt) for more configurable parameters.

    

Environment variables (defined in `valohai.yaml`) handle:

* GPU usage
* TAO Docker flags
* Output and data directories
* NGC API authentication

Training progress

### 3. Evaluate Model

* Evaluates trained `.hdf5` models using TFRecords and original validation data
* Generates metrics (e.g., precision, recall) and visual output snapshots
* Uses the same TAO container and config setup as training

## Run the Pipeline

```bash
vh pipeline run train_and_evaluate
```

This command will:

1. Load and preprocess the KITTI dataset
2. Train a DetectNet\_v2 model using TAO Toolkit
3. Evaluate the trained model

## Dataset

This project uses the **KITTI Object Detection** dataset:

* Images: [KITTI Images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
* Labels: [KITTI Labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

Specs must follow TAO Toolkit formatting.

## Model

The training pipeline uses **DetectNet\_v2**, an NVIDIA TAO object detection architecture optimized for real-time applications. Model and training parameters are defined in spec files, which are version-controlled and passed via Valohai inputs.

## Dependencies

Ensure your Valohai executions install required packages:

```bash
pip install -r requirements.txt
```

TAO Toolkit itself runs within NVIDIA’s prebuilt containers.

## License

This project uses:

* [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
* [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
* [Valohai MLOps platform](https://valohai.com/)

See each tool’s license for details.

## Acknowledgments

* **NVIDIA TAO Toolkit** for powerful model training
* **Valohai** for automating machine learning workflows
* **KITTI Dataset** by Karlsruhe Institute of Technology & Toyota Technological Institute at Chicago
