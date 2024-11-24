# Attention UNet for Semantic Segmentation

This repository provides a PyTorch implementation of the Attention UNet architecture, designed for semantic segmentation tasks. It supports flexible training, inference on large images, and fine-tuning with additional datasets.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Fine-Tuning](#fine-tuning)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [References](#references)
- [Future Improvements](#future-improvements)

---

## Features
- **Attention UNet architecture**: Implements the Attention UNet model to focus on important regions of an image.
- **Tile-based inference**: Enables predictions on large images using a sliding window approach.
- **Incremental training**: Fine-tune the model with new datasets without starting from scratch.
- **Customizable pipeline**: Easily adapt dataset preparation, model parameters, and metrics.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/attention-unet.git
   cd attention-unet
   ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Verify PyTorch is installed and GPU is accessible (if available):

## Usage

### Training
To train the Attention UNet model, use the `train_model.py` script:

```bash
python scripts/run_training.py \
    --images_dir data/train/images \
    --labels_dir data/train/labels \
    --output_path model/trained_model.pth \
    --model_path model/pre_trained_model.pth \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.0001
```

**Arguments**:
- `--images_dir`: Path to the directory containing input images.
- `--labels_dir`: Path to the directory containing ground truth masks.
- `--output_path`: Path to save the trained model.
- `--model_path`: Path to pre-existing trained model, if desired (default: None).
- `--epochs`: Number of training epochs (default: 50).
- `--batch_size`: Batch size for training (default: 16).
- `--learning_rate`: Learning rate for the optimizer (default: 0.0001).

---

### Inference
To perform inference on a large image, use the `run_inference.py` script:

```bash
python scripts/run_inference.py \
    --model_path model/best_model.pth \
    --input_image data/test/test_area.tif \
    --output_image data/test/test_pred.tif \
    --window_size 512 \
    --step_size 256 \
    --threshold 0.5
```

**Arguments**:
- `--model_path`: Path to the trained model file.
- `--input_image`: Path to the input image for prediction.
- `--output_image`: Path to save the output segmentation map.
- `--window_size`: Sliding window size for tile-based inference (default: 512).
- `--step_size`: Step size for sliding the window (default: 256).
- `--threshold`: Threshold for binary segmentation (default: 0.5).

---

## Repository Structure

The project is structured as follows:

```
attention-unet/
├── src/
│   ├── dataset.py         # Dataset handling
│   ├── model.py           # Attention UNet model
│   ├── train.py           # Training logic
│   ├── inference.py       # Inference logic
│   ├── utils.py           # Helper functions (e.g., metrics)
├── scripts/
│   ├── run_training.py    # Training script
│   ├── run_inference.py   # Inference script
├── data/                  # Folder for storing datasets (not included)
├── notebooks/             # Jupyter notebooks for experiments
├── tests/                 # Unit and integration tests
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
```

---

## Requirements

- Python 3.8 or later
- PyTorch 1.10 or later
- Additional libraries:
  - torchvision
  - numpy
  - scikit-learn
  - GDAL
  - tqdm

Install dependencies by running:

```bash
pip install -r requirements.txt
```

---

## References

- Attention UNet Paper: Oktay et al., 2018 (https://arxiv.org/abs/1804.03999)
- GDAL Documentation: GDAL (https://gdal.org/)
- PyTorch: PyTorch Official Website (https://pytorch.org/)

---

## Future Improvements

- Add multi-class segmentation support.
- Integrate pre-trained models for faster convergence.
- Visualize training and inference results with TensorBoard or Matplotlib.