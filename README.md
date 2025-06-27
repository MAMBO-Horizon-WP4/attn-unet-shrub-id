# Attention UNet for Semantic Segmentation

This repository provides a PyTorch implementation of the Attention UNet architecture, designed for semantic segmentation tasks.

* [workshops](https://github.com/MAMBO-Horizon-WP4/workshops/) repository contains walkthrough notebooks showing re-use of the code to create a shrub identification model.
* [shrub-prepro](https://github.com/MAMBO-Horizon-WP4/shrub-prepro) repository contains data preparation code for creating training data for this model, also shown in the notebooks. 

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/attention-unet.git
   cd attention-unet
   ```

2. Create a virtual environment:

  ```
    python -m venv .venv
  ```

3. Load the environment, install package + dependencies

  ```
    source .venv/bin/activate
    pip install -e .
```

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
├── scripts
│   ├── run_inference.py
│   └── run_training.py
├── src
│   └── shrubnet
│       ├── dataset.py
│       ├── inference.py
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
└── tests
    ├── conftest.py
    ├── data
    │   ├── test_input.tif
    │   └── test_mask.tif
    ├── test_inference.py
    ├── test_model.py
    └── test_util.py
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

## Contributors

* [Rafael Barbedo](https://github.com/barbedorafael)
* [Jo Walsh](https://github.com/metazool)