import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(predictions, labels, threshold=0.5):
    """
    Calculate precision, recall, and F1 score for binary segmentation.

    Args:
        predictions (torch.Tensor): Predicted outputs from the model.
        labels (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to binarize predictions.

    Returns:
        tuple: Precision, Recall, and F1 score.
    """
    preds = (predictions > threshold).float().cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    f1 = f1_score(labels, preds, zero_division=1)

    return precision, recall, f1


def calculate_accuracy(predictions, labels, threshold=0.5):
    """
    Calculate accuracy for binary segmentation.

    Args:
        predictions (torch.Tensor): Predicted outputs from the model.
        labels (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Accuracy score.
    """
    preds = (predictions > threshold).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.numel()

    return accuracy.item()


def compute_iou(predictions, labels, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for binary segmentation.

    Args:
        predictions (torch.Tensor): Predicted outputs from the model.
        labels (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: IoU score.
    """
    preds = (predictions > threshold).float().cpu().numpy()
    labels = labels.cpu().numpy()

    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection

    iou = intersection / union if union != 0 else 0.0
    return iou


def preprocess_images(images, target_size=(256, 256)):
    """
    Resize and normalize images for input to the model.

    Args:
        images (list or np.ndarray): List or array of input images.
        target_size (tuple): Desired dimensions for resizing.

    Returns:
        np.ndarray: Preprocessed and normalized images.
    """
    processed_images = []
    for img in images:
        resized = cv2.resize(img, target_size)
        normalized = resized / 255.0
        processed_images.append(normalized)

    return np.array(processed_images)
