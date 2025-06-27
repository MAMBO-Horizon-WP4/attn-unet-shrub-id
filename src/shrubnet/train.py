import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from shrubnet.utils import calculate_metrics, calculate_accuracy, compute_iou


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=50,
    batch_size=16,
    lr=0.0001,
    accumulation_steps=4,
    device="cpu",
    model_dir="model_states",
):
    """
    Train the Attention UNet model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        accumulation_steps (int): Number of steps for gradient accumulation.
        device (str): Device to use for training ("cpu" or "cuda").

    Returns:
        torch.nn.Module: The trained model.
    """
    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Move the model to the device
    model.to(device)

    # Initialize variables to track the best validation performance
    best_val_loss = float("inf")

    if not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = Path(model_dir) / "best_model.pth"

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        optimizer.zero_grad()

        for i, (images, labels) in enumerate(progress_bar):
            # Move data to the device
            images, labels = images.float().to(device), labels.float().to(device)

            # Ensure labels are within [0, 1]
            labels = torch.clamp(labels, 0, 1)

            # Forward pass
            outputs = torch.sigmoid(model(images))

            # Adjust labels shape to match outputs shape if needed
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)

            # Check if labels are in the correct range
            if labels.max() > 1 or labels.min() < 0:
                raise ValueError(
                    f"Labels out of bounds: min={labels.min()}, max={labels.max()}"
                )

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and gradient accumulation
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Update metrics
            epoch_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            progress_bar.set_postfix({"loss": loss.item(), "accuracy": acc})

        # Learning rate scheduler step
        scheduler.step()

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        val_loss, val_acc, val_precision, val_recall, val_f1, val_iou = validate_model(
            model, val_loader, criterion, device
        )

        print(
            f"Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}, "
            f"Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, "
            f"F1 = {val_f1:.4f}, IoU = {val_iou:.4f}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    return model


def validate_model(model, val_loader, criterion, device):
    """
    Perform validation on the model.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The validation DataLoader.
        criterion (torch.nn.Module): The loss function.
        device (str): Device to use for validation ("cpu" or "cuda").

    Returns:
        tuple: Validation loss, accuracy, precision, recall, F1 score, and IoU.
    """
    model.eval()
    val_loss, val_acc, val_precision, val_recall, val_f1, val_iou = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate metrics
            acc = calculate_accuracy(outputs, labels)
            precision, recall, f1 = calculate_metrics(outputs, labels)
            iou = compute_iou(outputs, labels)

            # Accumulate metrics
            val_acc += acc
            val_precision += precision
            val_recall += recall
            val_f1 += f1
            val_iou += iou

    # Compute average metrics
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_precision /= len(val_loader)
    val_recall /= len(val_loader)
    val_f1 /= len(val_loader)
    val_iou /= len(val_loader)

    return val_loss, val_acc, val_precision, val_recall, val_f1, val_iou
