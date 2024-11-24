import sys
import os

# Add the parent folder of the scripts directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import torch
from torch.utils.data import random_split
from src.model import AttentionUNet
from src.dataset import RSDataset
from src.train import train_model

def main(args):
    # Load dataset
    dataset = RSDataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        max_samples=args.max_samples,
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize model
    model = AttentionUNet(img_ch=3, output_ch=1)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    # Train the model
    trained_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Save the final model
    torch.save(trained_model.state_dict(), args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention UNet")
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--labels_dir", required=True, help="Path to labels directory")
    parser.add_argument("--output_path", default="att_unet_trained.pth", help="Path to save the trained model")
    parser.add_argument("--model_path", default=None, help="Path to pre-existing trained model (if desired)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to load")

    args = parser.parse_args()
    main(args)
