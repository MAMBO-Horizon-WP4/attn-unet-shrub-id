import sys
import os

# Add the parent folder of the scripts directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import torch
from src.model import AttentionUNet
from src.inference import run_inference

def main(args):
    # Load the model
    model = AttentionUNet(img_ch=3, output_ch=1)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    print(f"Loaded model from {args.model_path}")

    # Run inference
    run_inference(
        model=model,
        input_image_path=args.input_image,
        output_image_path=args.output_image,
        window_size=(args.window_size, args.window_size),
        step_size=(args.step_size, args.step_size),
        threshold=args.threshold,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Attention UNet")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_image", required=True, help="Path to save the output prediction")
    parser.add_argument("--window_size", type=int, default=512, help="Size of the sliding window")
    parser.add_argument("--step_size", type=int, default=256, help="Step size for sliding window")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")

    args = parser.parse_args()
    main(args)
