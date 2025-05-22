import os
from pathlib import Path
from shrubnet.inference import run_inference, sliding_window
from shrubnet.model import AttentionUNet
import rasterio


def test_inference(input_image_path, tmp_path):

    # Empty model, random weights
    model = AttentionUNet()
    output_image_path = tmp_path / "test.tif"
    run_inference(model, input_image_path, output_image_path)

    assert os.path.exists(output_image_path)

    with rasterio.open(output_image_path) as out:
        assert out.transform


def test_sliding_window(input_image_path):
    window_size = (128, 128)
    step_size = (64, 64)
    with rasterio.open(input_image_path) as src:
        image = src.read()
        windows = [w for w in sliding_window(image, step_size, window_size)]
        assert len(windows[0]) == 3
        assert windows[0][2].shape[1:] == window_size
