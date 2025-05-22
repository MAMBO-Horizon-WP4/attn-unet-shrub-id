import os
from pathlib import Path
from shrubnet.inference import run_inference
from shrubnet.model import AttentionUNet
import rasterio


def test_inference(input_image_path, tmp_path):

    model = AttentionUNet()
    output_image_path = tmp_path / "test.tif"
    run_inference(model, input_image_path, output_image_path)

    assert os.path.exists(output_image_path)

    with rasterio.open(output_image_path) as out:
        assert out.transform
