from shrubnet.model import AttentionUNet
from pathlib import Path
import torch


def test_load():
    """Simplest 'will it run' test"""
    model = AttentionUNet()
    assert isinstance(model, AttentionUNet)


def test_save(tmp_path):
    """Test saving and loading the model"""
    model = AttentionUNet()
    model_path = tmp_path / "test_model.pth"

    # Save the model
    torch.save(model.state_dict(), model_path)

    # Load the model
    loaded_model = AttentionUNet()
    loaded_model.load_state_dict(torch.load(model_path))

    # Check if the loaded model is the same as the original
    for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param1, param2)

    # Clean up
    Path(model_path).unlink()  # Remove the test file
