import os
import pytest


@pytest.fixture
def fixture_dir():
    """
    Base directory for the test fixtures (images, metadata)
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/")


@pytest.fixture
def input_image_path(fixture_dir):
    return os.path.join(fixture_dir, "test_input.tif")
