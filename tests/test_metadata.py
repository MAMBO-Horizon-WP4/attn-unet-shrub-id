from shrubnet.metadata import means_and_stds


def test_metadata(images_dir, labels_dir):
    metadata = means_and_stds(images_dir, labels_dir)
    assert "std" in metadata
    assert len(metadata["std"]) == 3
