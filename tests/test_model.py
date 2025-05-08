from shrubnet.model import AttentionUNet


def test_load():
    """Simplest 'will it run' test"""
    model = AttentionUNet()
    assert isinstance(model, AttentionUNet)
