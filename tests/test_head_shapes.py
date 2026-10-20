import torch

from inference.head import FrozenFeatureHead


def test_state_dict_keys_match_research_layout():
    keys = set(FrozenFeatureHead(49, 1280, 128).state_dict())
    assert keys == {
        "audio.pool.layer_logits", "audio.pool.mu", "audio.pool.sd",
        "audio.proj.1.weight", "audio.proj.1.bias",
        "classifier.0.weight", "classifier.0.bias", "classifier.2.weight", "classifier.2.bias",
    }


def test_forward_and_roundtrip():
    h = FrozenFeatureHead(49, 1280, 128).eval()
    x = torch.randn(2, 49, 2, 1280)
    y = h(x)
    assert y.shape == (2, 1)
    h2 = FrozenFeatureHead(49, 1280, 128).eval()
    h2.load_state_dict(h.state_dict())
    torch.testing.assert_close(h2(x), y)


def test_layer_range_and_linear_probe():
    assert FrozenFeatureHead(49, 1280, 0, layer_range=(0, 16)).eval()(torch.randn(1, 49, 2, 1280)).shape == (1, 1)
