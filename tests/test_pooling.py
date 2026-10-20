import torch

from inference.head import layer_stats


def test_layer_stats_shape_and_population_std():
    hs = tuple(torch.randn(1, 50, 8) for _ in range(4))
    out = layer_stats(hs)
    assert out.shape == (4, 2, 8)
    torch.testing.assert_close(out[0, 0], hs[0][0].mean(0))
    torch.testing.assert_close(out[0, 1], hs[0][0].std(0, unbiased=False))


def test_layer_stats_single_frame_has_zero_std_not_nan():
    out = layer_stats(tuple(torch.randn(1, 1, 8) for _ in range(2)))
    assert torch.all(out[:, 1] == 0)
