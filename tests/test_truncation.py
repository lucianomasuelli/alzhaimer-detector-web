"""El encoder recortado tiene que dar exactamente las mismas capas que usa la cabeza."""
import numpy as np
import pytest
import torch

from inference.config import ModelSpec
from inference.encoder import XlsrEncoder
from inference.head import FrozenFeatureHead

N_FULL, KEEP, HIDDEN = 4, 2, 32       # 4 capas transformer -> 5 hidden states; la cabeza usa [0, 2)


def _wave(seconds=1.5, seed=0):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(16000 * seconds))).astype(np.float32)


@pytest.fixture(params=[True, False], ids=["stable_ln", "post_ln"])
def encoders(request, tmp_path):
    torch.manual_seed(0)
    full = XlsrEncoder.tiny_random(hidden=HIDDEN, layers=N_FULL, stable_layer_norm=request.param)
    full.model.save_pretrained(tmp_path)
    full.processor.save_pretrained(tmp_path)
    return full, XlsrEncoder.from_pretrained(str(tmp_path), num_layers=KEEP, pad_to=N_FULL + 1)


def test_truncated_layers_identical_and_rest_zero(encoders):
    full, trunc = encoders
    w = _wave()
    a, b = full.embed(w), trunc.embed(w)
    assert a.shape == b.shape == (N_FULL + 1, 2, HIDDEN)
    torch.testing.assert_close(b[:KEEP], a[:KEEP], rtol=0, atol=0)
    assert torch.all(b[KEEP:] == 0)


def test_head_output_identical(encoders):
    full, trunc = encoders
    head = FrozenFeatureHead(N_FULL + 1, HIDDEN, head_dim=8, layer_range=(0, KEEP)).eval()
    with torch.no_grad():
        head.audio["pool"].mu.normal_()
        head.audio["pool"].sd.uniform_(0.5, 2)
        head.audio["pool"].layer_logits.normal_()
    w = _wave(seed=1)
    with torch.no_grad():
        torch.testing.assert_close(head(trunc.embed(w)[None]), head(full.embed(w)[None]), rtol=0, atol=0)


def test_spec_encoder_layers():
    assert ModelSpec(layer_range=[0, 16]).encoder_layers() is None            # default: completo
    assert ModelSpec(layer_range=[0, 16], truncate_encoder=True).encoder_layers() == 16
    with pytest.raises(ValueError):
        ModelSpec(truncate_encoder=True).encoder_layers()
