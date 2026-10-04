import pytest
import torch
from torch import nn

from mio.models.base import MLP, IncompatibleModelError, ParamSpec


def test_param_count_and_layout():
    m = MLP(784, [64, 32], 10)
    assert m.num_params == 52650
    assert [t.name for t in m.spec.tensors] == [
        "layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias"]
    assert m.spec.tensors[-1].offset + m.spec.tensors[-1].numel == m.num_params


def test_flatten_roundtrip_and_spec_serialization():
    m = MLP(20, [16, 8], 4)
    theta = m.init(torch.Generator().manual_seed(0))
    assert torch.equal(m.spec.flatten(m.spec.unflatten(theta)), theta)
    spec2 = ParamSpec.from_dict(m.spec.to_dict())
    assert spec2 == m.spec and spec2.structure_hash == m.spec.structure_hash


def test_forward_matches_torch_reference():
    m = MLP(20, [16, 8], 4)
    theta = m.init(torch.Generator().manual_seed(1))
    p = m.spec.unflatten(theta)
    ref = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    with torch.no_grad():
        for i, layer in enumerate([ref[0], ref[2], ref[4]]):
            layer.weight.copy_(p[f"layers.{i}.weight"])
            layer.bias.copy_(p[f"layers.{i}.bias"])
    x = torch.randn(32, 20)
    assert torch.allclose(m.forward(theta, x), ref(x), atol=1e-6)


def test_incompatible_structures_fail_clearly():
    a, b = MLP(20, [16, 8], 4), MLP(20, [16, 4], 4)
    assert a.spec.structure_hash != b.spec.structure_hash
    with pytest.raises(IncompatibleModelError):
        a.spec.check_compatible(b.spec)
    with pytest.raises(IncompatibleModelError):
        a.spec.check_vector(torch.zeros(b.num_params))
    bad = torch.zeros(a.num_params)
    bad[3] = float("nan")
    with pytest.raises(IncompatibleModelError):
        a.spec.check_vector(bad)


def test_hidden_permutation_preserves_function():
    m = MLP(20, [16, 8], 4)
    theta = m.init(torch.Generator().manual_seed(2))
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(3))
    theta_p = m.permute_hidden(theta, 0, perm)
    x = torch.randn(10, 20)
    assert not torch.equal(theta, theta_p)
    assert torch.allclose(m.forward(theta, x), m.forward(theta_p, x), atol=1e-6)
