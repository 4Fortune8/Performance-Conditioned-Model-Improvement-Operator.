import torch

from mio.models.base import MLP
from mio.state import OptState
from mio.trajectories.training import BatchSampler, TrainSettings, adamw_update, train_steps
from mio.datasets.tasks import Split


def test_adamw_matches_torch_optim():
    torch.manual_seed(0)
    theta = torch.randn(50)
    ref = torch.nn.Parameter(theta.clone())
    opt_ref = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    ours, state = theta.clone(), OptState.zeros(50)
    for _ in range(5):
        g = torch.randn(50)
        ref.grad = g.clone()
        opt_ref.step()
        adamw_update(ours, g, state, 1e-2, 0.9, 0.999, 1e-8, 0.1)
    assert torch.allclose(ours, ref.detach(), atol=1e-6)


def test_batch_sampler_is_stateless():
    a, b = BatchSampler(100, 10, order_seed=7), BatchSampler(100, 10, order_seed=7)
    seq = [a.indices(k) for k in range(25)]
    assert torch.equal(b.indices(17), seq[17])  # random access == sequential access
    epoch0 = torch.cat(seq[:10])
    assert sorted(epoch0.tolist()) == list(range(100))  # each epoch is a permutation


def _toy():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 6, generator=g)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    m = MLP(6, [8], 3)
    return m, m.init(torch.Generator().manual_seed(1)), Split(x, y)


def test_resume_is_bitwise_exact():
    m, theta, data = _toy()
    s = TrainSettings(lr=1e-2, weight_decay=1e-3, batch_size=16, order_seed=3)
    straight = train_steps(m, theta, OptState.zeros(m.num_params), data, s, 40)
    first = train_steps(m, theta, OptState.zeros(m.num_params), data, s, 20)
    second = train_steps(m, first.theta, first.opt, data, s, 20, data_step_start=20)
    assert torch.equal(straight.theta, second.theta)
    assert torch.equal(straight.opt.m, second.opt.m) and straight.opt.step == second.opt.step == 40


def test_inputs_not_mutated_and_lr_decay():
    m, theta, data = _toy()
    before = theta.clone()
    s = TrainSettings(lr=1e-2, batch_size=16, lr_schedule="linear_decay", decay_steps=10)
    assert s.lr_at(0) == 1e-2 and s.lr_at(10) == 0.0
    train_steps(m, theta, OptState.zeros(m.num_params), data, s, 5)
    assert torch.equal(theta, before)
