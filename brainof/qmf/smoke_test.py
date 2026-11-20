"""Data-free checks. Run from the BrainOF repository root BEFORE any training:

  python -m qmf.smoke_test --cfg configs/BrainOF_Base_LEMON_qmf.yaml [--ckpt /path/to/brainof_base_50M_A22M.pth]

Part 1 (CPU): the QMF formulas on hand-computed cases.
Part 2 (GPU): one forward/backward pass of the three-stream model on random signals with LEMON token counts.
Part 3 (only with --ckpt): the pretraining checkpoint loads into all three streams.
"""
import math
import argparse

import torch

from qmf.core import energy_score, fusion_weights, fuse, rank_loss, LossHistory, pearson_r, trim_padding, corrupt


def part1():
    # Eq. 13: two equal logits -> energy = -log(2)
    e = energy_score(torch.zeros(1, 2))
    assert abs(e.item() + math.log(2.0)) < 1e-6, e
    # Eq. 9: the confident stream gets the larger weight
    logits = {"m1": torch.tensor([[0.1, -0.1]]), "m2": torch.tensor([[4.0, -4.0]])}
    w = fusion_weights(logits, "qmf", alpha=-0.1, beta=0.0, temperature=1.0)
    assert w["m2"].item() > w["m1"].item() > 0, w
    assert abs(w["m2"].item() - 0.1 * torch.logsumexp(logits["m2"], dim=1).item()) < 1e-6
    # Eq. 11: weighted sum of logits
    f = fuse(logits, w)
    expect = w["m1"].item() * 0.1 + w["m2"].item() * 4.0
    assert f.shape == (1, 2) and abs(f[0, 0].item() - expect) < 1e-6, f
    ws = fusion_weights(logits, "static")
    assert abs(ws["m1"].item() - 0.5) < 1e-6 and abs(ws["m2"].item() - 0.5) < 1e-6

    # Eq. 14 / 16: sample 0 is hard (mean loss 2), sample 1 is easy (mean loss 0) -> kappa = [1, 0]
    h = LossHistory(2, "cpu")
    idx = torch.tensor([0, 1])
    h.update(idx, torch.tensor([2.0, 0.0]))
    h.update(idx, torch.tensor([2.0, 0.0]))
    k, seen = h.kappa(idx)
    assert torch.allclose(k, torch.tensor([1.0, 0.0])) and bool(seen.all())
    # wrong order: the hard sample has the larger weight. Paper form: both pairs give relu(1 + 0.4) = 1.4
    w_bad = torch.tensor([0.5, 0.1], requires_grad=True)
    loss_bad = rank_loss(w_bad, idx, h, "paper")
    assert abs(loss_bad.item() - 1.4) < 1e-6, loss_bad
    loss_bad.backward()
    assert w_bad.grad[0] > 0 and w_bad.grad[1] < 0  # gradient descent lowers w of the hard sample
    # right order with |dw| <= |dkappa|: no penalty
    assert rank_loss(torch.tensor([0.1, 0.5]), idx, h, "paper").item() == 0.0
    # released-code form: relu(sign(dk) * dw - |dk|) -> relu(0.4 - 1) = 0 for both orders at this scale
    assert rank_loss(torch.tensor([0.5, 0.1]), idx, h, "official").item() == 0.0
    assert abs(rank_loss(torch.tensor([2.5, 0.1]), idx, h, "official").item() - 1.4) < 1e-6
    # samples without history are ignored
    h2 = LossHistory(2, "cpu")
    assert rank_loss(torch.tensor([0.5, 0.1]), idx, h2, "paper").item() == 0.0

    assert abs(pearson_r([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9 and math.isnan(pearson_r([1, 1, 1], [1, 2, 3]))
    x = torch.zeros(2, 5, 4)
    m = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]).bool()
    xt, mt = trim_padding(x, m)
    assert xt.shape == (2, 3, 4) and mt.shape == (2, 3)
    g = torch.Generator().manual_seed(0)
    xc = corrupt(xt, mt, 1.0, torch.tensor([True, False]), g)
    assert xc[0, :3].abs().sum() > 0 and xc[1].abs().sum() == 0 and xc[0, 2].abs().sum() > 0
    xc2 = corrupt(xt, mt, 1.0, torch.tensor([False, True]), g)
    assert xc2[1, 2].abs().sum() == 0  # padded token of sample 1 stays untouched
    print("part 1 (formulas): OK")


def part2(cfg_path, ckpt):
    from utils import get_config
    from qmf.model import QMFBrainOF
    cfg = get_config(argparse.Namespace(cfg=cfg_path))
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    device = torch.device("cuda")
    model = QMFBrainOF(cfg, ("m1", "m2", "fused"))
    if ckpt:
        print("part 3 (checkpoint): keys left at random init:", model.load_pretrained(ckpt))
    model.to(device).train()
    torch.manual_seed(0)
    b = 2
    x1 = torch.randn(b, 1230, 64, device=device)  # fMRI: 246 ROIs x 5 patches
    x2 = torch.randn(b, 1952, 64, device=device)  # EEG : 61 channels x 32 patches
    k1 = torch.ones(b, 1230, dtype=torch.bool, device=device)
    k2 = torch.ones(b, 1952, dtype=torch.bool, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp_enable):
        logits = model((x1, x2), (k1, k2))
    logits = {s: v.float() for s, v in logits.items()}
    for s, v in logits.items():
        assert v.shape == (b, cfg.model.num_classes) and torch.isfinite(v).all(), (s, v)
    w = fusion_weights(logits, "qmf")
    fused = fuse(logits, w)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    loss = torch.nn.functional.cross_entropy(fused, target) + sum(
        torch.nn.functional.cross_entropy(v, target) for v in logits.values())
    loss.backward()
    for s, net in model.nets.items():
        assert net.head.weight.grad is not None and torch.isfinite(net.head.weight.grad).all(), s
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"part 2 (model): OK - loss {loss.item():.4f}, {n_params:.1f}M parameters, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB for batch {b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="")
    parser.add_argument("--ckpt", default="")
    a = parser.parse_args()
    part1()
    if a.cfg:
        part2(a.cfg, a.ckpt)
