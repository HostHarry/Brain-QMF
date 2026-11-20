"""QMF core math: energy uncertainty, fusion weights, ranking regulariser, loss history.

Reference: Zhang et al., "Provable Dynamic Fusion for Low-Quality Multimodal Data", ICML 2023.
  Eq. 9   w^m(x) = alpha^m * u^m(x) + beta^m, alpha^m < 0
  Eq. 11  f(x)   = sum_m w^m(x) * f^m(x)
  Eq. 13  Energy = -T * log sum_k exp(f_k / T)
  Eq. 14  kappa  = average training loss of a sample along the training trajectory
  Eq. 16  L_reg  = max(0, g(w_i, w_j) * (kappa_i - kappa_j) + |w_i - w_j|)
  Eq. 18  L      = CE(y, f) + sum_m CE(y, f^m) + lambda * L_reg
"""
import numpy as np
import torch
import torch.nn.functional as F


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Eq. 13. Shape (B, C) -> (B,). Lower energy means a more confident (in-distribution) sample."""
    logits = logits.float()
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def fusion_weights(logits: dict, fusion: str = "qmf", alpha: float = -0.1, beta: float = 0.0,
                   temperature: float = 1.0) -> dict:
    """Per-sample fusion weight of every stream. Returns {stream: (B,) float32 tensor}.

    fusion == "qmf"   : w = alpha * Energy + beta (Eq. 9). The defaults alpha=-0.1, beta=0, T=1 reproduce the
                        official implementation (conf = -0.1 * energy). Gradients flow through w.
    fusion == "static": w = 1 / M for every sample (plain late fusion, the control condition).
    """
    names = list(logits.keys())
    if fusion == "static":
        ref = logits[names[0]]
        w = torch.full((ref.shape[0],), 1.0 / len(names), device=ref.device, dtype=torch.float32)
        return {s: w for s in names}
    if fusion != "qmf":
        raise ValueError(f"unknown fusion: {fusion}")
    if alpha >= 0:
        raise ValueError("alpha must be negative so that the weight decreases with uncertainty (Eq. 9)")
    return {s: alpha * energy_score(logits[s], temperature) + beta for s in names}


def fuse(logits: dict, weights: dict) -> torch.Tensor:
    """Eq. 11: decision-level weighted sum of the per-stream logits. Returns (B, C)."""
    out = None
    for s in logits:
        term = weights[s].unsqueeze(1) * logits[s].float()
        out = term if out is None else out + term
    return out


class LossHistory:
    """Per-sample running mean of the unimodal training loss (Eq. 14), one instance per stream."""

    def __init__(self, num_samples: int, device):
        self.sum = torch.zeros(num_samples, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(num_samples, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, idx: torch.Tensor, loss: torch.Tensor) -> None:
        loss = loss.detach().float()
        self.sum.index_add_(0, idx, loss)
        self.cnt.index_add_(0, idx, torch.ones_like(loss))

    @torch.no_grad()
    def kappa(self, idx: torch.Tensor):
        """Returns (kappa, seen): kappa is min-max normalised to [0, 1] over all samples seen so far."""
        seen = self.cnt > 0
        mean = self.sum / self.cnt.clamp(min=1.0)
        if bool(seen.any()):
            lo, hi = mean[seen].min(), mean[seen].max()
        else:
            lo, hi = mean.new_zeros(()), mean.new_zeros(())
        k = (mean[idx] - lo) / (hi - lo).clamp(min=1e-8)
        return k, seen[idx]

    def state_dict(self) -> dict:
        return {"sum": self.sum.cpu(), "cnt": self.cnt.cpu()}


def rank_loss(w: torch.Tensor, idx: torch.Tensor, history: LossHistory, mode: str = "paper") -> torch.Tensor:
    """Ranking regulariser between each sample i and its neighbour j = i + 1 in the batch (cyclic).

    mode == "paper"   : Eq. 16 as printed, loss_ij = relu(sign(w_i - w_j) * (k_i - k_j) + |w_i - w_j|).
    mode == "official": what the released code computes, loss_ij = relu(sign(k_i - k_j) * (w_i - w_j) - |k_i - k_j|).
    Pairs in which either sample has no loss history yet contribute nothing.
    """
    k, seen = history.kappa(idx)
    w = w.float()
    wj, kj, seenj = torch.roll(w, -1), torch.roll(k, -1), torch.roll(seen, -1)
    valid = (seen & seenj).float()
    if mode == "paper":
        g = torch.sign(w.detach() - wj.detach())
        per_pair = F.relu(g * (k - kj) + (w - wj).abs())
    elif mode == "official":
        t = torch.sign(k - kj)
        per_pair = F.relu(t * (w - wj) - (k - kj).abs())
    else:
        raise ValueError(f"unknown rank mode: {mode}")
    return (per_pair * valid).sum() / valid.sum().clamp(min=1.0)


def pearson_r(a, b) -> float:
    """Pearson correlation of two 1-D arrays; nan when either one is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def trim_padding(x: torch.Tensor, mask: torch.Tensor):
    """Drop the trailing all-padding tokens of a batch. x (B, N, P), mask (B, N) bool. Padding is always trailing."""
    n = int(mask.sum(dim=1).max().item())
    n = max(n, 1)
    return x[:, :n], mask[:, :n]


def corrupt(x: torch.Tensor, mask: torch.Tensor, sigma: float, apply: torch.Tensor, generator: torch.Generator):
    """Add N(0, sigma^2) noise to the valid tokens of the samples selected by `apply` (B,) bool.

    Signals are z-scored per channel by DataTransform, so sigma is in units of the channel standard deviation.
    """
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device) * sigma
    sel = (apply.to(x.device).unsqueeze(1) & mask).unsqueeze(-1)
    return torch.where(sel, x + noise.to(x.dtype), x)
