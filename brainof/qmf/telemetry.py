"""Read-only summaries of the actual router decisions and QMF objective."""
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from models.transformer import Router


class BatchTrace:
    def __init__(self, run_dir, model):
        self.file = (Path(run_dir) / "batches.jsonl").open("x", buffering=1)
        self.routers = {n: m for n, m in model.named_modules() if isinstance(m, Router)}
        self.routing = {}
        for name, module in self.routers.items():
            module.observe_routing = partial(self.route, name)

    @torch.no_grad()
    def route(self, name, router, probs, weights, indices):
        # Copy small reductions, not token activations; bias is the one used for selection.
        mass = probs.new_zeros(router.num_experts)
        mass.scatter_add_(0, indices.flatten(), weights.flatten().float())
        self.routing[name] = torch.stack((router.counts, probs.mean(0),
                                          mass / indices.shape[0], router.probs_correction_bias))

    @torch.no_grad()
    def write(self, split, epoch, batch, sigma, targets, idx, output, args, **context):
        loss, reg, fused, logits, weights, hard_ce = output
        branches, fused_ce, ranks = self.terms
        streams = list(logits)
        scalars = torch.stack([loss, reg, fused_ce, *branches.values(), *ranks.values()]).tolist()
        arr = torch.stack([torch.cat((weights[s][:, None], hard_ce[s][:, None], logits[s]), 1)
                           for s in streams]).detach().cpu().numpy()
        y = targets.argmax(1).cpu().numpy()
        pred = fused.argmax(1).cpu().numpy()
        row = dict(split=split, epoch=epoch, batch=batch, n=len(y), sigma=sigma,
                   loss=scalars[0], rank_raw=scalars[1], rank_weighted=args.lamb * scalars[1],
                   fusion_ce_term=scalars[2], idx=idx.tolist(), y=y.tolist(),
                   accuracy=float((pred == y).mean()), streams={}, moe={}, **context)
        parts = arr[:, :, 0:1] * arr[:, :, 2:]
        for i, s in enumerate(streams):
            w, ce, lg = arr[i, :, 0], arr[i, :, 1], arr[i, :, 2:]
            # LEMON is binary. Leave-one-stream-out holds all other logits and weights fixed.
            margin = (parts[i, :, 1] - parts[i, :, 0]) * (2 * y - 1)
            branch = dict(ce=scalars[3 + i], hard_ce=float(ce.mean()),
                          rank_raw=scalars[3 + len(streams) + i], w_mean=float(w.mean()),
                          w_std=float(w.std()), weight=w.tolist(), logits=lg.tolist(),
                          weighted_true_margin=float(margin.mean()), accuracy=float((lg.argmax(1) == y).mean()))
            if len(streams) > 1:
                without = np.delete(parts, i, axis=0).sum(0).argmax(1)
                branch.update(without_correct=int((without == y).sum()), without_flips=int((without != pred).sum()))
            row["streams"][s] = branch
        if self.routing.keys() != self.routers.keys():
            raise RuntimeError(f"missing router observations: {self.routers.keys() - self.routing.keys()}")
        values = torch.stack(list(self.routing.values())).cpu().numpy().astype(np.float64).round(7)
        for name, (counts, probs, mass, bias) in zip(self.routing, values):
            row["moe"][name] = dict(count=counts.astype(int).tolist(), sigmoid_mean=probs.tolist(),
                                    gate_mass_per_token=mass.tolist(), bias_used=bias.tolist())
        self.routing.clear()
        self.file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self):
        for module in self.routers.values():
            del module.observe_routing
        self.file.close()
