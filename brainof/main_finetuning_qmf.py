"""Brain-OF + QMF fine-tuning and evaluation on a paired (two-modality) dataset. Single GPU.

Launch from the BrainOF repository root, always through torchrun (the MoE routers need a process group):

  torchrun --standalone --nnodes=1 --nproc_per_node=1 main_finetuning_qmf.py \
      --cfg configs/BrainOF_Base_LEMON_qmf.yaml --root $LMDB --out_root $OUT \
      --pretrained_model $CKPT --streams m1,m2,fused --fusion qmf --lamb 0.1 --seed 3407 --tag R7
"""
import os
import json
import math
import time
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from timm.scheduler.cosine_lr import CosineLRScheduler

import utils
from utils import create_logger, get_config
from models.transformer import Router
from datasets import BrainSignalDataset_multimodal, DataTransform
from qmf.model import QMFBrainOF, STREAMS
from qmf.core import fusion_weights, fuse, rank_loss, LossHistory, pearson_r, trim_padding, corrupt
from qmf.telemetry import BatchTrace

REFERENCE_BATCH = 512.0  # reference global batch size of the official multimodal fine-tuning script


def get_parse_option():
    parser = argparse.ArgumentParser("Brain-OF + QMF fine-tuning and evaluation script")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    # same switches as the official scripts (consumed by utils.get_config)
    parser.add_argument("--root", type=str, help="root path of the LMDB datasets")
    parser.add_argument("--out_root", type=str, help="root path of the output folder")
    parser.add_argument("--dataset", type=str, help="name of the paired dataset folder")
    parser.add_argument("--batch_size", type=int, help="batch size on the single GPU")
    parser.add_argument("--pretrained_model", type=str, help="pretraining checkpoint (.pth)")
    parser.add_argument("--resume", type=str, help="checkpoint written by this script (used with --eval_mode)")
    parser.add_argument("--accumulation_steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--disable_amp", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--eval_mode", action="store_true", help="evaluate --resume on the test split only")
    # QMF switches
    parser.add_argument("--streams", type=str, default="m1,m2,fused", help="comma separated subset of m1,m2,fused")
    parser.add_argument("--fusion", type=str, default="qmf", choices=["qmf", "static"])
    parser.add_argument("--lamb", type=float, default=0.1, help="weight of the ranking regulariser (Eq. 18)")
    parser.add_argument("--alpha", type=float, default=-0.1, help="slope of weight vs energy (Eq. 9), must be < 0")
    parser.add_argument("--beta", type=float, default=0.0, help="offset of the weight (Eq. 9)")
    parser.add_argument("--temperature", type=float, default=1.0, help="energy temperature (Eq. 13)")
    parser.add_argument("--rank_mode", type=str, default="paper", choices=["paper", "official"])
    parser.add_argument("--reg_start_epoch", type=int, default=1, help="first epoch (0-based) that uses L_reg")
    parser.add_argument("--lr_scale", type=str, default="",
                        help="per-stream learning-rate multipliers, e.g. m1=1.875,m2=0.125 (default 1 for all)")
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed of the YAML")
    parser.add_argument("--tag", type=str, default="run", help="free-text label stored with the results")
    parser.add_argument("--noise_sigmas", type=str, default="0,0.5,1,2", help="test-time Gaussian noise levels")
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--no_save", action="store_true", help="do not write checkpoint-best.pth")
    parser.add_argument("--batch_trace", action="store_true", help="write QMF loss and MoE routing per batch")
    args = parser.parse_args()
    cfg = get_config(args)
    return args, cfg


class Indexed(torch.utils.data.Dataset):
    """Adds the sample index to every item so that the per-sample loss history can be addressed."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        signals, target, masks = self.base[i]
        return signals, target, masks, i


def to_device(batch, device):
    signals, targets, masks, idx = batch
    x1 = signals[0].to(device, non_blocking=True)
    x2 = signals[1].to(device, non_blocking=True)
    k1 = masks[0].to(device, non_blocking=True).bool()
    k2 = masks[1].to(device, non_blocking=True).bool()
    x1, k1 = trim_padding(x1, k1)
    x2, k2 = trim_padding(x2, k2)
    return x1, x2, k1, k2, targets.to(device, non_blocking=True), idx.to(device, non_blocking=True)


def forward_loss(model, x1, x2, k1, k2, targets, idx, args, cfg, criterion, histories, use_reg, train, trace=None):
    """One forward pass. Returns (loss, reg, fused_logits, stream_logits, weights, per_sample_ce)."""
    with torch.amp.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.amp_enable):
        logits = model((x1, x2), (k1, k2))
    logits = {s: v.float() for s, v in logits.items()}
    hard = targets.argmax(dim=1)
    per_ce = {s: F.cross_entropy(v.detach(), hard, reduction="none") for s, v in logits.items()}

    branch_ce = {s: criterion(v, targets) for s, v in logits.items()}
    loss = sum(branch_ce.values())
    reg = torch.zeros((), device=targets.device)
    fused_ce = reg
    rank_terms = {s: reg for s in logits}
    if len(logits) == 1:
        s = next(iter(logits))
        fused = logits[s]
        weights = {s: torch.ones(targets.shape[0], device=targets.device)}
    else:
        weights = fusion_weights(logits, args.fusion, args.alpha, args.beta, args.temperature)
        fused = fuse(logits, weights)
        fused_ce = criterion(fused, targets)
        loss = loss + fused_ce
        if use_reg:
            rank_terms = {s: rank_loss(weights[s], idx, histories[s], args.rank_mode) for s in logits}
            reg = sum(rank_terms.values())
            loss = loss + args.lamb * reg
    if trace is not None:
        trace.terms = ({s: v.detach() for s, v in branch_ce.items()}, fused_ce.detach(),
                       {s: v.detach() for s, v in rank_terms.items()})
    if train and histories is not None:
        for s in logits:
            histories[s].update(idx, per_ce[s])
    return loss, reg, fused, logits, weights, per_ce


def train_one_epoch(model, loader, optimizer, lr_scheduler, loss_scaler, router_alpha, epoch, device,
                    cfg, args, criterion, histories, logger, trace=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(loader)
    accum = cfg.train.accumulation_steps
    use_reg = args.fusion == "qmf" and args.lamb > 0 and epoch >= args.reg_start_epoch and len(model.streams) > 1
    loss_sum, reg_sum, seen, correct = 0.0, 0.0, 0, 0
    start = time.time()
    for it, batch in enumerate(loader):
        x1, x2, k1, k2, targets, idx = to_device(batch, device)
        output = forward_loss(model, x1, x2, k1, k2, targets, idx, args, cfg, criterion,
                              histories, use_reg, train=True, trace=trace)
        loss, reg, fused, _, _, _ = output
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} iteration {it}: {loss.item()}")
        group_size = min(accum, num_steps - (it // accum) * accum)
        update = (it + 1) % accum == 0 or it + 1 == num_steps
        if trace is not None:
            lrs = {g["stream"]: g["lr"] for g in optimizer.param_groups}
        loss_scaler(loss / group_size, optimizer, clip_grad=cfg.train.clip_grad, parameters=model.parameters(),
                    update_grad=update, model=model)
        if update:
            optimizer.zero_grad()
            lr_scheduler.step(epoch * math.ceil(num_steps / accum) + it // accum)
        # auxiliary-loss-free load balancing of the MoE routers, as in the official engine
        for m in model.modules():
            if isinstance(m, Router):
                m.update_bias(aux_loss_free_alpha=router_alpha)
        if trace is not None:
            trace.write("train", epoch + 1, it + 1, 0.0, targets, idx, output, args,
                        rank_active=use_reg, lr=lrs, optimizer_update=update, accumulation_size=group_size)

        bsz = targets.shape[0]
        loss_sum += loss.item() * bsz
        reg_sum += float(reg) * bsz
        seen += bsz
        correct += int((fused.argmax(dim=1) == targets.argmax(dim=1)).sum())
        if it % cfg.print_freq == 0:
            logger.info(f"Train: [{epoch + 1}/{cfg.train.epochs}][{it + 1}/{num_steps}] "
                        f"lr {optimizer.param_groups[0]['lr']:.3e} loss {loss.item():.4f} reg {float(reg):.4f} "
                        f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MB")
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(time.time() - start))}")
    return {"loss": loss_sum / max(seen, 1), "reg": reg_sum / max(seen, 1), "acc": correct / max(seen, 1)}


def binary_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    out = {"bac": float(balanced_accuracy_score(y, logits.argmax(axis=1)))}
    if logits.shape[1] == 2 and len(np.unique(y)) == 2:
        z = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        out["auroc"] = float(roc_auc_score(y, prob[:, 1]))
    return out


@torch.no_grad()
def evaluate(model, loader, device, cfg, args, criterion, sigma=0.0, return_outputs=False,
             trace=None, split="test", epoch=None):
    """Dataset-level metrics. With sigma > 0 each modality of each sample is corrupted independently with p = 0.5."""
    model.eval()
    n = len(loader.dataset)
    gen = torch.Generator().manual_seed(args.noise_seed)
    flags = torch.rand(n, 2, generator=gen) < 0.5  # column 0: modality 1, column 1: modality 2
    col = {"y": [], "idx": [], "fused": []}
    per = {s: {"logits": [], "w": [], "ce": []} for s in model.streams}
    loss_sum, seen, bac_batch = 0.0, 0, 0.0
    for it, batch in enumerate(loader):
        x1, x2, k1, k2, targets, idx = to_device(batch, device)
        if sigma > 0:
            x1 = corrupt(x1, k1, sigma, flags[idx.cpu(), 0], gen)
            x2 = corrupt(x2, k2, sigma, flags[idx.cpu(), 1], gen)
        output = forward_loss(model, x1, x2, k1, k2, targets, idx, args, cfg,
                              criterion, None, False, train=False, trace=trace)
        loss, _, fused, logits, weights, per_ce = output
        if trace is not None:
            trace.write(split, epoch, it + 1, sigma, targets, idx, output, args, rank_active=False)
        y = targets.argmax(dim=1)
        bsz = targets.shape[0]
        loss_sum += loss.item() * bsz
        seen += bsz
        # batch-averaged BAC: the quantity the official engines log (kept for comparison with their logs)
        bac_batch += float(balanced_accuracy_score(y.cpu().numpy(), fused.argmax(dim=1).cpu().numpy())) * bsz
        col["y"].append(y.cpu())
        col["idx"].append(idx.cpu())
        col["fused"].append(fused.cpu())
        for s in model.streams:
            per[s]["logits"].append(logits[s].cpu())
            per[s]["w"].append(weights[s].float().cpu())
            per[s]["ce"].append(per_ce[s].cpu())

    y = torch.cat(col["y"]).numpy()
    fused = torch.cat(col["fused"]).numpy()
    metrics = binary_metrics(fused, y)
    metrics["loss"] = loss_sum / max(seen, 1)
    metrics["bac_batchavg"] = bac_batch / max(seen, 1)
    metrics["n"] = int(seen)
    outputs = {"y": y, "idx": torch.cat(col["idx"]).numpy(), "fused": fused}
    for s in model.streams:
        lg = torch.cat(per[s]["logits"]).numpy()
        w = torch.cat(per[s]["w"]).numpy()
        ce = torch.cat(per[s]["ce"]).numpy()
        metrics[f"bac_{s}"] = float(balanced_accuracy_score(y, lg.argmax(axis=1)))
        metrics[f"w_mean_{s}"] = float(w.mean())
        metrics[f"w_negative_frac_{s}"] = float((w < 0).mean())
        metrics[f"pearson_w_loss_{s}"] = pearson_r(w, ce)  # QMF Theorem 2 wants this <= 0
        outputs[f"logits_{s}"], outputs[f"w_{s}"], outputs[f"ce_{s}"] = lg, w, ce
    if trace is not None and len(model.streams) > 1:
        parts = {s: outputs[f"logits_{s}"] * outputs[f"w_{s}"][:, None] for s in model.streams}
        for s in model.streams:
            remaining = sum(v for name, v in parts.items() if name != s).argmax(axis=1)
            without = float(balanced_accuracy_score(y, remaining))
            metrics[f"bac_without_{s}"] = without
            metrics[f"delta_bac_pp_{s}"] = 100 * (metrics["bac"] - without)
    return (metrics, outputs) if return_outputs else metrics


def noise_sweep(model, loader, device, cfg, args, criterion, logger, trace=None, epoch=None) -> dict:
    result = {}
    for sigma in [float(v) for v in args.noise_sigmas.split(",") if v.strip() != ""]:
        result[str(sigma)] = evaluate(model, loader, device, cfg, args, criterion, sigma=sigma,
                                      trace=trace, split="noise_test", epoch=epoch)
        logger.info(f"noise sigma {sigma}: " + json.dumps(result[str(sigma)]))
    return result


def main():
    args, cfg = get_parse_option()
    utils.init_distributed_mode(cfg)
    if not cfg.distribute.distributed_mode:
        raise SystemExit("Launch with: torchrun --standalone --nnodes=1 --nproc_per_node=1 main_finetuning_qmf.py ...")
    if dist.get_world_size() != 1:
        raise SystemExit("main_finetuning_qmf.py supports exactly one GPU (the loss history is kept in one process).")
    if args.seed is not None:
        cfg.defrost()
        cfg.seed = args.seed
        cfg.freeze()
    if cfg.aug.mixup_alpha > 0:
        raise SystemExit("Set aug.mixup_alpha: 0.0 in the YAML: mixed samples have no per-sample loss history.")
    if cfg.train.criterion.task != "classification":
        raise SystemExit("Only classification is implemented (the energy score needs class logits).")
    streams = tuple(s.strip() for s in args.streams.split(",") if s.strip())

    writer, logger = create_logger(cfg, dist_rank=dist.get_rank(), name=f"qmf-{args.tag}")
    run_dir = os.path.join(cfg.out_root, "results", cfg.dataset.name, cfg.save_dir)
    logger.info(cfg.dump())
    logger.info("QMF arguments: " + json.dumps(vars(args)))
    with open(os.path.join(run_dir, "qmf_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if cfg.seed is not None:
        utils.set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ------ data ------
    transform = DataTransform(signal_size=cfg.dataset.signal_size, patch_size=cfg.model.patch_size,
                              max_seq_length=cfg.model.max_seq_length)
    sets = {split: Indexed(BrainSignalDataset_multimodal(root=cfg.root, datasets=cfg.dataset.name, split=split,
                                                         transform=transform)) for split in ("train", "val", "test")}
    for split, ds in sets.items():
        if len(ds) == 0:
            raise SystemExit(f"No samples in {os.path.join(cfg.root, cfg.dataset.name, split)} - check --root.")
        logger.info(f"{split}: {len(ds)} paired samples")
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                  multiprocessing_context="fork" if cfg.num_workers > 0 else None)
    train_loader = torch.utils.data.DataLoader(sets["train"], shuffle=True, drop_last=True, **common)
    val_loader = torch.utils.data.DataLoader(sets["val"], shuffle=False, drop_last=False, **common)
    test_loader = torch.utils.data.DataLoader(sets["test"], shuffle=False, drop_last=False, **common)
    if len(train_loader) < cfg.train.accumulation_steps:
        raise SystemExit("Fewer training batches than accumulation steps: lower --batch_size/--accumulation_steps.")

    # ------ model ------
    model = QMFBrainOF(cfg, streams)
    if cfg.model.resume:
        state = torch.load(cfg.model.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        logger.info(f"resumed {cfg.model.resume} (epoch {state.get('epoch')})")
    elif cfg.model.pretrained_model:
        logger.info(f"pretrained init, keys left at random init: {model.load_pretrained(cfg.model.pretrained_model)}")
    else:
        logger.info("WARNING: training from random initialisation (no --pretrained_model)")
    model.to(device)
    trace = BatchTrace(run_dir, model) if args.batch_trace else None
    logger.info(f"streams {model.streams}, parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if cfg.train.criterion.label_smoothing > 0:
        logger.info(f"label smoothing {cfg.train.criterion.label_smoothing}")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.label_smoothing)

    if cfg.eval_mode:
        if not cfg.model.resume:
            raise SystemExit("--eval_mode needs --resume")
        test_metrics, outputs = evaluate(model, test_loader, device, cfg, args, criterion,
                                         return_outputs=True, trace=trace)
        logger.info("test " + json.dumps(test_metrics))
        np.savez_compressed(os.path.join(run_dir, "test_outputs.npz"), **outputs)
        result = {"args": vars(args), "streams": list(model.streams), "test": test_metrics,
                  "noise": noise_sweep(model, test_loader, device, cfg, args, criterion, logger, trace)}
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        if trace is not None:
            trace.close()
        dist.destroy_process_group()
        return

    # ------ optimiser and schedule: same rules as main_finetuning_multimodal.py ------
    accum = cfg.train.accumulation_steps
    scale = cfg.train.batch_size * accum / REFERENCE_BATCH
    lr = cfg.train.optimizer.base_lr * scale
    warmup_lr = cfg.train.lr_scheduler.warmup_lr * scale
    min_lr = cfg.train.lr_scheduler.min_lr * scale
    router_alpha = cfg.train.aux_loss_free_alpha * scale
    n_iter_per_epoch = math.ceil(len(train_loader) / accum)
    logger.info(f"effective batch {cfg.train.batch_size * accum}, lr {lr:.3e}, router alpha {router_alpha:.3e}, "
                f"{n_iter_per_epoch} optimizer steps per epoch")
    no_wd_keywords = ["gamma", "norm", "lambda_qk", "patch_embed", "router.gate", "latents"]
    lr_mult = {s: 1.0 for s in STREAMS}
    for item in args.lr_scale.split(","):
        if item.strip():
            name, value = item.split("=")
            if name.strip() not in STREAMS:
                raise SystemExit(f"--lr_scale: unknown stream {name}")
            lr_mult[name.strip()] = float(value)
    param_groups = []
    for s, net in model.nets.items():  # one call per stream: layer ids are parsed from names relative to the net
        values = list(cfg.train.optimizer.layer_decay ** i for i in reversed(range(net.num_layers + 2)))
        assigner = utils.LayerDecayValueAssigner(values, "DDP")
        groups = utils.get_finetune_param_groups_ddp(net, lr, cfg.train.optimizer.weight_decay,
                                                     assigner.get_layer_id, assigner.get_scale,
                                                     skip_keywords=no_wd_keywords)
        for group in groups:  # timm schedulers set lr = scheduled value * group["lr_scale"]
            group["lr_scale"] = group["lr_scale"] * lr_mult[s]
            group["stream"] = s
        param_groups += groups
        logger.info(f"stream {s}: peak lr {lr * lr_mult[s]:.3e}")
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=cfg.train.optimizer.betas, eps=cfg.train.optimizer.eps)
    num_steps = int(cfg.train.epochs * n_iter_per_epoch)
    warmup_steps = int(cfg.train.lr_scheduler.warmup_epochs * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if cfg.train.lr_scheduler.warmup_prefix else num_steps,
        cycle_mul=1.,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        warmup_prefix=cfg.train.lr_scheduler.warmup_prefix,
    )
    loss_scaler = utils.NativeScalerWithGradNormCount()
    histories = {s: LossHistory(len(sets["train"]), device) for s in model.streams}

    # ------ training ------
    best = {"val_bac": -float("inf"), "epoch": -1, "val": None, "test": None}
    best_state, log = None, []
    start_time = time.time()
    for epoch in range(cfg.train.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_scaler, router_alpha,
                                      epoch, device, cfg, args, criterion, histories, logger, trace)
        val_metrics = evaluate(model, val_loader, device, cfg, args, criterion, trace=trace, split="val", epoch=epoch+1)
        entry = {"epoch": epoch, "train": train_stats, "val": val_metrics}
        if val_metrics["bac"] >= best["val_bac"]:  # ties go to the later epoch, as in the official script
            test_metrics, outputs = evaluate(model, test_loader, device, cfg, args, criterion, return_outputs=True,
                                             trace=trace, split="test_at_best", epoch=epoch+1)
            best = {"val_bac": val_metrics["bac"], "epoch": epoch, "val": val_metrics, "test": test_metrics}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            np.savez_compressed(os.path.join(run_dir, "test_outputs.npz"), **outputs)
            if not args.no_save:
                torch.save({"model": best_state, "epoch": epoch, "best": best, "args": vars(args)},
                           os.path.join(run_dir, "checkpoint-best.pth"))
            entry["test"] = test_metrics
        log.append(entry)
        logger.info(f"Epoch {epoch + 1:03d} train {json.dumps(train_stats)}\nval {json.dumps(val_metrics)}\n"
                    f"best val bac {best['val_bac']:.4f} at epoch {best['epoch'] + 1}, "
                    f"its test bac {best['test']['bac']:.4f}")
        if writer is not None:
            writer.add_scalars("bac", {"val": val_metrics["bac"], "test_at_best": best["test"]["bac"]}, epoch)
            writer.add_scalars("loss", {"train": train_stats["loss"], "val": val_metrics["loss"]}, epoch)
        with open(os.path.join(run_dir, "epochs.json"), "w") as f:
            json.dump(log, f, indent=2)

    # ------ robustness of the selected checkpoint ------
    model.load_state_dict(best_state)
    result = {"args": vars(args), "streams": list(model.streams), "seed": cfg.seed, "best_epoch": best["epoch"],
              "val": best["val"], "test": best["test"],
              "noise": noise_sweep(model, test_loader, device, cfg, args, criterion, logger, trace, best["epoch"]+1),
              "train_time": str(datetime.timedelta(seconds=int(time.time() - start_time)))}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"results written to {os.path.join(run_dir, 'results.json')}")
    logger.info("========> Done!!!")
    if trace is not None:
        trace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
