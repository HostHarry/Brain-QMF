"""Three-stream wrapper around the official Brain-OF models.

Streams (each one is an independent copy of the pretrained backbone with its own linear head):
  m1    : unimodal Brain-OF on modality 1 (LEMON: fMRI, the first element of every paired sample)
  m2    : unimodal Brain-OF on modality 2 (LEMON: EEG, the second element)
  fused : official multimodal Brain-OF (serial ARNESS fusion of modality 1 then modality 2)
"""
import re
import torch
from torch import nn

from models import transformer, transformer_multimodal

STREAMS = ("m1", "m2", "fused")


def _build(module, cfg):
    """Same constructor call as main_finetuning.py / main_finetuning_multimodal.py."""
    return getattr(module, cfg.model.name)(
        num_classes=cfg.model.num_classes,
        drop_path_rate=cfg.model.droppath,
        drop_path_type=cfg.model.droppath_type,
        max_seq_length=cfg.model.max_seq_length,
        qk_scale=cfg.model.qk_scale,
        ffn_dropout_rate=cfg.model.dropout,
        attn_dropout_rate=cfg.model.attn_drop,
        layer_scale_type=cfg.model.layer_scale_type,
        layer_scale_init_value=cfg.model.init_values,
        use_checkpoint=cfg.model.use_checkpoint,
        checkpoint_every_n=cfg.model.checkpoint_every_n,
        checkpoint_blk_ids=cfg.model.checkpoint_blk_ids,
        weight_init_style=cfg.model.weight_init_style,
        classifier_head_init_scale=cfg.model.init_scale,
    )


def encoder_state_dict(checkpoint_path: str) -> dict:
    """Pretraining checkpoint -> encoder weights (same rule as the official fine-tuning scripts)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {k.replace("encoder.", ""): v for k, v in checkpoint["model"].items() if k.startswith("encoder.")}


def remap_for_multimodal(state: dict, model_name: str) -> dict:
    """Copy of the DDP branch of main_finetuning_multimodal.py: one pretrained ARNESS is split over arness1/arness2."""
    if "base" in model_name or "large" in model_name:
        plan = {0: [("arness1", 0)],
                1: [("arness1", 1), ("arness1", 2), ("arness1", 3)],
                2: [("arness2", 0), ("arness2", 1), ("arness2", 2)],
                3: [("arness2", 3)]}
    elif "huge" in model_name:
        plan = {0: [("arness1", 0)],
                1: [("arness1", 1)],
                2: [("arness1", 2), ("arness1", 3), ("arness1", 4), ("arness1", 5)],
                3: [("arness2", 0), ("arness2", 1), ("arness2", 2), ("arness2", 3)],
                4: [("arness2", 4)],
                5: [("arness2", 5)]}
    else:
        raise ValueError(f"The model name doesn't match {model_name}!")

    out = dict(state)
    layer_pattern = re.compile(r"(.*?arness\.ARNESSLayers\.)(\d+)(\..+)")
    for k, v in list(state.items()):
        if k.startswith("patch_embed."):
            out[k.replace("patch_embed.", "patch_embed1.")] = v
            out[k.replace("patch_embed.", "patch_embed2.")] = v
            del out[k]
        elif k.startswith("arness."):
            if k == "arness.latents":
                out["arness1.latents"] = v
                out["arness2.latents"] = v
            match = layer_pattern.search(k)
            if match:
                prefix, src_idx, suffix = match.groups()
                for target_module, target_idx in plan.get(int(src_idx), []):
                    out[f"{prefix.replace('arness', target_module)}{target_idx}{suffix}"] = v
            del out[k]
    return out


class QMFBrainOF(nn.Module):
    def __init__(self, cfg, streams):
        super().__init__()
        streams = tuple(streams)
        if not streams or any(s not in STREAMS for s in streams) or len(set(streams)) != len(streams):
            raise ValueError(f"streams must be a non-empty subset of {STREAMS}, got {streams}")
        self.streams = tuple(s for s in STREAMS if s in streams)  # canonical order
        self.model_name = cfg.model.name
        self.nets = nn.ModuleDict()
        for s in self.streams:
            self.nets[s] = _build(transformer_multimodal if s == "fused" else transformer, cfg)

    def load_pretrained(self, checkpoint_path: str) -> dict:
        """Initialise every stream from the same pretraining checkpoint. Only the heads may stay uninitialised."""
        state = encoder_state_dict(checkpoint_path)
        report = {}
        for s, net in self.nets.items():
            sd = remap_for_multimodal(state, self.model_name) if s == "fused" else state
            msg = net.load_state_dict(sd, strict=False)
            missing = [k for k in msg.missing_keys if not k.startswith("head.")]
            if missing or msg.unexpected_keys:
                raise RuntimeError(
                    f"[{s}] the checkpoint does not match the model. Missing (non-head): {missing[:8]} "
                    f"Unexpected: {list(msg.unexpected_keys)[:8]}. Check that cfg.model.name fits the checkpoint.")
            report[s] = list(msg.missing_keys)
        return report

    def forward(self, signals, masks) -> dict:
        """signals = (x1, x2) with x of shape (B, N, 64); masks = (m1, m2) bool (B, N). Returns {stream: logits (B, C)}."""
        x1, x2 = signals
        k1, k2 = masks
        out = {}
        if "m1" in self.nets:
            out["m1"] = self.nets["m1"](x1, k1)
        if "m2" in self.nets:
            out["m2"] = self.nets["m2"](x2, k2)
        if "fused" in self.nets:
            out["fused"] = self.nets["fused"]((x1, x2), (k1, k2))
        return out
