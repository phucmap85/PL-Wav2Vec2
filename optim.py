from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _use_muon_for_param(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad or param.ndim != 2:
        return False
    if name.startswith("wav2vec2."):
        return False

    blocked_name_parts = ("emb", "embed", "embedding", "lm_head")
    return not any(part in name.lower() for part in blocked_name_parts)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    try:
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(
            "This training pipeline always uses Muon. Install it with: "
            "pip install git+https://github.com/KellerJordan/Muon"
        ) from exc

    muon_params, adam_params, muon_names, adam_names = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _use_muon_for_param(name, param):
            muon_params.append(param)
            muon_names.append(name)
        else:
            adam_params.append(param)
            adam_names.append(name)

    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": float(cfg["muon_learning_rate"]),
                "momentum": float(cfg["muon_momentum"]),
                "weight_decay": float(cfg["muon_weight_decay"]),
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "use_muon": False,
                "lr": float(cfg["adam_learning_rate"]),
                "betas": tuple(cfg["adam_betas"]),
                "eps": float(cfg["adam_eps"]),
                "weight_decay": float(cfg["adam_weight_decay"]),
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found for the optimizer.")

    use_distributed_muon = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    optimizer_cls = MuonWithAuxAdam if use_distributed_muon else SingleDeviceMuonWithAuxAdam

    print("Muon tensors:", ", ".join(muon_names) if muon_names else "none")
    print("Aux Adam tensors:", ", ".join(adam_names) if adam_names else "none")
    return optimizer_cls(param_groups)


def _load_optimizer_state_from_checkpoint(checkpoint_path: Path):
    optimizer_file = Path(checkpoint_path) / "optimizer.pt"
    if not optimizer_file.exists():
        return None

    try:
        return torch.load(str(optimizer_file), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(optimizer_file), map_location="cpu")


def _optimizer_group_sizes_from_optimizer(optimizer):
    return [len(group["params"]) for group in optimizer.param_groups]


def _optimizer_group_sizes_from_checkpoint(checkpoint_path: Path):
    state = _load_optimizer_state_from_checkpoint(checkpoint_path)
    if state is None:
        return None
    return [len(group.get("params", [])) for group in state.get("param_groups", [])]


def choose_resume_checkpoint_for_trainer(checkpoint_path, optimizer):
    if checkpoint_path is None:
        return None
    if optimizer is None:
        return str(checkpoint_path)

    checkpoint_path = Path(checkpoint_path)
    saved_group_sizes = _optimizer_group_sizes_from_checkpoint(checkpoint_path)
    if saved_group_sizes is None:
        print(f"No optimizer.pt found in {checkpoint_path}.")
        print("Will load model weights only and start a new optimizer.")
        return None

    current_group_sizes = _optimizer_group_sizes_from_optimizer(optimizer)
    if saved_group_sizes == current_group_sizes:
        print(f"Optimizer groups match: {current_group_sizes}")
        print("Resume full Trainer state.")
        return str(checkpoint_path)

    print("Optimizer groups mismatch.")
    print(f"  checkpoint optimizer groups: {saved_group_sizes}")
    print(f"  current optimizer groups   : {current_group_sizes}")
    print("Will load model weights only and start a new optimizer.")
    return None
