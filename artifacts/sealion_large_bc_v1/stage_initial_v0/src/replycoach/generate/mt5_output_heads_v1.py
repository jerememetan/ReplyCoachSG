"""Trainable token-output alternatives to LoRA on a frozen mT5 backbone."""

import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file


class ResidualTokenHead(nn.Module):
    """Identity-initialized bottleneck MLP before the frozen vocabulary matrix."""

    def __init__(self, projection, bottleneck=256, dropout=0.1):
        super().__init__()
        self.projection = projection
        width = projection.weight.shape[1]
        self.down = nn.Linear(width, bottleneck).to(projection.weight)
        self.up = nn.Linear(bottleneck, width).to(projection.weight)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, hidden):
        delta = self.up(self.dropout(nn.functional.gelu(self.down(hidden))))
        return self.projection(hidden + delta)


def configure_head(model, variant, bottleneck=256, dropout=0.1):
    if variant not in {"linear", "mlp"}:
        raise ValueError(variant)
    if isinstance(model.lm_head, ResidualTokenHead):
        raise ValueError("Configure each freshly loaded base only once")
    if model.lm_head.weight is model.shared.weight:
        raise ValueError("Output-only training requires the independent stored output matrix")
    model.requires_grad_(False)
    if variant == "linear":
        model.lm_head.weight.requires_grad_(True)
    else:
        model.lm_head = ResidualTokenHead(model.lm_head, bottleneck, dropout)
    actual = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {"lm_head.weight"} if variant == "linear" else {
        "lm_head.down.weight", "lm_head.down.bias", "lm_head.up.weight", "lm_head.up.bias",
    }
    if actual != expected:
        raise ValueError(f"Unexpected trainable parameters: {actual}")
    return model


def train_head_mode(model):
    # Frozen encoders/decoders supply fixed features, including no backbone
    # dropout. Only the MLP branch has dropout during its own training.
    model.eval()
    model.lm_head.train()
    if model.encoder.training or model.decoder.training:
        raise ValueError("Frozen backbone must remain in evaluation mode")


def head_parameters(model, variant):
    return {name: p for name, p in model.lm_head.named_parameters()
            if variant == "linear" or name.startswith(("down.", "up."))}


def save_head(model, path, variant, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {name: p.detach().cpu().contiguous() for name, p in head_parameters(model, variant).items()}
    target = path / "head.safetensors"
    temporary = path / "head.tmp"
    save_file(state, str(temporary))
    temporary.replace(target)
    (path / "head_config.json").write_text(json.dumps(dict(variant=variant, **metadata), indent=2) + "\n")


def load_head(model, path, variant):
    path = Path(path)
    metadata = json.loads((path / "head_config.json").read_text())
    if metadata["variant"] != variant:
        raise ValueError("Wrong task-head variant")
    params = head_parameters(model, variant)
    state = load_file(str(path / "head.safetensors"), device="cpu")
    if state.keys() != params.keys() or any(state[name].shape != p.shape for name, p in params.items()):
        raise ValueError("Task-head tensor names or shapes changed")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state[name].to(p))
    model.requires_grad_(False)
    model.eval()
    return model
