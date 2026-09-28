"""Mamba signal adaptation from raw I/Q to an adapted snapshot sequence."""

import math
from numbers import Integral, Real

from mamba_ssm import Mamba
import torch
from torch import nn


def _require_dict(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _required(config, key, section):
    if key not in config:
        raise KeyError(f"Missing config key: {section}.{key}")
    return config[key]


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class ResidualMambaBlock(nn.Module):
    """Pre-norm residual block preserving (B, T, d_model)."""

    def __init__(self, d_model, d_state, d_conv, expand, dt_rank):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
        )

    def forward(self, h):
        return h + self.mamba(self.norm(h))


class MambaSignalAdapter(nn.Module):
    """Adapt (B, IQ, M, T) to (B, T, M * IQ) using caller-provided config.

    Each snapshot interleaves antenna I/Q pairs: [I1, Q1, I2, Q2, ...].
    A small, non-zero delta projection starts signal adaptation near identity
    while allowing gradients to reach the Mamba blocks on the first backward.
    """

    def __init__(self, dataset_config, model_config):
        super().__init__()
        _require_dict(dataset_config, "dataset_config")
        _require_dict(model_config, "model_config")

        self.snapshots = _positive_int(
            _required(dataset_config, "snapshots", "dataset"), "dataset.snapshots"
        )
        self.antennas = _positive_int(
            _required(dataset_config, "antennas", "dataset"), "dataset.antennas"
        )
        self.iq_channels = _positive_int(
            _required(dataset_config, "iq_channels", "dataset"), "dataset.iq_channels"
        )
        if self.iq_channels != 2:
            raise ValueError("dataset.iq_channels must equal 2 for I/Q signals")
        self.input_dim = self.antennas * self.iq_channels

        mamba_config = _require_dict(
            _required(model_config, "mamba", "model"), "model.mamba"
        )
        mamba_args = {
            key: _positive_int(
                _required(mamba_config, key, "model.mamba"), f"model.mamba.{key}"
            )
            for key in ("d_model", "d_state", "d_conv", "expand")
        }
        num_layers = _positive_int(
            _required(mamba_config, "num_layers", "model.mamba"), "model.mamba.num_layers"
        )
        dt_rank = _required(mamba_config, "dt_rank", "model.mamba")
        if not (isinstance(dt_rank, str) and dt_rank == "auto"):
            dt_rank = _positive_int(dt_rank, "model.mamba.dt_rank ('auto' or integer)")
        mamba_args["dt_rank"] = dt_rank

        adapter_config = _require_dict(
            _required(model_config, "signal_adapter", "model"), "model.signal_adapter"
        )
        delta_init_std = _required(adapter_config, "delta_init_std", "model.signal_adapter")
        if (
            isinstance(delta_init_std, bool)
            or not isinstance(delta_init_std, Real)
            or not math.isfinite(delta_init_std)
            or delta_init_std <= 0
        ):
            raise ValueError("model.signal_adapter.delta_init_std must be finite and > 0")

        d_model = mamba_args["d_model"]
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMambaBlock(**mamba_args) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.delta_proj = nn.Linear(d_model, self.input_dim)
        # Initialize only the delta projection, preserving Mamba's own initialization.
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=float(delta_init_std))
        nn.init.zeros_(self.delta_proj.bias)

    def _to_snapshot_sequence(self, x):
        """Validate and interleave (B, IQ, M, T) into (B, T, M * IQ)."""
        expected = (self.iq_channels, self.antennas, self.snapshots)
        layout = f"(B, IQ, M, T) = (B, {expected[0]}, {expected[1]}, {expected[2]})"
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor with layout {layout}; actual type {type(x).__name__}")
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input layout {layout}; actual shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"Expected floating-point input with layout {layout}; actual dtype {x.dtype}, shape {tuple(x.shape)}")

        # (B, IQ, M, T) -> (B, T, M, IQ) -> (B, T, input_dim).
        s = x.permute(0, 3, 2, 1)
        return s.contiguous().reshape(x.shape[0], self.snapshots, self.input_dim)

    def forward(self, x):
        s = self._to_snapshot_sequence(x)  # Raw signal: (B, T, input_dim).
        h = self.input_proj(s)  # Latent features: (B, T, d_model).
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        delta_s = self.delta_proj(h)  # Signal-adaptation delta: (B, T, input_dim).
        s_adapted = s + delta_s
        return s_adapted
