"""Spatial covariance and CNN classification of an adapted I/Q sequence."""

from numbers import Integral

import torch
from torch import nn


def _required(config, key, section):
    if key not in config:
        raise KeyError(f"Missing config key: {section}.{key}")
    return config[key]


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class DOACovarianceCNNHead(nn.Module):
    """Map (B, T, M * IQ) to raw DOA logits using caller-provided config.

    Snapshots contain interleaved [I1, Q1, I2, Q2, ...] antenna pairs.
    The uncentered spatial covariance YY^H / T supplies real and imaginary
    feature channels to three convolutions with one pool after Conv2.
    """

    def __init__(self, dataset_config, model_config):
        super().__init__()
        if not isinstance(dataset_config, dict):
            raise ValueError("dataset_config must be a dictionary")
        if not isinstance(model_config, dict):
            raise ValueError("model_config must be a dictionary")

        self.snapshots = _positive_int(
            _required(dataset_config, "snapshots", "dataset"), "dataset.snapshots"
        )
        self.antennas = _positive_int(
            _required(dataset_config, "antennas", "dataset"), "dataset.antennas"
        )
        self.iq_channels = _positive_int(
            _required(dataset_config, "iq_channels", "dataset"), "dataset.iq_channels"
        )
        self.num_doa_classes = _positive_int(
            _required(dataset_config, "num_doa_classes", "dataset"),
            "dataset.num_doa_classes",
        )
        if self.iq_channels != 2:
            raise ValueError("dataset.iq_channels must equal 2 for I/Q signals")
        self.input_dim = self.antennas * self.iq_channels

        head_config = _required(model_config, "doa_head", "model")
        if not isinstance(head_config, dict):
            raise ValueError("model.doa_head must be a dictionary")
        channels = _required(head_config, "conv_channels", "model.doa_head")
        if not isinstance(channels, (list, tuple)) or len(channels) != 3:
            raise ValueError("model.doa_head.conv_channels must be a list/tuple of 3 integers")
        c1, c2, c3 = (
            _positive_int(value, f"model.doa_head.conv_channels[{index}]")
            for index, value in enumerate(channels)
        )
        kernel_size = _positive_int(
            _required(head_config, "kernel_size", "model.doa_head"),
            "model.doa_head.kernel_size",
        )
        if kernel_size % 2 == 0:
            raise ValueError("model.doa_head.kernel_size must be odd to preserve spatial size")
        pool_kernel_size = _positive_int(
            _required(head_config, "pool_kernel_size", "model.doa_head"),
            "model.doa_head.pool_kernel_size",
        )
        pool_stride = _positive_int(
            _required(head_config, "pool_stride", "model.doa_head"),
            "model.doa_head.pool_stride",
        )
        feature_dim = _positive_int(
            _required(head_config, "feature_dim", "model.doa_head"),
            "model.doa_head.feature_dim",
        )
        if pool_kernel_size > self.antennas:
            raise ValueError(
                f"model.doa_head.pool_kernel_size={pool_kernel_size} must be "
                f"<= dataset.antennas={self.antennas}"
            )
        self.pooled_size = (self.antennas - pool_kernel_size) // pool_stride + 1
        if self.pooled_size < 1:
            raise ValueError("Pooling must preserve a spatial size of at least 1 x 1")
        self.flatten_dim = c3 * self.pooled_size * self.pooled_size

        padding = kernel_size // 2
        # Two covariance channels: real, then imaginary.
        self.conv1 = nn.Conv2d(2, c1, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_proj = nn.Linear(self.flatten_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, self.num_doa_classes)

    def _restore_iq(self, s_adapted):
        """Validate and restore interleaved (B, T, M * IQ) to (B, IQ, M, T)."""
        layout = f"(B, T, input_dim) = (B, {self.snapshots}, {self.input_dim})"
        if not isinstance(s_adapted, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor with layout {layout}; "
                f"actual type {type(s_adapted).__name__}"
            )
        if s_adapted.ndim != 3 or tuple(s_adapted.shape[1:]) != (self.snapshots, self.input_dim):
            raise ValueError(
                f"Expected input layout {layout}; actual shape {tuple(s_adapted.shape)}"
            )
        if not s_adapted.is_floating_point():
            raise TypeError(
                f"Expected real floating-point input with layout {layout}; "
                f"actual dtype {s_adapted.dtype}, shape {tuple(s_adapted.shape)}"
            )

        pairs = s_adapted.reshape(
            s_adapted.shape[0], self.snapshots, self.antennas, self.iq_channels
        )
        return pairs.permute(0, 3, 2, 1).contiguous()

    def _covariance_features(self, iq):
        """Compute uncentered YY^H / T from (B, IQ, M, T) using real operations.

        Returns (B, 2, M, M), ordered as [real, imaginary], with gradients
        preserved through both I and Q. Input is the validated _restore_iq output.
        """
        i_signal = iq[:, 0]  # (B, M, T).
        q_signal = iq[:, 1]
        i_t = i_signal.transpose(-1, -2)
        q_t = q_signal.transpose(-1, -2)
        real_cov = (i_signal @ i_t + q_signal @ q_t) / self.snapshots
        imag_cov = (q_signal @ i_t - i_signal @ q_t) / self.snapshots
        return torch.stack([real_cov, imag_cov], dim=1)

    def forward(self, s_adapted):
        iq = self._restore_iq(s_adapted)
        cov_features = self._covariance_features(iq)  # (B, 2, M, M).
        features = self.relu(self.conv1(cov_features))
        features = self.relu(self.conv2(features))
        features = self.pool(features)  # The only pooling operation.
        features = self.relu(self.conv3(features))
        features = self.flatten(features)  # (B, C3 * pooled_size * pooled_size).
        features = self.relu(self.feature_proj(features))
        doa_logits = self.classifier(features)
        return doa_logits
