from numbers import Integral
from pathlib import Path
import re

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class MoDANetDatasetLoader(Dataset):
    """Lazy .mat dataset configured by the caller's dataset dictionary."""

    def __init__(self, dataset_config):
        if not isinstance(dataset_config, dict):
            raise ValueError("dataset_config must be a dictionary")

        required_keys = (
            "root", "mat_variable", "num_samples", "snapshots", "antennas",
            "iq_channels", "num_mod_classes", "modulations", "doa_min",
            "doa_max", "doa_step", "num_doa_classes",
        )
        for key in required_keys:
            if key not in dataset_config:
                raise KeyError(f"Missing dataset config key: {key}")

        root = dataset_config["root"]
        if not isinstance(root, (str, Path)) or not str(root).strip():
            raise ValueError("dataset.root must be a non-empty path")
        self.root_dir = Path(root).expanduser().resolve()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root is not a directory: {self.root_dir}")

        self.mat_variable = dataset_config["mat_variable"]
        if not isinstance(self.mat_variable, str) or not self.mat_variable.strip():
            raise ValueError("dataset.mat_variable must be a non-empty string")

        positive_keys = (
            "num_samples", "snapshots", "antennas", "iq_channels",
            "num_mod_classes", "doa_step", "num_doa_classes",
        )
        for key in (*positive_keys, "doa_min", "doa_max"):
            value = dataset_config[key]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"dataset.{key} must be an integer, got {value!r}")
            if key in positive_keys and value <= 0:
                raise ValueError(f"dataset.{key} must be > 0, got {value!r}")
            setattr(self, key, int(value))

        modulations = dataset_config["modulations"]
        if not isinstance(modulations, (list, tuple)) or not modulations:
            raise ValueError("dataset.modulations must be a non-empty list")
        if any(not isinstance(name, str) or not name.strip() for name in modulations):
            raise ValueError("dataset.modulations must contain non-empty strings")
        if len(set(modulations)) != len(modulations):
            raise ValueError("dataset.modulations must not contain duplicates")
        if self.num_mod_classes != len(modulations):
            raise ValueError(
                f"dataset.num_mod_classes={self.num_mod_classes} does not match "
                f"len(modulations)={len(modulations)}"
            )
        self.modulations = list(modulations)
        self.mod_to_index = {name: idx for idx, name in enumerate(self.modulations)}

        if self.doa_max < self.doa_min:
            raise ValueError("dataset.doa_max must be >= dataset.doa_min")
        doa_span = self.doa_max - self.doa_min
        if doa_span % self.doa_step != 0:
            raise ValueError("dataset.doa_max - doa_min must be divisible by doa_step")
        expected_classes = doa_span // self.doa_step + 1
        if self.num_doa_classes != expected_classes:
            raise ValueError(
                f"dataset.num_doa_classes: expected {expected_classes}, "
                f"actual {self.num_doa_classes}"
            )

        self.expected_shape = (self.snapshots, self.antennas, self.iq_channels)
        # Only enumerate paths here; sample contents are loaded in __getitem__.
        self.files = sorted(self.root_dir.glob("SNR*dB/*/*/*.mat"))
        if len(self.files) != self.num_samples:
            raise ValueError(
                f"Dataset num_samples mismatch: expected {self.num_samples}, "
                f"actual {len(self.files)}, root: {self.root_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        mat = loadmat(file_path)
        if self.mat_variable not in mat:
            raise KeyError(f"Missing MATLAB variable {self.mat_variable!r}: {file_path}")

        x = mat[self.mat_variable]
        if x.shape != self.expected_shape:
            raise ValueError(
                f"Invalid sample shape: actual {x.shape}, "
                f"expected {self.expected_shape}, file: {file_path}"
            )

        # (T, M, IQ) -> (IQ, M, T), preserving sample values.
        x = np.transpose(x, (2, 1, 0))
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

        snr_dir, modulation_name, doa_dir, _ = file_path.relative_to(self.root_dir).parts
        match = re.fullmatch(r"SNR([+-]\d+)dB", snr_dir)
        if match is None:
            raise ValueError(f"Cannot parse SNR from {snr_dir!r}: {file_path}")
        snr_db = int(match.group(1))

        if modulation_name not in self.mod_to_index:
            raise ValueError(f"Unknown modulation {modulation_name!r}: {file_path}")
        mod_label = self.mod_to_index[modulation_name]

        try:
            doa_deg = int(doa_dir)
        except ValueError as exc:
            raise ValueError(f"Cannot parse DOA from {doa_dir!r}: {file_path}") from exc
        if not self.doa_min <= doa_deg <= self.doa_max:
            raise ValueError(
                f"DOA {doa_deg} outside [{self.doa_min}, {self.doa_max}]: {file_path}"
            )
        if (doa_deg - self.doa_min) % self.doa_step != 0:
            raise ValueError(
                f"DOA {doa_deg} is off grid (doa_min={self.doa_min}, "
                f"doa_step={self.doa_step}): {file_path}"
            )
        doa_label = (doa_deg - self.doa_min) // self.doa_step

        return {
            "x": x,
            "mod_label": torch.tensor(mod_label, dtype=torch.long),
            "doa_label": torch.tensor(doa_label, dtype=torch.long),
            "doa_deg": doa_deg,
            "snr_db": snr_db,
            "path": str(file_path),
        }
