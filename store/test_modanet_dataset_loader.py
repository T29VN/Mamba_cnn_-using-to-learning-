"""Run directly with python3 store/test_modanet_dataset_loader.py (no pytest)."""

from numbers import Integral
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.modanet_dataset_loader import MoDANetDatasetLoader


def assert_raises(error_type, action, *message_parts):
    try:
        action()
    except error_type as exc:
        for part in message_parts:
            assert str(part) in str(exc), f"Missing {part!r} in error: {exc}"
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def check_sample(sample, cfg):
    assert set(sample) == {"x", "mod_label", "doa_label", "doa_deg", "snr_db", "path"}
    assert isinstance(sample["x"], torch.Tensor)
    assert sample["x"].shape == (cfg["iq_channels"], cfg["antennas"], cfg["snapshots"])
    assert sample["x"].dtype == torch.float32
    assert sample["x"].is_contiguous()
    for key in ("mod_label", "doa_label"):
        assert isinstance(sample[key], torch.Tensor)
        assert sample[key].dtype == torch.long
        assert sample[key].ndim == 0
    assert isinstance(sample["doa_deg"], Integral)
    assert isinstance(sample["snr_db"], Integral)
    assert isinstance(sample["path"], str)


def test_real_dataset(cfg):
    with patch("data.modanet_dataset_loader.loadmat", wraps=loadmat) as reader:
        dataset = MoDANetDatasetLoader(dataset_config=cfg)
        reader.assert_not_called()
    assert len(dataset) == cfg["num_samples"]
    assert dataset.mod_to_index == {name: idx for idx, name in enumerate(cfg["modulations"])}
    check_sample(dataset[0], cfg)

    # Select by metadata, independent of the order returned by sorted().
    needed_angles = {cfg["doa_min"], 0, cfg["doa_max"]}
    selected = {}
    for index, path in enumerate(dataset.files):
        if path.parents[2].name == "SNR+00dB" and path.parents[1].name == "16APSK":
            angle = int(path.parent.name)
            if angle in needed_angles:
                selected.setdefault(angle, index)
                if selected.keys() == needed_angles:
                    break
    assert selected.keys() == needed_angles, f"Missing real DOA samples: {needed_angles - selected.keys()}"
    for angle, index in selected.items():
        sample = dataset[index]
        check_sample(sample, cfg)
        assert sample["mod_label"].item() == cfg["modulations"].index("16APSK") == 0
        assert sample["snr_db"] == 0
        assert sample["doa_deg"] == angle
        assert sample["doa_label"].item() == (angle - cfg["doa_min"]) // cfg["doa_step"]
        if angle == cfg["doa_min"]:
            assert sample["doa_label"].item() == 0
        if angle == cfg["doa_max"]:
            assert sample["doa_label"].item() == cfg["num_doa_classes"] - 1

    batch_size = 4
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    batch = next(iter(loader))
    assert batch["x"].shape == (batch_size, cfg["iq_channels"], cfg["antennas"], cfg["snapshots"])
    assert batch["mod_label"].shape == (batch_size,)
    assert batch["doa_label"].shape == (batch_size,)
    assert batch["x"].dtype == torch.float32
    assert batch["mod_label"].dtype == batch["doa_label"].dtype == torch.long
    print(f"PASS: {len(dataset)} real samples indexed; sample types, mappings and batch {tuple(batch['x'].shape)}")


def test_validation_and_custom_config(dataset_cfg):
    # Small temporary fixtures exercise configuration changes and corrupt samples.
    with TemporaryDirectory(prefix="modanet_loader_test_") as temp_dir:
        root = Path(temp_dir)
        cfg = dict(
            dataset_cfg, root=str(root), mat_variable="signal", num_samples=1,
            snapshots=4, antennas=3, iq_channels=2,
            modulations=list(reversed(dataset_cfg["modulations"])),
            doa_min=-6, doa_max=6, doa_step=3, num_doa_classes=5,
        )
        path = root / "SNR-07dB" / cfg["modulations"][0] / "+03" / "sample.mat"
        path.parent.mkdir(parents=True)
        original = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
        savemat(path, {cfg["mat_variable"]: original})
        with patch("data.modanet_dataset_loader.loadmat", wraps=loadmat) as reader:
            dataset = MoDANetDatasetLoader(dataset_config=cfg)
            reader.assert_not_called()
            sample = dataset[0]
            reader.assert_called_once_with(path)
        check_sample(sample, cfg)
        np.testing.assert_array_equal(sample["x"].numpy(), original.transpose(2, 1, 0).astype(np.float32))
        assert sample["mod_label"].item() == 0
        assert sample["doa_label"].item() == 3
        assert sample["doa_deg"] == 3
        assert sample["snr_db"] == -7

        for key in cfg:
            incomplete = dict(cfg)
            del incomplete[key]
            assert_raises(KeyError, lambda: MoDANetDatasetLoader(incomplete), key)
        assert_raises(ValueError, lambda: MoDANetDatasetLoader(None), "dataset_config")
        for bad_root in (root / "missing", path):
            assert_raises(FileNotFoundError, lambda: MoDANetDatasetLoader(dict(cfg, root=bad_root)), bad_root)
        invalid_configs = [
            ({"root": ""}, "root"),
            ({"mat_variable": ""}, "mat_variable"),
            ({"modulations": []}, "modulations"),
            ({"modulations": ["same", "same"]}, "duplicates"),
            ({"modulations": [None]}, "modulations"),
            ({"num_mod_classes": cfg["num_mod_classes"] + 1}, "num_mod_classes"),
            ({"doa_step": 0}, "doa_step"),
            ({"doa_step": -1}, "doa_step"),
            ({"doa_min": 7}, "doa_max"),
            ({"doa_step": 5}, "divisible"),
            ({"num_doa_classes": 4}, "num_doa_classes"),
            ({"snapshots": 0}, "snapshots"),
            ({"antennas": 1.5}, "antennas"),
            ({"iq_channels": True}, "iq_channels"),
        ]
        for changes, message in invalid_configs:
            assert_raises(ValueError, lambda: MoDANetDatasetLoader(dict(cfg, **changes)), message)
        assert_raises(
            ValueError, lambda: MoDANetDatasetLoader(dict(cfg, num_samples=2)),
            "expected 2", "actual 1", root,
        )

        savemat(path, {"wrong_variable": original})
        dataset = MoDANetDatasetLoader(cfg)
        assert_raises(KeyError, lambda: dataset[0], cfg["mat_variable"], path)
        savemat(path, {cfg["mat_variable"]: original[:1]})
        dataset = MoDANetDatasetLoader(cfg)
        assert_raises(ValueError, lambda: dataset[0], "actual (1, 3, 2)", "expected (4, 3, 2)", path)
        savemat(path, {cfg["mat_variable"]: original})

        invalid_paths = [
            (("SNRbaddB", cfg["modulations"][0], "+03"), "SNR"),
            (("SNR-07dB", "unknown_modulation", "+03"), "modulation"),
            (("SNR-07dB", cfg["modulations"][0], "bad_angle"), "DOA"),
            (("SNR-07dB", cfg["modulations"][0], "+09"), "outside"),
            (("SNR-07dB", cfg["modulations"][0], "+02"), "off grid"),
        ]
        for directories, message in invalid_paths:
            bad_path = root.joinpath(*directories, "sample.mat")
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            path.rename(bad_path)
            dataset = MoDANetDatasetLoader(cfg)
            assert_raises(ValueError, lambda: dataset[0], message, bad_path)
            bad_path.rename(path)
    print("PASS: custom config, lazy loading, unchanged values and validation errors")


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    dataset_cfg = cfg["dataset"]
    test_validation_and_custom_config(dataset_cfg)
    test_real_dataset(dataset_cfg)
    print("All MoDANetDatasetLoader tests passed.")


if __name__ == "__main__":
    main()
