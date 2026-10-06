"""Run with the project venv: python3 store/test_train_mamba_cnn.py.

Only tiny batches are trained. CUDA and real-data checks explicitly report skips
when unavailable; all checkpoint and CSV artifacts use temporary directories.
"""

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
import csv
import io
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.modanet_dataset_loader import MoDANetDatasetLoader
from models.mamba_cnn import DOAMambaNet
from train import train_mamba_cnn as training


METRICS_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_accuracy", "train_rmse_deg",
    "eval_loss", "eval_accuracy", "eval_rmse_deg", "elapsed_seconds",
]
SNR_FIELDS = ["snr_db", "n_samples", "accuracy", "rmse_deg"]


def assert_raises(error_type, action, message=None):
    try:
        action()
    except error_type as exc:
        if message is not None:
            assert message in str(exc), f"Expected {message!r} in error: {exc}"
    else:
        raise AssertionError(f"Expected {error_type}")


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


class LengthOnlyDataset(Dataset):
    """Fail if splitting or constructing a loader reads signal contents."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise AssertionError("Splitting/indexing must not preload dataset contents")


class Samples(Dataset):
    """A small test fixture deliberately omitting modulation metadata."""

    def __init__(self, x, target, snr=None):
        self.x, self.target, self.snr = x, target, snr

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {"x": self.x[index], "doa_label": self.target[index]}
        if self.snr is not None:
            sample["snr_db"] = self.snr[index]
        return sample


class LogitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []
        self.latest_logits = None

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), x.dtype))
        self.latest_logits = x * self.scale
        return self.latest_logits


def test_config(cfg):
    original = deepcopy(cfg)
    assert training.validate_training_config(cfg) == cfg["training"]
    training.validate_loader_config(cfg["loader"])
    assert cfg == original
    assert_raises((KeyError, ValueError), lambda: training.validate_training_config({}))
    for key in cfg["training"]:
        invalid = deepcopy(cfg)
        del invalid["training"][key]
        assert_raises((KeyError, ValueError), lambda: training.validate_training_config(invalid))
    invalid_values = {
        "seed": [-1, True, 1.5, "42"],
        "train_fraction": [0, 1, -0.1, float("nan"), float("inf"), True, "0.8"],
        "batch_size": [0, -2, True, 1.5],
        "epochs": [0, -2, True, 1.5],
        "learning_rate": [0, -1, float("nan"), float("inf"), True, "0.1"],
        "weight_decay": [-1, float("nan"), float("inf"), True, "0"],
        "grad_clip_norm": [0, -1, float("nan"), float("inf"), True, "1"],
        "output_dir": ["", "   ", None, True, 123],
    }
    for key, values in invalid_values.items():
        for value in values:
            invalid = deepcopy(cfg)
            invalid["training"][key] = value
            assert_raises((ValueError, TypeError), lambda: training.validate_training_config(invalid))
    with TemporaryDirectory() as directory:
        valid = deepcopy(cfg)
        destination = Path(directory) / "validation-must-not-create-this"
        valid["training"]["output_dir"] = destination
        training.validate_training_config(valid)
        assert not destination.exists()
    loader_cfg = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    training.validate_loader_config(loader_cfg)
    assert_raises(ValueError, lambda: training.validate_loader_config(
        dict(loader_cfg, persistent_workers=True)))
    for key, invalid in (("num_workers", -1), ("num_workers", True),
                         ("pin_memory", "false"), ("persistent_workers", "true")):
        assert_raises((ValueError, TypeError), lambda: training.validate_loader_config(
            dict(loader_cfg, **{key: invalid})))
    print("PASS: real config, missing/invalid fields, loader validation and side-effect-free validation")


def test_seed_split_and_loaders(cfg):
    training.set_seed(42)
    first_rng = (random.random(), np.random.rand(), torch.rand(3))
    training.set_seed(42)
    second_rng = (random.random(), np.random.rand(), torch.rand(3))
    assert_tree_equal(first_rng, second_rng)
    dataset = LengthOnlyDataset(100)
    train, evaluation = training.split_dataset(dataset, 0.8, 42)
    train_again, eval_again = training.split_dataset(dataset, 0.8, 42)
    different, _ = training.split_dataset(dataset, 0.8, 43)
    assert len(train) == 80 and len(evaluation) == 20
    assert train.indices == train_again.indices and evaluation.indices == eval_again.indices
    assert train.indices != different.indices
    assert set(train.indices).isdisjoint(evaluation.indices)
    assert set(train.indices) | set(evaluation.indices) == set(range(100))
    expected = torch.randperm(100, generator=torch.Generator().manual_seed(42)).tolist()
    assert train.indices + evaluation.indices == expected
    rounded, _ = training.split_dataset(LengthOnlyDataset(7), 0.5, 42)
    assert len(rounded) == 4, "Split must round the training size"
    for size in (0, 1):
        assert_raises(ValueError, lambda: training.split_dataset(LengthOnlyDataset(size), 0.8, 42))
    for fraction in (0.001, 0.999):
        assert_raises(ValueError, lambda: training.split_dataset(LengthOnlyDataset(2), fraction, 42))
    train_cfg = dict(cfg["training"], batch_size=7)
    loader_cfg = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    train_loader, eval_loader = training.build_dataloaders(dataset, train_cfg, loader_cfg)
    other_train, _ = training.build_dataloaders(dataset, train_cfg, loader_cfg)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(eval_loader.sampler, SequentialSampler)
    assert train_loader.batch_size == eval_loader.batch_size == 7
    assert not train_loader.drop_last and not eval_loader.drop_last
    assert train_loader.num_workers == eval_loader.num_workers == 0
    assert not train_loader.pin_memory and not train_loader.persistent_workers
    assert train_loader.dataset.indices == train.indices
    assert eval_loader.dataset.indices == evaluation.indices
    assert list(iter(train_loader.sampler)) == list(iter(other_train.sampler))
    assert list(iter(eval_loader.sampler)) == list(range(20))
    assert train_loader.generator is not None
    assert train_loader.generator is not other_train.generator
    print("PASS: RNG seeds, global randperm split, rounding, no eager reads and loader policy")


def test_degree_mapping(dataset_cfg):
    mapped = training.class_indices_to_degrees(torch.tensor([0, 60, 120]), dataset_cfg)
    torch.testing.assert_close(mapped, torch.tensor([-60., 0., 60.], dtype=torch.float64))
    custom = dict(dataset_cfg, doa_min=-30, doa_step=3, doa_max=30, num_doa_classes=21)
    actual = training.class_indices_to_degrees(torch.tensor([0, 10, 20]), custom)
    torch.testing.assert_close(actual, torch.tensor([-30., 0., 30.], dtype=torch.float64))
    print("PASS: config-driven degree mapping, including doa_step=3")


def test_metrics_and_modes(dataset_cfg):
    target = torch.tensor([0, 60, 120], dtype=torch.int32)
    logits = torch.full((3, dataset_cfg["num_doa_classes"]), -1., dtype=torch.float64)
    logits[torch.arange(3), torch.tensor([1, 60, 118])] = torch.tensor([1., 2., 5.], dtype=torch.float64)
    snr = torch.tensor([-10, -10, 20])
    eval_loader = DataLoader(Samples(logits, target, snr), batch_size=2)
    train_loader = DataLoader(Samples(logits, target), batch_size=2)
    model = LogitsModel()
    criterion = nn.CrossEntropyLoss()
    loss_inputs = []

    def check_loss_input(module, inputs):
        values, labels = inputs
        assert values is model.latest_logits, "CrossEntropyLoss did not receive raw model logits"
        assert values.dtype == torch.float32
        assert labels.dtype == torch.long and labels.ndim == 1
        loss_inputs.append(values.detach().clone())

    hook = criterion.register_forward_pre_hook(check_loss_input)
    expected_loss = nn.CrossEntropyLoss()(logits.float(), target.long()).item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    model.eval()
    try:
        train_result = training.train_one_epoch(
            model, train_loader, criterion, optimizer, torch.device("cpu"), dataset_cfg, 1.0)
        assert all(mode and grads and dtype == torch.float32 for mode, grads, dtype in model.calls)
        assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
        model.zero_grad(set_to_none=True)
        before = deepcopy(model.state_dict())
        model.calls.clear()
        eval_result = training.evaluate(model, eval_loader, criterion, torch.device("cpu"), dataset_cfg)
    finally:
        hook.remove()
    assert all(not mode and not grads for mode, grads, _ in model.calls)
    assert_tree_equal(model.state_dict(), before)
    assert all(parameter.grad is None for parameter in model.parameters())
    for result in (train_result, eval_result):
        assert result["n_samples"] == 3
        assert math.isclose(result["loss"], expected_loss, rel_tol=1e-6)
        assert math.isclose(result["accuracy"], 1 / 3, rel_tol=1e-12)
        assert math.isclose(result["rmse_deg"], math.sqrt(5 / 3), rel_tol=1e-12)
    groups = eval_result["by_snr"]
    assert set(groups) == {-10, 20}
    assert groups[-10]["n_samples"] == 2 and groups[20]["n_samples"] == 1
    assert groups[-10]["accuracy"] == 0.5 and groups[20]["accuracy"] == 0
    assert math.isclose(groups[-10]["rmse_deg"], math.sqrt(0.5), rel_tol=1e-12)
    assert groups[20]["rmse_deg"] == 2
    assert len(loss_inputs) == 4
    print("PASS: raw CE logits/long targets, no mod_label, uneven-batch loss/accuracy/RMSE, "
          "per-SNR metrics and train/eval/gradient modes")


def make_history():
    return [dict(zip(METRICS_FIELDS, [1, 0.001, 4.2, 0.25, 25.0, 4.0, 0.3, 22.0, 1.5])),
            dict(zip(METRICS_FIELDS, [2, 0.001, 3.9, 0.35, 20.0, 3.8, 0.4, 18.0, 1.25]))]


def test_csv():
    history = make_history()
    groups = {snr: {"n_samples": 5, "accuracy": 0.4, "rmse_deg": 2.5}
              for snr in (20, 2, -9, 10, -10, 0)}
    with TemporaryDirectory() as directory:
        metrics_path = Path(directory) / "metrics.csv"
        snr_path = Path(directory) / "eval_by_snr.csv"
        training.write_metrics_csv(metrics_path, history[:1])
        training.write_metrics_csv(metrics_path, history)
        with metrics_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == METRICS_FIELDS
            rows = list(reader)
        assert len(rows) == len(history)
        for actual, expected in zip(rows, history):
            for key, value in expected.items():
                assert float(actual[key]) == value
        training.write_snr_metrics_csv(snr_path, groups)
        with snr_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == SNR_FIELDS
            rows = list(reader)
        assert [int(row["snr_db"]) for row in rows] == [-10, -9, 0, 2, 10, 20]
        for row in rows:
            assert int(row["n_samples"]) == 5
            assert float(row["accuracy"]) == 0.4
            assert float(row["rmse_deg"]) == 2.5
    print("PASS: incremental history CSV, exact headers/values and numeric SNR ordering")


def test_epoch_orchestration(cfg):
    """Exercise real checkpoint/CSV orchestration with deterministic CPU stand-ins."""
    class EpochModel(nn.Module):
        def __init__(self, dataset_config, model_config):
            super().__init__()
            self.signal_adapter = nn.Linear(1, 1, bias=False)
            self.doa_head = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.signal_adapter.weight)
            nn.init.zeros_(self.doa_head.weight)

        def to(self, *, device, dtype):
            assert device.type == "cuda" and dtype == torch.float32
            return self  # All orchestration math remains on CPU in this test.

    trained_epochs, evaluated_epochs, csv_epochs = [], [], []
    original_csv_writer = training.write_metrics_csv

    def fake_train(model, loader, criterion, optimizer, device, dataset_config,
                   grad_clip_norm, *, pin_memory=False):
        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert isinstance(optimizer, torch.optim.Adam)
        with torch.no_grad():
            model.signal_adapter.weight.add_(1)
        epoch = int(model.signal_adapter.weight.item())
        trained_epochs.append(epoch)
        return {"loss": 1., "accuracy": epoch / 10, "rmse_deg": 5.,
                "n_samples": len(loader.dataset)}

    def fake_evaluate(model, loader, criterion, device, dataset_config, *, pin_memory=False):
        epoch = int(model.signal_adapter.weight.item())
        evaluated_epochs.append(epoch)
        rmse = {1: 2., 2: 4., 3: 3.}[epoch]
        return {"loss": 1., "accuracy": epoch / 10, "rmse_deg": rmse,
                "n_samples": len(loader.dataset),
                "by_snr": {20: {"n_samples": 1, "accuracy": 0., "rmse_deg": rmse},
                           -10: {"n_samples": 1, "accuracy": 0., "rmse_deg": rmse}}}

    def record_csv(path, history):
        csv_epochs.append([row["epoch"] for row in history])
        original_csv_writer(path, history)

    with TemporaryDirectory() as directory:
        local_cfg = deepcopy(cfg)
        local_cfg["training"].update(epochs=2, batch_size=2, output_dir=str(Path(directory) / "first"))
        local_cfg["loader"].update(num_workers=0, persistent_workers=False)
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert_raises(RuntimeError, lambda: training.run_training(local_cfg), "CUDA is required")
        assert not Path(local_cfg["training"]["output_dir"]).exists()
        with ExitStack() as stack:
            for target, replacement in (
                ("MoDANetDatasetLoader", lambda dataset_config: LengthOnlyDataset(10)),
                ("DOAMambaNet", EpochModel), ("train_one_epoch", fake_train),
                ("evaluate", fake_evaluate), ("write_metrics_csv", record_csv),
            ):
                stack.enter_context(patch.object(training, target, replacement))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="CPU orchestration fixture"))
            stack.enter_context(patch.object(torch.cuda, "get_rng_state", return_value=None))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = training.run_training(local_cfg)
            assert result["rmse_deg"] == 2.
            assert trained_epochs == [1, 2] and evaluated_epochs == [1, 2, 1]
            assert csv_epochs == [[1], [1, 2]], "History CSV must be updated each epoch"
            first_dir = Path(local_cfg["training"]["output_dir"])
            first_last_path = first_dir / "checkpoints" / "last.pt"
            first_last = torch.load(first_last_path, map_location="cpu", weights_only=True)
            first_best = torch.load(first_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
            assert first_last["epoch"] == 2 and first_best["epoch"] == 1
            assert first_last["best_eval_rmse_deg"] == 2.
            assert first_last["best_checkpoint"]["epoch"] == 1
            assert "best_checkpoint" not in first_last["best_checkpoint"], "Best snapshots must not recurse"
            assert_raises(FileExistsError, lambda: training.run_training(local_cfg))

            # Move only last.pt to another output directory and extend by one epoch.
            moved = deepcopy(local_cfg)
            moved["training"].update(epochs=3, output_dir=str(Path(directory) / "moved"))
            result = training.run_training(moved, resume=first_last_path)
            assert result["rmse_deg"] == 2.
            assert trained_epochs == [1, 2, 3] and evaluated_epochs == [1, 2, 1, 3, 1]
            moved_dir = Path(moved["training"]["output_dir"])
            moved_last_path = moved_dir / "checkpoints" / "last.pt"
            moved_last = torch.load(moved_last_path, map_location="cpu", weights_only=True)
            moved_best = torch.load(moved_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
            assert moved_last["epoch"] == 3 and moved_best["epoch"] == 1
            assert moved_last["history"][:2] == first_last["history"]
            assert [row["epoch"] for row in moved_last["history"]] == [1, 2, 3]
            assert_tree_equal(moved_best["model_state_dict"], first_best["model_state_dict"])
            with (moved_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
                assert [int(row["epoch"]) for row in csv.DictReader(file)] == [1, 2, 3]

            # Completed runs still reconstruct outputs and evaluate the historical best.
            completed = deepcopy(moved)
            completed["training"]["output_dir"] = str(Path(directory) / "completed")
            training.run_training(completed, resume=moved_last_path)
            assert trained_epochs == [1, 2, 3] and evaluated_epochs[-1] == 1
            with (Path(completed["training"]["output_dir"]) / "eval_by_snr.csv").open(
                    newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            assert [int(row["snr_db"]) for row in rows] == [-10, 20]
            assert all(float(row["rmse_deg"]) == 2. for row in rows)

            # Resume best.pt itself and ensure its history cannot alias the growing history.
            from_best = deepcopy(local_cfg)
            from_best["training"]["output_dir"] = str(Path(directory) / "from-best")
            result = training.run_training(from_best, resume=first_dir / "checkpoints" / "best.pt")
            assert result["rmse_deg"] == 2.
            resumed_last_path = Path(from_best["training"]["output_dir"]) / "checkpoints" / "last.pt"
            resumed_last = torch.load(resumed_last_path, map_location="cpu", weights_only=True)
            assert resumed_last["epoch"] == 2
            assert resumed_last["best_checkpoint"]["history"] == first_best["history"]
            from_best["training"].update(epochs=3, output_dir=str(Path(directory) / "from-best-again"))
            result = training.run_training(from_best, resume=resumed_last_path)
            assert result["rmse_deg"] == 2.
    print("PASS: CUDA requirement, per-epoch last/best/CSV, best selected by RMSE, next-epoch resume, "
          "moved-output best preservation and completed-run final evaluation")


def assert_finite_metrics(result, expected_count):
    assert result["n_samples"] == expected_count
    for key in ("loss", "accuracy", "rmse_deg"):
        assert math.isfinite(result[key]), f"Non-finite {key}"
    assert result["loss"] >= 0 and 0 <= result["accuracy"] <= 1 and result["rmse_deg"] >= 0


def checked_optimizer_step(model, loader, cfg):
    """Use the actual epoch helper and observe clipping before Adam updates."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    events = []
    observed_logits = []
    clip = torch.nn.utils.clip_grad_norm_
    step = optimizer.step

    def checked_clip(parameters, max_norm, **kwargs):
        events.append("clip")
        assert max_norm == cfg["training"]["grad_clip_norm"]
        parameters = list(parameters)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        return clip(parameters, max_norm=max_norm, **kwargs)

    def checked_step(*args, **kwargs):
        assert events[-1] == "clip", "Adam step must follow clipping"
        grad_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]))
        assert grad_norm <= cfg["training"]["grad_clip_norm"] + 1e-5
        events.append("step")
        return step(*args, **kwargs)

    def observe_output(module, inputs, output):
        assert module.training and torch.is_grad_enabled()
        assert inputs[0].dtype == torch.float32
        assert output.shape == (inputs[0].shape[0], cfg["dataset"]["num_doa_classes"])
        assert torch.isfinite(output).all()
        observed_logits.append(output)

    def observe_loss(module, inputs):
        logits, target = inputs
        assert logits is observed_logits[-1]
        assert target.dtype == torch.long and target.shape == (logits.shape[0],)

    hooks = [model.register_forward_hook(observe_output), criterion.register_forward_pre_hook(observe_loss)]
    model.eval()
    try:
        with patch.object(torch.nn.utils, "clip_grad_norm_", checked_clip), \
                patch.object(optimizer, "step", checked_step):
            result = training.train_one_epoch(
                model, loader, criterion, optimizer, torch.device("cuda"), cfg["dataset"],
                cfg["training"]["grad_clip_norm"])
    finally:
        for hook in hooks:
            hook.remove()
    assert events == ["clip", "step"] * len(loader)
    assert_finite_metrics(result, len(loader.dataset))
    assert any(not torch.equal(parameter.detach(), before[name])
               for name, parameter in model.named_parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    return optimizer, result


def test_checkpoint(model, optimizer, cfg):
    history = make_history()
    generator = torch.Generator().manual_seed(43)
    torch.randperm(10, generator=generator)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "last.pt"
        training.save_checkpoint(path, model, optimizer, 2, 18.0, history,
                                 cfg["dataset"], cfg["model"], cfg["training"],
                                 train_generator=generator)
        expected_next_order = torch.randperm(10, generator=generator)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_eval_rmse_deg",
                    "history", "dataset_config", "model_config", "training_config"}
        assert required <= saved.keys()
        assert {name.split(".")[0] for name in saved["model_state_dict"]} == {"signal_adapter", "doa_head"}
        restored = DOAMambaNet(cfg["dataset"], cfg["model"]).to(next(model.parameters()).device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.5)
        restored_generator = torch.Generator().manual_seed(999)
        checkpoint = training.load_checkpoint(
            path, restored, restored_optimizer, cfg["dataset"], cfg["model"], cfg["training"],
            train_generator=restored_generator)
        assert checkpoint["epoch"] == 2 and checkpoint["best_eval_rmse_deg"] == 18.0
        assert checkpoint["history"] == history
        assert_tree_equal(restored.state_dict(), model.state_dict())
        assert_tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        torch.testing.assert_close(torch.randperm(10, generator=restored_generator), expected_next_order)

        def load_with(dataset_cfg=None, model_cfg=None, train_cfg=None, checkpoint_path=path):
            return training.load_checkpoint(
                checkpoint_path, restored, restored_optimizer,
                cfg["dataset"] if dataset_cfg is None else dataset_cfg,
                cfg["model"] if model_cfg is None else model_cfg,
                cfg["training"] if train_cfg is None else train_cfg)

        for field in ("snapshots", "antennas", "iq_channels", "num_doa_classes", "doa_min",
                      "doa_max", "doa_step", "num_samples"):
            invalid = dict(cfg["dataset"], **{field: cfg["dataset"][field] + 1})
            assert_raises(ValueError, lambda: load_with(dataset_cfg=invalid))
        for section, field in (("mamba", "d_model"), ("signal_adapter", "delta_init_std"),
                               ("doa_head", "feature_dim")):
            invalid = deepcopy(cfg["model"])
            invalid[section][field] *= 2
            assert_raises(ValueError, lambda: load_with(model_cfg=invalid))
        for field in ("seed", "train_fraction", "learning_rate", "weight_decay", "grad_clip_norm"):
            invalid = dict(cfg["training"], **{field: cfg["training"][field] + 0.1})
            assert_raises(ValueError, lambda: load_with(train_cfg=invalid))
        load_with(dataset_cfg=dict(cfg["dataset"], root="/different/dataset/mount"),
                  train_cfg=dict(cfg["training"], epochs=20, batch_size=2, output_dir=directory))
        corrupted = deepcopy(saved)
        del corrupted["model_state_dict"][next(iter(corrupted["model_state_dict"]))]
        corrupt_path = Path(directory) / "missing-parameter.pt"
        torch.save(corrupted, corrupt_path)
        assert_raises((ValueError, RuntimeError), lambda: load_with(checkpoint_path=corrupt_path))
        corrupted = deepcopy(saved)
        corrupted["model_state_dict"]["legacy.weight"] = torch.zeros(1)
        torch.save(corrupted, corrupt_path)
        assert_raises((ValueError, RuntimeError), lambda: load_with(checkpoint_path=corrupt_path))
    print("PASS: checkpoint model/optimizer/epoch/best/history/generator round trip, namespaces, "
          "strict state loading and config compatibility")


def test_cuda(cfg, real_dataset):
    training.set_seed(cfg["training"]["seed"])
    dataset_cfg = cfg["dataset"]
    x = torch.randn(2, dataset_cfg["iq_channels"], dataset_cfg["antennas"], dataset_cfg["snapshots"])
    samples = Samples(x, torch.tensor([0, dataset_cfg["num_doa_classes"] - 1]), torch.tensor([-10, 20]))
    loader = DataLoader(samples, batch_size=2, num_workers=0)
    model = DOAMambaNet(dataset_cfg, cfg["model"]).cuda()
    optimizer, _ = checked_optimizer_step(model, loader, cfg)
    optimizer.zero_grad(set_to_none=True)
    before = deepcopy(model.state_dict())
    eval_result = training.evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cuda"), dataset_cfg)
    assert_finite_metrics(eval_result, 2)
    assert not model.training
    assert_tree_equal(model.state_dict(), before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert set(eval_result["by_snr"]) == {-10, 20}
    test_checkpoint(model, optimizer, cfg)
    print("PASS: CUDA full-model CE/Adam step, finite gradients, parameter updates and immutable evaluation")
    if real_dataset is not None:
        real_loader = DataLoader(Subset(real_dataset, [0, len(real_dataset) - 1]), batch_size=2, num_workers=0)
        _, result = checked_optimizer_step(model, real_loader, cfg)
        print(f"PASS: real .mat -> loader -> model -> CE -> backward -> clipping -> Adam; "
              f"samples=2, loss={result['loss']:.6f}")
    else:
        print("SKIP: real-data optimizer step; external dataset unavailable")
    torch.cuda.synchronize()
    print(f"PASS: CUDA smoke on {torch.cuda.get_device_name()}")


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    original = deepcopy(cfg)
    test_config(cfg)
    test_seed_split_and_loaders(cfg)
    test_degree_mapping(cfg["dataset"])
    test_metrics_and_modes(cfg["dataset"])
    test_csv()
    test_epoch_orchestration(cfg)
    real_dataset = None
    if Path(cfg["dataset"]["root"]).expanduser().is_dir():
        real_dataset = MoDANetDatasetLoader(cfg["dataset"])
        with patch("data.modanet_dataset_loader.loadmat", side_effect=AssertionError("Split read a .mat file")):
            train, evaluation = training.split_dataset(real_dataset, cfg["training"]["train_fraction"],
                                                        cfg["training"]["seed"])
        assert len(real_dataset) == 450120
        assert len(train) == 360096 and len(evaluation) == 90024
        assert set(train.indices).isdisjoint(evaluation.indices)
        assert set(train.indices) | set(evaluation.indices) == set(range(len(real_dataset)))
        print("PASS: real dataset indexed lazily; 450120 samples -> 360096 train / 90024 eval")
    else:
        print(f"SKIP: real dataset unavailable: {cfg['dataset']['root']}")
    if torch.cuda.is_available():
        test_cuda(cfg, real_dataset)
    else:
        model = DOAMambaNet(cfg["dataset"], cfg["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
        test_checkpoint(model, optimizer, cfg)
        print("SKIP: CUDA unavailable; full-model training/evaluation and real optimizer step not run")
    assert cfg == original, "Tests or training helpers mutated caller configuration"
    print("All available Task 3 training-pipeline tests passed; no full training run was started.")


if __name__ == "__main__":
    main()
