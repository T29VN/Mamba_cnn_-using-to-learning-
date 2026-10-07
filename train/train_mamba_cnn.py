"""Single-GPU FP32 DOA training, evaluation and resumable checkpoints."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import shutil
import sys
from tempfile import NamedTemporaryFile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_DIR = PROJECT_ROOT / "store" / "run_config"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.modanet_dataset_loader import MoDANetDatasetLoader
from models.mamba_cnn import DOAMambaNet


METRICS_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_accuracy", "train_rmse_deg",
    "eval_loss", "eval_accuracy", "eval_rmse_deg", "elapsed_seconds",
]
SNR_FIELDS = ["snr_db", "n_samples", "accuracy", "rmse_deg"]
DATASET_COMPATIBILITY_FIELDS = (
    "snapshots", "antennas", "iq_channels", "num_doa_classes",
    "doa_min", "doa_max", "doa_step", "num_samples",
)


def _required(config, key, section):
    if key not in config:
        raise KeyError(f"Missing config key: {section}.{key}")
    return config[key]


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _number(value, name, minimum, *, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or (not inclusive and value == minimum):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {relation} {minimum}, got {value!r}")


def validate_training_config(cfg):
    """Validate without creating directories or changing the caller's config."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a dictionary")
    training = _required(cfg, "training", "config")
    if not isinstance(training, dict):
        raise ValueError("training must be a dictionary")
    for key in ("seed", "train_fraction", "batch_size", "epochs", "learning_rate",
                "weight_decay", "grad_clip_norm", "output_dir"):
        _required(training, key, "training")
    _integer(training["seed"], "training.seed", 0)
    for key in ("batch_size", "epochs"):
        _integer(training[key], f"training.{key}", 1)
    _number(training["train_fraction"], "training.train_fraction", 0)
    if training["train_fraction"] >= 1:
        raise ValueError("training.train_fraction must be < 1")
    for key in ("learning_rate", "grad_clip_norm"):
        _number(training[key], f"training.{key}", 0)
    _number(training["weight_decay"], "training.weight_decay", 0, inclusive=True)
    output = training["output_dir"]
    if not isinstance(output, (str, os.PathLike)) or not str(output).strip():
        raise ValueError("training.output_dir must be a non-empty string/path")
    return training


def validate_loader_config(loader_config):
    if not isinstance(loader_config, dict):
        raise ValueError("loader must be a dictionary")
    for key in ("num_workers", "pin_memory", "persistent_workers"):
        _required(loader_config, key, "loader")
    _integer(loader_config["num_workers"], "loader.num_workers", 0)
    for key in ("pin_memory", "persistent_workers"):
        if not isinstance(loader_config[key], bool):
            raise ValueError(f"loader.{key} must be bool")
    if loader_config["num_workers"] == 0 and loader_config["persistent_workers"]:
        raise ValueError("loader.persistent_workers must be False when num_workers == 0")
    return loader_config


def set_seed(seed):
    _integer(seed, "seed", 0)
    random.seed(seed)
    # NumPy's legacy RNG accepts 32-bit seeds; Torch accepts 64-bit seeds.
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 64))
    torch.cuda.manual_seed_all(seed % (2 ** 64))


def split_dataset(dataset, train_fraction, seed):
    _integer(seed, "seed", 0)
    _number(train_fraction, "train_fraction", 0)
    if train_fraction >= 1:
        raise ValueError("train_fraction must be < 1")
    n_train = round(train_fraction * len(dataset))
    if not 0 < n_train < len(dataset):
        raise ValueError("Split must contain non-empty train and eval subsets")
    generator = torch.Generator().manual_seed(seed % (2 ** 64))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, indices[:n_train]), Subset(dataset, indices[n_train:])


def build_dataloaders(dataset, training_config, loader_config):
    validate_training_config({"training": training_config})
    validate_loader_config(loader_config)
    seed = training_config["seed"]
    train_set, eval_set = split_dataset(dataset, training_config["train_fraction"], seed)
    options = dict(
        batch_size=training_config["batch_size"], drop_last=False,
        num_workers=loader_config["num_workers"], pin_memory=loader_config["pin_memory"],
        persistent_workers=loader_config["persistent_workers"],
    )
    train_generator = torch.Generator().manual_seed((seed + 1) % (2 ** 64))
    eval_generator = torch.Generator().manual_seed((seed + 2) % (2 ** 64))
    return (
        DataLoader(train_set, shuffle=True, generator=train_generator, **options),
        DataLoader(eval_set, shuffle=False, generator=eval_generator, **options),
    )


def class_indices_to_degrees(class_indices, dataset_config):
    return dataset_config["doa_min"] + class_indices.to(torch.float64) * dataset_config["doa_step"]


class _Metrics:
    """Accumulate sample totals, including a partial final batch."""

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.n_samples = 0
        self.total_loss = 0.0
        self.correct = 0
        self.squared_error = 0.0
        self.by_snr = {}

    def update(self, logits, target, loss, snr_db=None):
        predicted = logits.detach().argmax(dim=1).cpu()
        target = target.detach().cpu()
        correct = predicted.eq(target)
        errors = (class_indices_to_degrees(predicted, self.dataset_config)
                  - class_indices_to_degrees(target, self.dataset_config)).square()
        count = target.numel()
        self.n_samples += count
        self.total_loss += loss.item() * count
        self.correct += correct.sum().item()
        self.squared_error += errors.sum().item()
        if snr_db is not None:
            snr_db = torch.as_tensor(snr_db).detach().cpu()
            for snr in snr_db.unique().tolist():
                mask = snr_db == snr
                totals = self.by_snr.setdefault(int(snr), [0, 0, 0.0])
                totals[0] += mask.sum().item()
                totals[1] += correct[mask].sum().item()
                totals[2] += errors[mask].sum().item()

    def compute(self, *, include_snr=False):
        if self.n_samples == 0:
            raise ValueError("Cannot compute metrics for an empty loader")
        result = {
            "loss": self.total_loss / self.n_samples,
            "accuracy": self.correct / self.n_samples,
            "rmse_deg": math.sqrt(self.squared_error / self.n_samples),
            "n_samples": self.n_samples,
        }
        if include_snr:
            result["by_snr"] = {
                snr: {"n_samples": count, "accuracy": correct / count,
                      "rmse_deg": math.sqrt(squared / count)}
                for snr, (count, correct, squared) in sorted(self.by_snr.items())
            }
        return result


def _progress(stage, batch_index, loader, loss):
    if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
        print(f"{stage} batch {batch_index}/{len(loader)} | loss={loss.item():.6f}", flush=True)


def train_one_epoch(model, loader, criterion, optimizer, device, dataset_config,
                    grad_clip_norm, *, pin_memory=False):
    model.train()
    metrics = _Metrics(dataset_config)
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=pin_memory)
        target = batch["doa_label"].to(device, dtype=torch.long, non_blocking=pin_memory)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm,
                                 error_if_nonfinite=True)
        optimizer.step()
        metrics.update(logits, target, loss)
        _progress("Train", batch_index, loader, loss)
    return metrics.compute()


def evaluate(model, loader, criterion, device, dataset_config, *, pin_memory=False):
    model.eval()
    metrics = _Metrics(dataset_config)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=pin_memory)
            target = batch["doa_label"].to(device, dtype=torch.long, non_blocking=pin_memory)
            logits = model(x)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite evaluation loss at batch {batch_index}")
            metrics.update(logits, target, loss, batch["snr_db"])
            _progress("Eval", batch_index, loader, loss)
    return metrics.compute(include_snr=True)


def _cpu_snapshot(value):
    """Copy mutable training state so a retained best checkpoint cannot drift."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return deepcopy(value)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, epoch, best_eval_rmse_deg, history,
                    dataset_config, model_config, training_config, *,
                    train_generator=None, best_checkpoint=None):
    numpy_state = np.random.get_state()
    checkpoint = _cpu_snapshot({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_rmse_deg": best_eval_rmse_deg,
        "history": history,
        "dataset_config": dataset_config,
        "model_config": model_config,
        "training_config": training_config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                      numpy_state[3], numpy_state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        "train_generator_state": train_generator.get_state() if train_generator is not None else None,
    })
    if best_checkpoint is not None:
        # The embedded best is standalone, so checkpoint nesting never grows.
        checkpoint["best_checkpoint"] = {
            key: value for key, value in _cpu_snapshot(best_checkpoint).items()
            if key != "best_checkpoint"
        }
    _atomic_torch_save(path, checkpoint)
    return checkpoint


def _check_compatibility(checkpoint, dataset_config, model_config, training_config):
    for key in DATASET_COMPATIBILITY_FIELDS:
        if checkpoint["dataset_config"].get(key) != dataset_config[key]:
            raise ValueError(f"Checkpoint dataset.{key} does not match current config")
    for section in ("mamba", "signal_adapter", "doa_head"):
        if checkpoint["model_config"].get(section) != model_config[section]:
            raise ValueError(f"Checkpoint model.{section} does not match current config")
    for key in ("seed", "train_fraction", "learning_rate", "weight_decay", "grad_clip_norm"):
        if checkpoint["training_config"].get(key) != training_config[key]:
            raise ValueError(f"Checkpoint training.{key} does not match current config")


def load_checkpoint(path, model, optimizer, dataset_config, model_config, training_config, *,
                    train_generator=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("epoch", "model_state_dict", "optimizer_state_dict", "best_eval_rmse_deg",
                "history", "dataset_config", "model_config", "training_config"):
        _required(checkpoint, key, "checkpoint")
    _check_compatibility(checkpoint, dataset_config, model_config, training_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if train_generator is not None and checkpoint.get("train_generator_state") is not None:
        train_generator.set_state(checkpoint["train_generator_state"])
    rng = checkpoint.get("rng_state")
    if rng is not None:
        random.setstate(rng["python"])
        np.random.set_state((rng["numpy"][0], np.asarray(rng["numpy"][1], dtype=np.uint32),
                             *rng["numpy"][2:]))
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
    return checkpoint


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_metrics_csv(path, history):
    _write_csv(path, METRICS_FIELDS, history)


def write_snr_metrics_csv(path, by_snr):
    rows = (dict(snr_db=int(snr), **values)
            for snr, values in sorted(by_snr.items(), key=lambda item: int(item[0])))
    _write_csv(path, SNR_FIELDS, rows)


def _project_path(path):
    path = Path(path).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def check_run_csv_destinations(run_config_dir):
    """Fail before training if either fixed backup name is occupied."""
    paths = tuple(Path(run_config_dir) / name for name in ("metrics.csv", "eval_by_snr.csv"))
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Backup already exists: {path}; rename the previous CSV files before running again")
    return paths


def backup_run_csvs(metrics_path, snr_path, run_config_dir=RUN_CONFIG_DIR):
    """Copy completed CSV bytes, reserving both names without overwriting files."""
    sources = (Path(metrics_path), Path(snr_path))
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Cannot back up missing CSV file: {source}")
    destinations = check_run_csv_destinations(run_config_dir)
    Path(run_config_dir).mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            outputs = []
            for destination in destinations:
                # Exclusive creation also protects against a conflict after preflight.
                try:
                    output = stack.enter_context(destination.open("xb"))
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Backup already exists: {destination}; rename the previous CSV files before running again"
                    ) from exc
                created.append(destination)
                outputs.append(output)
            for source, output in zip(sources, outputs):
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    for destination in destinations:
        print(f"CSV backup: {destination}", flush=True)


def run_training(cfg, resume=None):
    """Run configured epochs; tests may supply a small in-memory configuration."""
    check_run_csv_destinations(RUN_CONFIG_DIR)
    training = validate_training_config(cfg)
    loader_config = validate_loader_config(_required(cfg, "loader", "config"))
    dataset_config = _required(cfg, "dataset", "config")
    model_config = _required(cfg, "model", "config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DOAMambaNet training")
    device = torch.device("cuda")
    set_seed(training["seed"])
    dataset = MoDANetDatasetLoader(dataset_config=dataset_config)
    train_loader, eval_loader = build_dataloaders(dataset, training, loader_config)
    model = DOAMambaNet(dataset_config=dataset_config, model_config=model_config).to(
        device=device, dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    output_dir = _project_path(training["output_dir"])
    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    snr_path = output_dir / "eval_by_snr.csv"
    history = []
    best_eval_rmse_deg = math.inf
    best_checkpoint = None
    start_epoch = 1

    if resume is not None:
        checkpoint = load_checkpoint(
            _project_path(resume), model, optimizer, dataset_config, model_config, training,
            train_generator=train_loader.generator,
        )
        history = deepcopy(checkpoint["history"])
        best_eval_rmse_deg = checkpoint["best_eval_rmse_deg"]
        start_epoch = checkpoint["epoch"] + 1
        if not history or history[-1]["epoch"] != checkpoint["epoch"]:
            raise ValueError("Checkpoint history does not match its completed epoch")
        if min(row["eval_rmse_deg"] for row in history) != best_eval_rmse_deg:
            raise ValueError("Checkpoint best_eval_rmse_deg does not match its history")
        best_checkpoint = checkpoint.get("best_checkpoint")
        if best_checkpoint is None:
            if history[-1]["eval_rmse_deg"] != best_eval_rmse_deg:
                raise ValueError("Checkpoint lacks historical best state; cannot restore best.pt")
            best_checkpoint = checkpoint
        _check_compatibility(best_checkpoint, dataset_config, model_config, training)
        if (best_checkpoint["best_eval_rmse_deg"] != best_eval_rmse_deg
                or best_checkpoint["history"][-1]["eval_rmse_deg"] != best_eval_rmse_deg):
            raise ValueError("Embedded best checkpoint does not match best_eval_rmse_deg")
        # Checkpoint history is authoritative even after an interrupted CSV write.
        _atomic_torch_save(best_path, best_checkpoint)
        _atomic_torch_save(last_path, checkpoint)
        write_metrics_csv(metrics_path, history)
    elif any(path.exists() for path in (best_path, last_path, metrics_path, snr_path)):
        raise FileExistsError(f"Training outputs already exist in {output_dir}; use --resume")

    print(
        f"Device: {device}\nGPU: {torch.cuda.get_device_name(device)}\n"
        f"Dataset samples: {len(dataset)}\nTrain samples: {len(train_loader.dataset)}\n"
        f"Eval samples: {len(eval_loader.dataset)}\nBatch size: {training['batch_size']}\n"
        f"Epochs: {training['epochs']}\nLearning rate: {optimizer.param_groups[0]['lr']}\n"
        f"Model parameters: {sum(p.numel() for p in model.parameters())}\n"
        f"Output: {output_dir}\nResume: {resume if resume is not None else 'none'}\n"
        f"Starting epoch: {start_epoch}", flush=True,
    )
    for epoch in range(start_epoch, training["epochs"] + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch:03d}/{training['epochs']:03d}", flush=True)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, dataset_config,
            training["grad_clip_norm"], pin_memory=loader_config["pin_memory"],
        )
        eval_metrics = evaluate(
            model, eval_loader, criterion, device, dataset_config,
            pin_memory=loader_config["pin_memory"],
        )
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "rmse_deg")},
            **{f"eval_{key}": eval_metrics[key] for key in ("loss", "accuracy", "rmse_deg")},
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        improved = eval_metrics["rmse_deg"] < best_eval_rmse_deg
        if improved:
            best_eval_rmse_deg = eval_metrics["rmse_deg"]
            best_checkpoint = save_checkpoint(
                best_path, model, optimizer, epoch, best_eval_rmse_deg, history,
                dataset_config, model_config, training, train_generator=train_loader.generator,
            )
        save_checkpoint(
            last_path, model, optimizer, epoch, best_eval_rmse_deg, history,
            dataset_config, model_config, training, train_generator=train_loader.generator,
            best_checkpoint=best_checkpoint,
        )
        write_metrics_csv(metrics_path, history)
        for label, metrics in (("Train", train_metrics), ("Eval", eval_metrics)):
            print(f"{label} | loss={metrics['loss']:.6f} | accuracy={metrics['accuracy']:.2%} "
                  f"| RMSE={metrics['rmse_deg']:.6f} deg", flush=True)
        print(f"LR: {optimizer.param_groups[0]['lr']}", flush=True)

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    final_metrics = evaluate(model, eval_loader, criterion, device, dataset_config,
                             pin_memory=loader_config["pin_memory"])
    write_snr_metrics_csv(snr_path, final_metrics["by_snr"])
    backup_run_csvs(metrics_path, snr_path, run_config_dir=RUN_CONFIG_DIR)
    print(f"Final holdout (best epoch {best['epoch']}) | loss={final_metrics['loss']:.6f} "
          f"| accuracy={final_metrics['accuracy']:.2%} "
          f"| RMSE={final_metrics['rmse_deg']:.6f} deg", flush=True)
    return final_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modanet.yaml", help="YAML configuration path")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    args = parser.parse_args(argv)
    with _project_path(args.config).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    run_training(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
