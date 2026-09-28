"""Run with python3 store/test_mamba_signal_adapter.py; full tests require CUDA."""

from copy import deepcopy
import math
from pathlib import Path
import sys

from mamba_ssm import Mamba
import torch
from torch import nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.mamba_signal_adapter import MambaSignalAdapter, ResidualMambaBlock


def assert_raises(error_type, action, *message_parts):
    try:
        action()
    except error_type as exc:
        for part in message_parts:
            assert str(part) in str(exc), f"Missing {part!r} in error: {exc}"
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def test_config_validation(dataset_cfg, model_cfg):
    for invalid in (None, [], "config.yaml"):
        assert_raises(ValueError, lambda: MambaSignalAdapter(invalid, model_cfg), "dataset_config")
        assert_raises(ValueError, lambda: MambaSignalAdapter(dataset_cfg, invalid), "model_config")

    sections = [("dataset", dataset_cfg, ("snapshots", "antennas", "iq_channels")),
                ("mamba", model_cfg["mamba"], ("d_model", "num_layers", "d_state", "d_conv", "expand"))]
    for section, config, keys in sections:
        for key in keys:
            bad = deepcopy(config)
            del bad[key]
            args = (bad, model_cfg) if section == "dataset" else (dataset_cfg, dict(model_cfg, mamba=bad))
            assert_raises(KeyError, lambda: MambaSignalAdapter(*args), key)
            for value in (0, -1, 1.5, True, False, "2", None):
                bad = dict(config, **{key: value})
                args = (bad, model_cfg) if section == "dataset" else (dataset_cfg, dict(model_cfg, mamba=bad))
                assert_raises(ValueError, lambda: MambaSignalAdapter(*args), key)
    for channels in (1, 3):
        assert_raises(
            ValueError, lambda: MambaSignalAdapter(dict(dataset_cfg, iq_channels=channels), model_cfg),
            "iq_channels",
        )
    for section in ("mamba", "signal_adapter"):
        bad = deepcopy(model_cfg)
        del bad[section]
        assert_raises(KeyError, lambda: MambaSignalAdapter(dataset_cfg, bad), section)
        bad[section] = []
        assert_raises(ValueError, lambda: MambaSignalAdapter(dataset_cfg, bad), section)
    for section, key, invalid_values in (
        ("mamba", "dt_rank", (0, -1, 1.5, True, "AUTO", None)),
        ("signal_adapter", "delta_init_std", (0, -0.001, True, "0.001", None, float("nan"), float("inf"))),
    ):
        bad = deepcopy(model_cfg)
        del bad[section][key]
        assert_raises(KeyError, lambda: MambaSignalAdapter(dataset_cfg, bad), key)
        for value in invalid_values:
            bad[section][key] = value
            assert_raises(ValueError, lambda: MambaSignalAdapter(dataset_cfg, bad), key)
    print("PASS: required config keys, dictionaries, integer fields, I/Q channels, dt_rank and delta std")


def test_structure(adapter, dataset_cfg, model_cfg):
    mamba_cfg = model_cfg["mamba"]
    input_dim = dataset_cfg["antennas"] * dataset_cfg["iq_channels"]
    d_model = mamba_cfg["d_model"]
    assert adapter.input_dim == input_dim
    assert isinstance(adapter.input_proj, nn.Linear)
    assert (adapter.input_proj.in_features, adapter.input_proj.out_features) == (input_dim, d_model)
    assert isinstance(adapter.delta_proj, nn.Linear)
    assert (adapter.delta_proj.in_features, adapter.delta_proj.out_features) == (d_model, input_dim)
    assert isinstance(adapter.final_norm, nn.LayerNorm)
    assert adapter.final_norm.normalized_shape == (d_model,)
    assert isinstance(adapter.blocks, nn.ModuleList)
    assert len(adapter.blocks) == mamba_cfg["num_layers"]
    seen_parameters = set()
    for block in adapter.blocks:
        assert isinstance(block, ResidualMambaBlock)
        assert isinstance(block.norm, nn.LayerNorm)
        assert block.norm.normalized_shape == (d_model,)
        assert isinstance(block.mamba, Mamba)
        for key in ("d_model", "d_state", "d_conv", "expand"):
            assert getattr(block.mamba, key) == mamba_cfg[key]
        rank = mamba_cfg["dt_rank"]
        assert block.mamba.dt_rank == (math.ceil(d_model / 16) if rank == "auto" else rank)
        parameter_ids = {id(parameter) for parameter in block.parameters()}
        assert seen_parameters.isdisjoint(parameter_ids), "Mamba blocks share parameters"
        seen_parameters.update(parameter_ids)
    assert len({id(block) for block in adapter.blocks}) == len(adapter.blocks)
    weights = adapter.delta_proj.weight.detach()
    assert torch.count_nonzero(weights) > 0
    measured_std = weights.std().item()
    configured_std = model_cfg["signal_adapter"]["delta_init_std"]
    assert 0.7 * configured_std < measured_std < 1.3 * configured_std
    assert torch.count_nonzero(adapter.delta_proj.bias) == 0


def test_snapshot_ordering(adapter):
    # Unique values across batches, snapshots and antenna/IQ pairs catch axis swaps.
    batch_size = 2
    x = torch.empty(batch_size, adapter.iq_channels, adapter.antennas, adapter.snapshots * 2)
    x = x[..., ::2]  # The helper must also accept a valid, non-contiguous input.
    offsets = (10000 * torch.arange(batch_size)[:, None]
               + 100 * torch.arange(adapter.snapshots)[None, :])
    expected = torch.empty(batch_size, adapter.snapshots, adapter.input_dim)
    for antenna in range(adapter.antennas):
        for iq in range(adapter.iq_channels):
            values = offsets + 10 * (antenna + 1) + iq + 1
            x[:, iq, antenna, :] = values
            expected[:, :, antenna * adapter.iq_channels + iq] = values
    original = x.clone()
    sequence = adapter._to_snapshot_sequence(x)
    assert sequence.shape == (batch_size, adapter.snapshots, adapter.input_dim)
    assert sequence.is_contiguous()
    torch.testing.assert_close(sequence, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        assert adapter._to_snapshot_sequence(x.to(dtype)).dtype == dtype


def test_input_validation(adapter):
    iq, antennas, snapshots = adapter.iq_channels, adapter.antennas, adapter.snapshots
    for shape in ((2, antennas, snapshots), (2, iq + 1, antennas, snapshots),
                  (2, iq, antennas - 1, snapshots), (2, iq, antennas, snapshots - 1)):
        assert_raises(ValueError, lambda: adapter(torch.zeros(shape)), "Expected", "actual shape", shape)
    for value in (None, [], "tensor"):
        assert_raises(TypeError, lambda: adapter(value), "Expected", "actual type")
    for dtype in (torch.long, torch.bool, torch.complex64):
        x = torch.zeros(1, iq, antennas, snapshots, dtype=dtype)
        assert_raises(TypeError, lambda: adapter(x), "floating-point", "actual dtype")
    print("PASS: numerical I/Q ordering, snapshot shapes, input validation, independent blocks and initialization")


def test_custom_config(dataset_cfg, model_cfg):
    custom_dataset = dict(dataset_cfg, antennas=3, snapshots=7)
    custom_model = deepcopy(model_cfg)
    custom_model["mamba"].update(d_model=24, num_layers=3, d_state=8, d_conv=3, expand=1, dt_rank=3)
    custom_model["signal_adapter"]["delta_init_std"] = 0.002
    adapter = MambaSignalAdapter(custom_dataset, custom_model)
    test_structure(adapter, custom_dataset, custom_model)
    test_snapshot_ordering(adapter)
    print("PASS: alternate in-memory config controls dimensions, blocks, Mamba parameters and initialization")


def check_gradients(module, name):
    for parameter_name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}.{parameter_name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}.{parameter_name}"
    assert any(torch.count_nonzero(p.grad).item() > 0 for p in module.parameters()), name


def test_cuda(adapter):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPU checks passed, but CUDA is unavailable: full Mamba forward/backward tests were not run. "
            "Run this script in the project environment with GPU access."
        )
    torch.cuda.reset_peak_memory_stats()
    adapter = adapter.cuda().train()
    x = torch.randn(2, adapter.iq_channels, adapter.antennas, adapter.snapshots, device="cuda")
    original = x.clone()
    raw = adapter._to_snapshot_sequence(x)

    # Observe the real CUDA path to verify pre-norm/residual math and stage connections.
    stages = {}
    def capture(name):
        def hook(module, inputs, output):
            stages[name] = (inputs[0].detach().clone(), output.detach().clone())
        return hook

    modules = {"input_proj": adapter.input_proj, "final_norm": adapter.final_norm,
               "delta_proj": adapter.delta_proj}
    for index, block in enumerate(adapter.blocks):
        modules.update({f"block{index}": block, f"norm{index}": block.norm, f"mamba{index}": block.mamba})
    handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
    try:
        output = adapter(x)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(output, torch.Tensor) and output.is_floating_point()
    assert output.shape == (x.shape[0], adapter.snapshots, adapter.input_dim)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(stages["input_proj"][0], raw, rtol=0, atol=0)
    previous = stages["input_proj"][1]
    for index, block in enumerate(adapter.blocks):
        block_input, block_output = stages[f"block{index}"]
        assert block_input.shape == (x.shape[0], adapter.snapshots, adapter.input_proj.out_features)
        torch.testing.assert_close(block_input, previous, rtol=0, atol=0)
        torch.testing.assert_close(stages[f"norm{index}"][0], block_input, rtol=0, atol=0)
        torch.testing.assert_close(stages[f"mamba{index}"][0], stages[f"norm{index}"][1], rtol=0, atol=0)
        torch.testing.assert_close(block_output, block_input + stages[f"mamba{index}"][1], rtol=0, atol=0)
        previous = block_output
    torch.testing.assert_close(stages["final_norm"][0], previous, rtol=0, atol=0)
    torch.testing.assert_close(stages["delta_proj"][0], stages["final_norm"][1], rtol=0, atol=0)
    torch.testing.assert_close(output, raw + stages["delta_proj"][1], rtol=0, atol=0)

    delta = output - raw
    ratio = (delta.square().mean().sqrt() / (raw.square().mean().sqrt() + 1e-12)).item()
    assert 0 < ratio < 0.05, f"Initial delta/raw RMS ratio is {ratio}"
    loss = output.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    check_gradients(adapter.input_proj, "input_proj")
    for index, block in enumerate(adapter.blocks):
        check_gradients(block, f"block{index}")
        assert torch.count_nonzero(block.mamba.in_proj.weight.grad) > 0, f"Mamba block {index} has zero gradient"
    check_gradients(adapter.final_norm, "final_norm")
    check_gradients(adapter.delta_proj, "delta_proj")
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"PASS: CUDA {torch.cuda.get_device_name()}, output {tuple(output.shape)}, delta/raw RMS={ratio:.6f}")
    print(f"PASS: real pre-norm/residual path, finite loss and gradients through every block; peak allocated={peak_mib:.1f} MiB")


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    torch.manual_seed(0)
    test_config_validation(cfg["dataset"], cfg["model"])
    adapter = MambaSignalAdapter(dataset_config=cfg["dataset"], model_config=cfg["model"])
    test_structure(adapter, cfg["dataset"], cfg["model"])
    test_snapshot_ordering(adapter)
    test_input_validation(adapter)
    test_custom_config(cfg["dataset"], cfg["model"])
    test_cuda(adapter)
    print("All MambaSignalAdapter tests passed.")


if __name__ == "__main__":
    main()
