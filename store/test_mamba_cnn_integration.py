"""Run directly with python3 store/test_mamba_cnn_integration.py (no pytest).

CUDA runtime and real-data tests report explicit skips when the required
device or external dataset is unavailable.
"""

from copy import deepcopy
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.modanet_dataset_loader import MoDANetDatasetLoader
from models.doa_covariance_cnn import DOACovarianceCNNHead
from models.mamba_cnn import DOAMambaNet
from models.mamba_signal_adapter import MambaSignalAdapter


def assert_raises(error_type, action, *message_parts):
    try:
        action()
    except error_type as exc:
        for part in message_parts:
            assert str(part) in str(exc), f"Missing {part!r} in error: {exc}"
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def test_structure(model, dataset_cfg):
    assert list(dict(model.named_children())) == ["signal_adapter", "doa_head"]
    assert isinstance(model.signal_adapter, MambaSignalAdapter)
    assert isinstance(model.doa_head, DOACovarianceCNNHead)
    assert list(model.named_parameters(recurse=False)) == []
    adapter_ids = {id(p) for p in model.signal_adapter.parameters()}
    head_ids = {id(p) for p in model.doa_head.parameters()}
    assert adapter_ids.isdisjoint(head_ids), "Adapter and head share parameters"
    assert {id(p) for p in model.parameters()} == adapter_ids | head_ids
    total = sum(p.numel() for p in model.parameters())
    assert total == (
        sum(p.numel() for p in model.signal_adapter.parameters())
        + sum(p.numel() for p in model.doa_head.parameters())
    )
    for key in ("snapshots", "antennas", "iq_channels"):
        assert getattr(model.signal_adapter, key) == getattr(model.doa_head, key) == dataset_cfg[key]
    expected_dim = dataset_cfg["antennas"] * dataset_cfg["iq_channels"]
    assert model.signal_adapter.input_dim == model.doa_head.input_dim == expected_dim
    assert model.doa_head.num_doa_classes == dataset_cfg["num_doa_classes"]
    for training in (False, True):
        model.train(training)
        assert all(module.training == training for module in model.modules())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PASS: composition, parameter ownership, config consistency and train/eval; "
          f"parameters={total}, trainable={trainable}")


def test_invalid_input(model, dataset_cfg):
    iq, antennas, snapshots = (
        dataset_cfg["iq_channels"], dataset_cfg["antennas"], dataset_cfg["snapshots"]
    )
    for shape in ((2, antennas, snapshots), (2, iq + 1, antennas, snapshots),
                  (2, iq, antennas - 1, snapshots), (2, iq, antennas, snapshots - 1)):
        assert_raises(ValueError, lambda: model(torch.zeros(shape)), "Expected", "actual shape", shape)
    for dtype in (torch.long, torch.bool, torch.complex64):
        x = torch.zeros(2, iq, antennas, snapshots, dtype=dtype)
        assert_raises(TypeError, lambda: model(x), "floating-point", "actual dtype")
    for invalid in (None, [], "tensor"):
        assert_raises(TypeError, lambda: model(invalid), "Expected", "actual type")
    assert_raises(TypeError, lambda: DOAMambaNet(num_antennas=antennas), "num_antennas")
    print("PASS: invalid input rejected through adapter; obsolete constructor rejected")


def test_state_dict_load(model, dataset_cfg, model_cfg):
    state = model.state_dict()
    assert state
    assert {key.split(".", 1)[0] for key in state} == {"signal_adapter", "doa_head"}
    restored = DOAMambaNet(dataset_config=dataset_cfg, model_config=model_cfg)
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert {id(p) for p in model.parameters()}.isdisjoint(id(p) for p in restored.parameters())
    print("PASS: new state_dict namespaces and strict in-memory load without missing/unexpected keys")
    return restored


def assert_nonzero_finite_gradient(tensor, name):
    assert tensor.grad is not None, f"Missing gradient: {name}"
    assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient: {name}"
    assert torch.count_nonzero(tensor.grad) > 0, f"Zero gradient: {name}"


def test_forward_backward(model, dataset_cfg):
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(
        2, dataset_cfg["iq_channels"], dataset_cfg["antennas"], dataset_cfg["snapshots"],
        device="cuda", requires_grad=True,
    )
    original = x.detach().clone()
    captured = {}
    calls = []

    def capture_adapter(module, inputs, output):
        calls.append("adapter")
        captured["adapted"] = output
        captured["adapted_values"] = output.detach().clone()
        output.retain_grad()

    def capture_head_input(module, inputs):
        calls.append("head_input")
        captured["head_input"] = inputs[0]
        captured["head_input_values"] = inputs[0].detach().clone()

    def capture_head_output(module, inputs, output):
        calls.append("head_output")
        captured["head_output"] = output
        captured["head_output_values"] = output.detach().clone()

    handles = [
        model.signal_adapter.register_forward_hook(capture_adapter),
        model.doa_head.register_forward_pre_hook(capture_head_input),
        model.doa_head.register_forward_hook(capture_head_output),
    ]
    try:
        logits = model(x)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == ["adapter", "head_input", "head_output"]
    expected_sequence_shape = (
        x.shape[0], dataset_cfg["snapshots"], dataset_cfg["antennas"] * dataset_cfg["iq_channels"]
    )
    assert captured["adapted"].shape == captured["head_input"].shape == expected_sequence_shape
    assert captured["adapted"] is captured["head_input"], "Wrapper transformed adapter output"
    torch.testing.assert_close(captured["adapted_values"], captured["head_input_values"], rtol=0, atol=0)
    assert logits is captured["head_output"], "Wrapper transformed head output"
    torch.testing.assert_close(logits, captured["head_output_values"], rtol=0, atol=0)
    assert isinstance(logits, torch.Tensor) and logits.is_floating_point()
    assert logits.shape == (x.shape[0], dataset_cfg["num_doa_classes"])
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)

    loss = logits.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert_nonzero_finite_gradient(x, "raw input")
    assert_nonzero_finite_gradient(captured["adapted"], "adapted sequence")
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"

    adapter, head = model.signal_adapter, model.doa_head
    for name, module in (
        ("input_proj", adapter.input_proj), ("delta_proj", adapter.delta_proj),
        ("conv1", head.conv1), ("conv2", head.conv2), ("conv3", head.conv3),
        ("feature_proj", head.feature_proj), ("classifier", head.classifier),
    ):
        assert_nonzero_finite_gradient(module.weight, name)
    for index, block in enumerate(adapter.blocks):
        assert_nonzero_finite_gradient(block.mamba.in_proj.weight, f"Mamba block {index}")
    torch.cuda.synchronize()
    print(f"PASS: CUDA {tuple(x.shape)} -> {expected_sequence_shape} -> {tuple(logits.shape)}; "
          "direct tensor handoff, unchanged input, finite nonzero gradients through every stage")
    return x.detach()


def test_state_dict_output(model, restored, x):
    model.eval()
    restored.cuda().eval()
    with torch.no_grad():
        expected = model(x)
        actual = restored(x)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("PASS: equivalent eval outputs after state_dict round trip")


def test_real_batch(model, dataset_cfg):
    root = Path(dataset_cfg["root"]).expanduser()
    if not root.exists():
        print(f"SKIP: real MoDANet dataset is absent: {root}")
        return False
    dataset = MoDANetDatasetLoader(dataset_config=dataset_cfg)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["x"].shape == (
        2, dataset_cfg["iq_channels"], dataset_cfg["antennas"], dataset_cfg["snapshots"]
    )
    x = batch["x"].cuda()
    original = x.clone()
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (x.shape[0], dataset_cfg["num_doa_classes"])
    assert logits.is_floating_point() and torch.isfinite(logits).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    print(f"PASS: real MoDANet batch {tuple(x.shape)} -> finite logits {tuple(logits.shape)} "
          f"({len(dataset)} files indexed; only batch['x'] supplied to model)")
    return True


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    before = deepcopy(cfg)
    dataset_cfg, model_cfg = cfg["dataset"], cfg["model"]
    torch.manual_seed(0)
    model = DOAMambaNet(dataset_config=dataset_cfg, model_config=model_cfg)
    test_structure(model, dataset_cfg)
    test_invalid_input(model, dataset_cfg)
    restored = test_state_dict_load(model, dataset_cfg, model_cfg)
    assert cfg == before, "Constructor mutated caller config"

    if not torch.cuda.is_available():
        print("PASS: CPU structure, validation and state_dict tests")
        print("SKIP: CUDA unavailable; full forward/backward, checkpoint output, custom runtime "
              "and real-data runtime tests were not run")
        return

    torch.cuda.reset_peak_memory_stats()
    model.cuda()
    x = test_forward_backward(model, dataset_cfg)
    test_state_dict_output(model, restored, x)

    custom_dataset = dict(dataset_cfg, antennas=6, snapshots=20, num_doa_classes=31)
    custom_model_cfg = deepcopy(model_cfg)
    custom_before = deepcopy((custom_dataset, custom_model_cfg))
    custom_model = DOAMambaNet(custom_dataset, custom_model_cfg).cuda()
    test_structure(custom_model, custom_dataset)
    test_forward_backward(custom_model, custom_dataset)
    assert (custom_dataset, custom_model_cfg) == custom_before, "Custom config was mutated"

    real_data_passed = test_real_batch(model, dataset_cfg)
    assert cfg == before, "Forward mutated caller config"
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"PASS: caller configs unchanged; GPU {torch.cuda.get_device_name()}, "
          f"peak allocated={peak_mib:.1f} MiB")
    if real_data_passed:
        print("All DOAMambaNet integration tests passed, including CUDA and real MoDANet data.")
    else:
        print("All synthetic DOAMambaNet integration tests passed; real-data test skipped.")


if __name__ == "__main__":
    main()
