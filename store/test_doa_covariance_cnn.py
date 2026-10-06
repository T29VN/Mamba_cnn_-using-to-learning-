"""Run with python3 store/test_doa_covariance_cnn.py; CUDA checks are optional."""

from copy import deepcopy
import math
from pathlib import Path
import sys

import torch
from torch import nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.doa_covariance_cnn import DOACovarianceCNNHead


def assert_raises(error_type, action, *message_parts):
    try:
        action()
    except error_type as exc:
        for part in message_parts:
            assert str(part).lower() in str(exc).lower(), f"Missing {part!r} in error: {exc}"
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def test_config_validation(dataset_cfg, model_cfg):
    for invalid in (None, [], "config.yaml"):
        assert_raises(ValueError, lambda: DOACovarianceCNNHead(invalid, model_cfg), "dataset_config")
        assert_raises(ValueError, lambda: DOACovarianceCNNHead(dataset_cfg, invalid), "model_config")

    bad_model = deepcopy(model_cfg)
    del bad_model["doa_head"]
    assert_raises(KeyError, lambda: DOACovarianceCNNHead(dataset_cfg, bad_model), "doa_head")
    for invalid in (None, [], "config.yaml"):
        bad_model["doa_head"] = invalid
        assert_raises(ValueError, lambda: DOACovarianceCNNHead(dataset_cfg, bad_model), "doa_head")

    sections = (
        ("dataset", dataset_cfg, ("snapshots", "antennas", "iq_channels", "num_doa_classes")),
        ("doa_head", model_cfg["doa_head"],
         ("kernel_size", "pool_kernel_size", "pool_stride", "feature_dim")),
    )
    for section, config, keys in sections:
        for key in keys:
            bad = deepcopy(config)
            del bad[key]
            args = ((bad, model_cfg) if section == "dataset"
                    else (dataset_cfg, dict(model_cfg, doa_head=bad)))
            assert_raises(KeyError, lambda: DOACovarianceCNNHead(*args), key)
            for value in (0, -1, 1.5, True, False, "2", None):
                bad = dict(config, **{key: value})
                args = ((bad, model_cfg) if section == "dataset"
                        else (dataset_cfg, dict(model_cfg, doa_head=bad)))
                assert_raises(ValueError, lambda: DOACovarianceCNNHead(*args), key)

    for channels in (1, 3):
        assert_raises(
            ValueError,
            lambda: DOACovarianceCNNHead(dict(dataset_cfg, iq_channels=channels), model_cfg),
            "iq_channels",
        )

    bad_model = deepcopy(model_cfg)
    del bad_model["doa_head"]["conv_channels"]
    assert_raises(KeyError, lambda: DOACovarianceCNNHead(dataset_cfg, bad_model), "conv_channels")
    for invalid in (None, 32, "32,64,64", {}, [], [32], [32, 64], [32, 64, 64, 64],
                    [0, 64, 64], [32, -1, 64], [32, 64, True], [False, 64, 64],
                    [32, 1.5, 64], [32, 64, "64"]):
        bad_model["doa_head"]["conv_channels"] = invalid
        assert_raises(ValueError, lambda: DOACovarianceCNNHead(dataset_cfg, bad_model), "conv_channels")

    for key, invalid in (("kernel_size", 2), ("kernel_size", 4),
                         ("pool_kernel_size", dataset_cfg["antennas"] + 1)):
        bad_model = deepcopy(model_cfg)
        bad_model["doa_head"][key] = invalid
        assert_raises(ValueError, lambda: DOACovarianceCNNHead(dataset_cfg, bad_model), key)
    print("PASS: required config sections/keys, positive integers, I/Q, channels, kernel and pooling validation")


def test_restore_iq(head):
    batch_size = 2
    # Assign antenna/IQ values directly so the reference does not repeat the reshape logic.
    sequence = torch.empty(batch_size, head.snapshots, head.input_dim * 2)[..., ::2]
    expected = torch.empty(batch_size, 2, head.antennas, head.snapshots)
    offsets = (10000 * torch.arange(batch_size)[:, None]
               + 100 * torch.arange(head.snapshots)[None, :])
    for antenna in range(head.antennas):
        for channel in range(2):
            values = offsets + 10 * (antenna + 1) + channel + 1
            sequence[:, :, 2 * antenna + channel] = values
            expected[:, channel, antenna, :] = values
    assert not sequence.is_contiguous()
    original = sequence.clone()
    restored = head._restore_iq(sequence)
    assert restored.is_contiguous()
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(sequence, original, rtol=0, atol=0)

    # Match Task 2.1's interface without importing or instantiating Mamba.
    iq = torch.randn(batch_size, 2, head.antennas, head.snapshots)
    interleaved = iq.permute(0, 3, 2, 1).contiguous().reshape(
        batch_size, head.snapshots, head.input_dim,
    )
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        actual = head._restore_iq(interleaved.to(dtype))
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, iq.to(dtype), rtol=0, atol=0)


def test_input_validation(head):
    snapshots, input_dim = head.snapshots, head.input_dim
    for action in (head, head._restore_iq):
        for shape in ((2, input_dim), (2, snapshots - 1, input_dim),
                      (2, snapshots, input_dim - 1), (2, snapshots, input_dim + 1),
                      (2, input_dim, snapshots), (2, 1, snapshots, input_dim)):
            assert_raises(ValueError, lambda: action(torch.zeros(shape)), "expected", "actual", shape)
        for invalid in (None, [], "tensor"):
            assert_raises(TypeError, lambda: action(invalid), "expected", "actual")
        for dtype in (torch.long, torch.bool, torch.complex64):
            sequence = torch.zeros(2, snapshots, input_dim, dtype=dtype)
            assert_raises(TypeError, lambda: action(sequence), "floating-point", "actual")
    print("PASS: numerical interleaved I/Q restore, round trip, non-contiguous input and input validation")


def test_covariance(dataset_cfg, model_cfg):
    small_dataset = dict(dataset_cfg, antennas=3, snapshots=4)
    small_model = deepcopy(model_cfg)
    small_model["doa_head"].update(pool_kernel_size=2, pool_stride=2)
    head = DOACovarianceCNNHead(small_dataset, small_model)
    # Nonzero means and imaginary covariance expose centering, conjugation and sign errors.
    iq = torch.tensor(
        [[[[1, 2, -1, 4], [3, -2, 5, 1], [-1, 0, 2, 3]],
          [[2, -1, 3, 1], [0, 4, -2, 2], [3, 1, 0, -4]]]],
        dtype=torch.float64,
    )
    original = iq.clone()
    covariance = head._covariance_features(iq)
    y = torch.complex(iq[:, 0], iq[:, 1])
    reference = (y @ y.conj().transpose(-1, -2)) / small_dataset["snapshots"]
    assert torch.count_nonzero(reference.imag) > 0
    assert torch.count_nonzero(y.mean(dim=-1)) > 0
    assert covariance.shape == (1, 2, 3, 3)
    assert covariance.dtype == iq.dtype
    assert covariance.is_floating_point()
    torch.testing.assert_close(covariance, torch.stack((reference.real, reference.imag), dim=1),
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(iq, original, rtol=0, atol=0)

    real_cov, imag_cov = covariance[:, 0], covariance[:, 1]
    torch.testing.assert_close(real_cov, real_cov.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(imag_cov, -imag_cov.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    diagonal = torch.diagonal(imag_cov, dim1=-2, dim2=-1)
    torch.testing.assert_close(diagonal, torch.zeros_like(diagonal), rtol=0, atol=1e-12)

    phi = 0.7
    i_rotated = iq[:, 0] * math.cos(phi) - iq[:, 1] * math.sin(phi)
    q_rotated = iq[:, 0] * math.sin(phi) + iq[:, 1] * math.cos(phi)
    rotated_cov = head._covariance_features(torch.stack((i_rotated, q_rotated), dim=1))
    torch.testing.assert_close(rotated_cov, covariance, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(head._covariance_features(3 * iq), 9 * covariance,
                               rtol=1e-12, atol=1e-12)
    print("PASS: complex covariance reference, /T scale, Hermitian properties and global phase invariance")


def test_structure(head, dataset_cfg, model_cfg):
    cnn_cfg = model_cfg["doa_head"]
    channels = cnn_cfg["conv_channels"]
    antennas = dataset_cfg["antennas"]
    pooled_size = (antennas - cnn_cfg["pool_kernel_size"]) // cnn_cfg["pool_stride"] + 1
    flatten_dim = channels[-1] * pooled_size * pooled_size
    assert head.snapshots == dataset_cfg["snapshots"]
    assert head.antennas == antennas
    assert head.iq_channels == dataset_cfg["iq_channels"]
    assert head.input_dim == antennas * dataset_cfg["iq_channels"]
    assert head.num_doa_classes == dataset_cfg["num_doa_classes"]
    assert head.pooled_size == pooled_size
    assert head.flatten_dim == flatten_dim

    convolutions = [module for module in head.modules() if isinstance(module, nn.Conv2d)]
    assert convolutions == [head.conv1, head.conv2, head.conv3]
    for conv, in_channels, out_channels in zip(convolutions, (2, *channels[:-1]), channels):
        assert (conv.in_channels, conv.out_channels) == (in_channels, out_channels)
        assert conv.kernel_size == (cnn_cfg["kernel_size"],) * 2
        assert conv.padding == (cnn_cfg["kernel_size"] // 2,) * 2
        assert conv.stride == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1

    pool_types = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)
    pools = [module for module in head.modules() if isinstance(module, pool_types)]
    assert pools == [head.pool] and isinstance(head.pool, nn.MaxPool2d)
    assert head.pool.kernel_size == cnn_cfg["pool_kernel_size"]
    assert head.pool.stride == cnn_cfg["pool_stride"]
    assert head.pool.padding == 0 and head.pool.dilation == 1
    assert head.pool.ceil_mode is False
    assert isinstance(head.relu, nn.ReLU)
    assert isinstance(head.flatten, nn.Flatten) and head.flatten.start_dim == 1
    assert isinstance(head.feature_proj, nn.Linear) and isinstance(head.classifier, nn.Linear)
    assert (head.feature_proj.in_features, head.feature_proj.out_features) == (
        flatten_dim, cnn_cfg["feature_dim"],
    )
    assert (head.classifier.in_features, head.classifier.out_features) == (
        cnn_cfg["feature_dim"], dataset_cfg["num_doa_classes"],
    )
    forbidden_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.Softmax, nn.LogSoftmax, nn.Sigmoid, nn.MultiheadAttention,
    )
    assert not any(isinstance(module, forbidden_types) for module in head.modules())


def check_gradients(head, sequence):
    assert sequence.grad is not None, "Missing input gradient through covariance"
    assert torch.isfinite(sequence.grad).all(), "Non-finite input gradient"
    assert torch.count_nonzero(sequence.grad) > 0, "Zero input gradient through covariance"
    for name in ("conv1", "conv2", "conv3", "feature_proj", "classifier"):
        module = getattr(head, name)
        for parameter_name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}.{parameter_name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}.{parameter_name}"
        assert torch.count_nonzero(module.weight.grad) > 0, f"Zero weight gradient: {name}"


def test_forward_backward(head, dataset_cfg, model_cfg, batch_size=4, inspect_stages=True):
    head.train()
    head.zero_grad(set_to_none=True)
    sequence = torch.randn(batch_size, head.snapshots, head.input_dim,
                           device=head.conv1.weight.device, requires_grad=True)
    original = sequence.detach().clone()
    stages = []

    def capture(name):
        def hook(module, inputs, output):
            stages.append((name, inputs[0].detach().clone(), output.detach().clone()))
        return hook

    handles = []
    if inspect_stages:
        names = ("conv1", "conv2", "pool", "conv3", "relu", "flatten", "feature_proj", "classifier")
        handles = [getattr(head, name).register_forward_hook(capture(name)) for name in names]
    try:
        logits = head(sequence)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(logits, torch.Tensor) and logits.is_floating_point()
    assert logits.shape == (batch_size, dataset_cfg["num_doa_classes"])
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(sequence, original, rtol=0, atol=0)

    if inspect_stages:
        names = [name for name, _, _ in stages]
        assert names == ["conv1", "relu", "conv2", "relu", "pool", "conv3", "relu",
                         "flatten", "feature_proj", "relu", "classifier"], names
        cnn_cfg = model_cfg["doa_head"]
        c1, c2, c3 = cnn_cfg["conv_channels"]
        m = dataset_cfg["antennas"]
        p = (m - cnn_cfg["pool_kernel_size"]) // cnn_cfg["pool_stride"] + 1
        shapes = {
            "conv1": (batch_size, c1, m, m), "conv2": (batch_size, c2, m, m),
            "pool": (batch_size, c2, p, p), "conv3": (batch_size, c3, p, p),
            "flatten": (batch_size, c3 * p * p),
            "feature_proj": (batch_size, cnn_cfg["feature_dim"]),
            "classifier": (batch_size, dataset_cfg["num_doa_classes"]),
        }
        previous = head._covariance_features(head._restore_iq(original))
        assert previous.shape == (batch_size, 2, m, m)
        for name, actual_input, actual_output in stages:
            torch.testing.assert_close(actual_input, previous, rtol=0, atol=0)
            if name == "relu":
                torch.testing.assert_close(actual_output, actual_input.clamp_min(0), rtol=0, atol=0)
            else:
                assert actual_output.shape == shapes[name], (name, actual_output.shape)
            if name == "flatten":
                torch.testing.assert_close(actual_output, actual_input.flatten(start_dim=1), rtol=0, atol=0)
            previous = actual_output
        # Equality with the classifier output also detects functional softmax/sigmoid.
        torch.testing.assert_close(logits, previous, rtol=0, atol=0)

    loss = logits.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    check_gradients(head, sequence)
    return tuple(logits.shape)


def test_custom_config(dataset_cfg, model_cfg):
    custom_dataset = dict(dataset_cfg, antennas=6, snapshots=20, num_doa_classes=31)
    head = DOACovarianceCNNHead(custom_dataset, model_cfg)
    test_structure(head, custom_dataset, model_cfg)
    test_restore_iq(head)
    test_forward_backward(head, custom_dataset, model_cfg, batch_size=2)

    custom_dataset = dict(dataset_cfg, antennas=7, snapshots=13, num_doa_classes=7)
    custom_model = deepcopy(model_cfg)
    custom_model["doa_head"].update(conv_channels=(4, 7, 9), kernel_size=5,
                                    pool_kernel_size=3, pool_stride=1, feature_dim=11)
    head = DOACovarianceCNNHead(custom_dataset, custom_model)
    test_structure(head, custom_dataset, custom_model)
    test_forward_backward(head, custom_dataset, custom_model, batch_size=2)
    print("PASS: custom antennas/snapshots/classes, tuple channels, convolution/pooling kernels and feature dimensions")


def test_cuda(dataset_cfg, model_cfg):
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; all CPU DOACovarianceCNNHead tests completed")
        return
    torch.cuda.reset_peak_memory_stats()
    head = DOACovarianceCNNHead(dataset_cfg, model_cfg).cuda()
    shape = test_forward_backward(head, dataset_cfg, model_cfg, batch_size=2, inspect_stages=False)
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"PASS: CUDA {torch.cuda.get_device_name()}, output {shape}, finite nonzero gradients; "
          f"peak allocated={peak_mib:.1f} MiB")


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    torch.manual_seed(0)
    dataset_cfg, model_cfg = cfg["dataset"], cfg["model"]
    assert "num_doa_classes" not in model_cfg["doa_head"], "Class count belongs in dataset config"
    test_config_validation(dataset_cfg, model_cfg)
    head = DOACovarianceCNNHead(dataset_config=dataset_cfg, model_config=model_cfg)
    test_restore_iq(head)
    test_input_validation(head)
    test_covariance(dataset_cfg, model_cfg)
    test_structure(head, dataset_cfg, model_cfg)
    shape = test_forward_backward(head, dataset_cfg, model_cfg)
    print(f"PASS: CPU output {shape}, 3 convolutions, one MaxPool after Conv2, stage shapes and raw logits")
    print("PASS: finite nonzero input/weight gradients through the complete covariance and CNN head")
    test_custom_config(dataset_cfg, model_cfg)
    test_cuda(dataset_cfg, model_cfg)
    print("All CPU DOACovarianceCNNHead tests passed; CUDA status reported above.")


if __name__ == "__main__":
    main()
