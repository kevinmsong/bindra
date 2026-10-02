"""``bindra doctor`` — report whether this machine can actually train.

Prints the torch build, the GPU inventory with per-device verdicts, and then
runs real work on the selected device: a bf16 matmul and, when the weights are
present, an ESM-C-600M forward pass. Exits non-zero if the environment is not
fit for training, so CI and shell scripts can gate on it.
"""

from __future__ import annotations

import argparse
import sys
import time

from bindra.runtime.device import (
    DeviceRequirement,
    DeviceSelectionError,
    list_cuda_devices,
    select_gpu,
    torch_build_info,
)

OK = "  ok   "
FAIL = " FAIL  "
SKIP = " skip  "


def _print_build() -> None:
    info = torch_build_info()
    print("torch build")
    print(f"    version            {info['torch_version']}")
    print(f"    cuda build         {info['cuda_build_version'] or 'NONE (cpu-only wheel)'}")
    print(f"    cudnn              {info['cudnn_version']}")
    print(f"    cuda available     {info['cuda_available']}")
    print(f"    device count       {info['device_count']}")
    print(f"    kernel archs       {', '.join(info['arch_list']) or '(none)'}")
    print(f"    CUDA_DEVICE_ORDER  {info['cuda_device_order']}")
    print(f"    CUDA_VISIBLE_DEVICES {info['cuda_visible_devices']}")
    print()


def _print_inventory(requirement: DeviceRequirement) -> None:
    devices = list_cuda_devices()
    print(f"visible CUDA devices (requirement: {requirement.describe()})")
    if not devices:
        print("    (none)")
    for gpu in devices:
        print(f"    {gpu.describe()}")
    print()


def _check_bf16_matmul(device) -> bool:
    import torch

    try:
        size = 4096
        repeats = 20
        a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=device, dtype=torch.bfloat16)

        # The first CUDA op on a fresh process pays for context creation, the
        # cuBLAS handle and workspace allocation. Timing that reports seconds
        # for a matmul that takes milliseconds, so warm up before measuring.
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(repeats):
            c = a @ b
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / repeats

        # 2*N^3 flops for an NxN matmul.
        tflops = (2 * size**3) / elapsed / 1e12
        if not bool(torch.isfinite(c.float()).all().item()):
            print(f"[{FAIL}] bf16 matmul produced non-finite values")
            return False
        print(
            f"[{OK}] bf16 matmul {size}x{size}: {elapsed * 1000:.2f} ms/iter "
            f"({tflops:.1f} TFLOP/s, best of {repeats} after warmup)"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        print(f"[{FAIL}] bf16 matmul failed: {type(exc).__name__}: {exc}")
        return False


def _check_vram(device) -> bool:
    import torch

    free, total = torch.cuda.mem_get_info(device)
    print(
        f"[{OK}] VRAM {free / 1024**3:.1f} GiB free of {total / 1024**3:.1f} GiB total"
    )
    return True


def _check_esmc(device, enabled: bool, *, verify_weights: bool = False) -> bool | None:
    """Encode a short sequence through the real adapter. None when skipped.

    This deliberately goes through :class:`ESMCProteinEncoder` rather than
    calling ``esm`` directly. ``doctor`` gates the feature-cache run, so it has
    to exercise the code path that populates the cache -- including the pinned
    revision, the post-LayerNorm tensor selected by name, and special-token
    stripping via the tokenizer's own mask. An earlier version of this check
    used the deprecated ``ESMC`` shim and ``output.embeddings``, which are the
    three defects ``docs/MIGRATION.md`` records as fixed: it could have passed
    while the adapter was broken, or vice versa.
    """
    if not enabled:
        print(f"[{SKIP}] ESM-C forward pass (use --esmc to run it)")
        return None

    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
        ProteinEncoderError,
        detect_kernel_backends,
    )

    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    encoder = ESMCProteinEncoder(device=device)

    try:
        features = encoder.encode(sequence)
    except ProteinEncoderError as exc:
        # Missing weights or a missing 'esm' install is a skip, not a failure:
        # the data layer and the test suite are installable without them.
        if "is required for the ESM-C encoder" in str(exc):
            print(f"[{SKIP}] {exc}")
            return None
        print(f"[{FAIL}] ESM-C encode failed: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        print(f"[{FAIL}] ESM-C encode failed: {type(exc).__name__}: {exc}")
        return False

    shape = tuple(features.residue_features.shape)
    print(
        f"[{OK}] ESM-C-600M encode: {shape} ({features.residue_features.dtype}) "
        f"for a {len(sequence)}-residue sequence"
    )

    ok = True
    if features.feature_dim != EXPECTED_ESMC_600M_DIM:
        print(
            f"[{FAIL}] expected a {EXPECTED_ESMC_600M_DIM}-dim embedding, "
            f"got {features.feature_dim}"
        )
        ok = False
    if shape[0] != len(sequence):
        print(f"[{FAIL}] expected {len(sequence)} residue rows, got {shape[0]}")
        ok = False
    if not features.covers_full_sequence:
        print(f"[{FAIL}] encoder did not cover every residue of the input")
        ok = False

    backends = detect_kernel_backends()
    print(
        f"[{OK}] kernel backends: {', '.join(backends) if backends else 'none (pure PyTorch)'}"
        "  (part of the cache key)"
    )
    print(f"[{OK}] pinned revision {encoder.revision}")

    if verify_weights:
        # Hashing the state dict is seconds of work but reads every parameter,
        # so it is opt-in rather than part of the default gate.
        checksum = encoder.weights_checksum()
        if checksum == ESMC_600M_WEIGHTS_SHA256:
            print(f"[{OK}] weights checksum matches the pinned digest")
        else:
            print(
                f"[{FAIL}] weights checksum {checksum[:16]}... does not match the "
                f"pinned {ESMC_600M_WEIGHTS_SHA256[:16]}...; the HF cache may be "
                "corrupt or locally overridden"
            )
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bindra doctor",
        description="Check that this machine can train BINDRA on the GPU.",
    )
    parser.add_argument(
        "--min-memory-gib",
        type=float,
        default=DeviceRequirement().min_memory_gib,
        help="Reject devices smaller than this (default: %(default)s).",
    )
    parser.add_argument(
        "--esmc",
        action="store_true",
        help="Also load ESM-C-600M and run a forward pass (downloads weights on first use).",
    )
    parser.add_argument(
        "--verify-weights",
        action="store_true",
        help="With --esmc, also hash the state dict and compare it to the pinned digest.",
    )
    args = parser.parse_args(argv)

    requirement = DeviceRequirement(min_memory_gib=args.min_memory_gib)

    _print_build()
    _print_inventory(requirement)

    try:
        gpu = select_gpu(requirement)
    except DeviceSelectionError as exc:
        print(f"[{FAIL}] device selection\n")
        print(exc)
        return 1

    import torch

    device = torch.device(f"cuda:{gpu.index}")
    print(f"[{OK}] selected {gpu.describe()}")

    results = [
        _check_vram(device),
        _check_bf16_matmul(device),
        _check_esmc(device, args.esmc, verify_weights=args.verify_weights),
    ]
    print()

    if any(result is False for result in results):
        print("doctor: FAILED - see above.")
        return 1
    print("doctor: environment is fit for GPU training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
