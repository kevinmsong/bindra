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


def _check_esmc(device, enabled: bool) -> bool | None:
    """Load ESM-C-600M and embed a short sequence. Returns None when skipped."""
    if not enabled:
        print(f"[{SKIP}] ESM-C forward pass (use --esmc to run it)")
        return None

    import torch

    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError as exc:
        print(f"[{SKIP}] ESM-C not installed ({exc}); install the 'encoders' extra")
        return None

    try:
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        model = ESMC.from_pretrained("esmc_600m").to(device).eval()
        tokenizer = EsmSequenceTokenizer()
        tokens = torch.tensor([tokenizer.encode(sequence)], device=device)
        with torch.inference_mode():
            output = model(tokens)
        embeddings = output.embeddings
        print(
            f"[{OK}] ESM-C-600M forward pass: {tuple(embeddings.shape)} "
            f"({embeddings.dtype}) for a {len(sequence)}-residue sequence"
        )
        if embeddings.shape[-1] != 1152:
            print(
                f"[{FAIL}] expected a 1152-dim ESM-C-600M embedding, got {embeddings.shape[-1]}"
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        print(f"[{FAIL}] ESM-C forward pass failed: {type(exc).__name__}: {exc}")
        return False


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
        _check_esmc(device, args.esmc),
    ]
    print()

    if any(result is False for result in results):
        print("doctor: FAILED — see above.")
        return 1
    print("doctor: environment is fit for GPU training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
