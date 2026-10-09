"""GPU discovery and selection.

Two failure modes motivate this module, both observed in the legacy
DeepDTA-iBAM project this one is ported from:

1. The legacy environment ran ``torch 2.9.1+cpu``. Every run silently fell back
   to CPU and nothing said so. Selection here raises instead of falling back,
   unless CPU is asked for explicitly.

2. This machine has two NVIDIA GPUs: an RTX 5060 Ti (16 GB, Blackwell sm_120)
   and a T1200 Laptop GPU (4 GB, Turing sm_75). CUDA's default device order is
   not nvidia-smi's, so ``cuda:0`` is not reliably the big card. The legacy
   engine hardcoded ``cuda:0`` and then wrapped the model in ``DataParallel``
   over every visible device, which would shard batches across a 16 GB and a
   4 GB card. Selection here is by capability and memory, never by bare index.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import torch

# CUDA_DEVICE_ORDER must be set before the CUDA context is created for it to
# take effect, which is why bindra/__init__.py sets it at import time. Re-assert
# it here so that importing this module directly is still correct.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

BYTES_PER_GIB = 1024**3

#: The RTX 5060 Ti is Blackwell. Anything below sm_120 on this machine is the
#: 4 GB T1200, which cannot hold the training workload.
DEFAULT_MIN_CAPABILITY: tuple[int, int] = (12, 0)
DEFAULT_MIN_MEMORY_GIB: float = 15.0


class DeviceSelectionError(RuntimeError):
    """No device satisfied the stated requirement.

    Carries a rendered inventory of what was actually found so the caller does
    not have to re-derive it.
    """


@dataclass(frozen=True)
class GpuInfo:
    """What we know about one visible CUDA device."""

    index: int
    name: str
    uuid: str | None
    total_memory_bytes: int
    capability: tuple[int, int]

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / BYTES_PER_GIB

    @property
    def sm(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"

    def describe(self) -> str:
        uuid = self.uuid or "unknown-uuid"
        return (
            f"cuda:{self.index}  {self.name}  "
            f"{self.total_memory_gib:.1f} GiB  {self.sm}  {uuid}"
        )


@dataclass(frozen=True)
class DeviceRequirement:
    """What a workload needs from a device.

    ``allow_cpu`` is deliberately opt-in and deliberately loud. Falling back to
    CPU changes a multi-hour GPU job into a multi-week one; that should never
    happen because a driver hiccup made ``is_available()`` return False.
    """

    min_memory_gib: float = DEFAULT_MIN_MEMORY_GIB
    min_capability: tuple[int, int] = DEFAULT_MIN_CAPABILITY
    require_uuid: str | None = None
    require_name_contains: str | None = None
    allow_cpu: bool = False

    def describe(self) -> str:
        parts = [
            f">= {self.min_memory_gib:.1f} GiB",
            f">= sm_{self.min_capability[0]}{self.min_capability[1]}",
        ]
        if self.require_uuid:
            parts.append(f"uuid == {self.require_uuid}")
        if self.require_name_contains:
            parts.append(f"name contains {self.require_name_contains!r}")
        return ", ".join(parts)


def torch_build_info() -> dict[str, object]:
    """Version and build facts worth printing in every run log."""
    return {
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "is_cpu_only_build": torch.version.cuda is None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def list_cuda_devices() -> list[GpuInfo]:
    """Inventory every visible CUDA device. Empty list if CUDA is unusable."""
    if not torch.cuda.is_available():
        return []

    devices: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        # .uuid exists on recent torch builds; treat it as best-effort because
        # it is the stable identity we would rather pin runs to.
        raw_uuid = getattr(props, "uuid", None)
        devices.append(
            GpuInfo(
                index=index,
                name=props.name,
                uuid=str(raw_uuid) if raw_uuid is not None else None,
                total_memory_bytes=int(props.total_memory),
                capability=(int(props.major), int(props.minor)),
            )
        )
    return devices


def _unmet_reason(gpu: GpuInfo, requirement: DeviceRequirement) -> str | None:
    """Why this device fails the requirement, or None if it passes."""
    if gpu.capability < requirement.min_capability:
        want = requirement.min_capability
        return f"{gpu.sm} is below the required sm_{want[0]}{want[1]}"
    if gpu.total_memory_gib < requirement.min_memory_gib:
        return (
            f"{gpu.total_memory_gib:.1f} GiB is below the required "
            f"{requirement.min_memory_gib:.1f} GiB"
        )
    if requirement.require_uuid and gpu.uuid != requirement.require_uuid:
        return f"uuid {gpu.uuid} does not match the pinned {requirement.require_uuid}"
    if (
        requirement.require_name_contains
        and requirement.require_name_contains.lower() not in gpu.name.lower()
    ):
        return f"name {gpu.name!r} does not contain {requirement.require_name_contains!r}"
    return None


def _render_inventory(devices: Sequence[GpuInfo], requirement: DeviceRequirement) -> str:
    if not devices:
        return "  (no CUDA devices visible to torch)"
    lines = []
    for gpu in devices:
        reason = _unmet_reason(gpu, requirement)
        verdict = "OK" if reason is None else f"rejected: {reason}"
        lines.append(f"  {gpu.describe()}  -> {verdict}")
    return "\n".join(lines)


def select_gpu(requirement: DeviceRequirement | None = None) -> GpuInfo:
    """Pick the best device meeting ``requirement``, or raise.

    "Best" is the most memory, then the highest compute capability. Ties break
    on index only as a last resort, so selection is deterministic without ever
    being *driven* by index.
    """
    requirement = requirement or DeviceRequirement()
    info = torch_build_info()

    if info["is_cpu_only_build"]:
        raise DeviceSelectionError(
            f"This torch build has no CUDA support at all (torch {torch.__version__}, "
            f"torch.version.cuda is None).\n"
            "Install a CUDA 12.8+ build, which is what the RTX 5060 Ti (sm_120) needs:\n"
            "  uv pip install --index-url https://download.pytorch.org/whl/cu128 torch"
        )

    devices = list_cuda_devices()
    if not devices:
        raise DeviceSelectionError(
            f"torch {torch.__version__} was built against CUDA {torch.version.cuda}, but no CUDA "
            "device is visible. Check the driver and CUDA_VISIBLE_DEVICES "
            f"(currently {os.environ.get('CUDA_VISIBLE_DEVICES')!r})."
        )

    candidates = [gpu for gpu in devices if _unmet_reason(gpu, requirement) is None]
    if not candidates:
        raise DeviceSelectionError(
            f"No visible GPU meets the requirement ({requirement.describe()}).\n"
            f"{_render_inventory(devices, requirement)}\n"
            "Pass allow_cpu=True only if a slow CPU run is genuinely what you want."
        )

    chosen = max(candidates, key=lambda gpu: (gpu.total_memory_bytes, gpu.capability, -gpu.index))

    # A device can be visible and still unsupported by this build: a torch wheel
    # compiled without sm_120 kernels sees the 5060 Ti but cannot run on it.
    arch_list = info["arch_list"]
    assert isinstance(arch_list, list)
    if arch_list and not _arch_supported(chosen, arch_list):
        raise DeviceSelectionError(
            f"{chosen.name} reports {chosen.sm}, but this torch build only ships kernels for "
            f"{', '.join(arch_list)}.\n"
            "Reinstall torch from the cu128 index so sm_120 kernels are present."
        )

    return chosen


def _arch_supported(gpu: GpuInfo, arch_list: Sequence[str]) -> bool:
    """Does the build ship kernels this device can run?

    An exact ``sm_120`` match counts. So does a ``compute_NN`` PTX entry at or
    below the device's capability, since PTX is JIT-compiled forward.
    """
    if gpu.sm in arch_list:
        return True
    device_cc = gpu.capability[0] * 10 + gpu.capability[1]
    for arch in arch_list:
        if not arch.startswith("compute_"):
            continue
        try:
            arch_cc = int(arch.removeprefix("compute_"))
        except ValueError:
            continue
        if arch_cc <= device_cc:
            return True
    return False


def describe_device(device) -> str:
    """``cuda:0 (NVIDIA GeForce RTX 5060 Ti, 14.8 of 15.9 GiB free)``.

    The index alone is not an identity. torch enumerates FASTEST_FIRST while
    nvidia-smi goes by PCI bus, so the same number names different cards in the
    two tools, and the order is not stable across runs. A log that records only
    "cuda:1" cannot answer which GPU produced a number, or why a run that fitted
    yesterday is out of memory today.
    """
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return str(device)
    index = device.index if getattr(device, "index", None) is not None else 0
    try:
        name = torch.cuda.get_device_properties(index).name
        free, total = torch.cuda.mem_get_info(index)
        return (
            f"cuda:{index} ({name}, {free / 2**30:.1f} of "
            f"{total / 2**30:.1f} GiB free)"
        )
    except Exception:  # noqa: BLE001 - a log line must never end a run
        return f"cuda:{index}"


def select_device(requirement: DeviceRequirement | None = None) -> torch.device:
    """``select_gpu`` as a ``torch.device``, honouring ``allow_cpu``."""
    requirement = requirement or DeviceRequirement()
    try:
        return torch.device(f"cuda:{select_gpu(requirement).index}")
    except DeviceSelectionError:
        if requirement.allow_cpu:
            return torch.device("cpu")
        raise


__all__ = [
    "DEFAULT_MIN_CAPABILITY",
    "DEFAULT_MIN_MEMORY_GIB",
    "DeviceRequirement",
    "DeviceSelectionError",
    "GpuInfo",
    "list_cuda_devices",
    "describe_device",
    "select_device",
    "select_gpu",
    "torch_build_info",
]
