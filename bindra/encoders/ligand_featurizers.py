"""Per-arm ligand featurization, in a form a process pool can carry.

``PairDataset.build_batch`` hands ligand featurization to a process pool,
because it is 72% of the batch build and GIL-bound inside RDKit's wrappers --
threads measured at 1.1x over serial, so eight cores went unused. That pool is
the constraint this module is shaped around: **whatever it maps must be
picklable**, which rules out a bound method, a closure and a lambda. So the
featurizers here are module-level functions, and
:class:`LigandBatchBuilder` carries only a string naming which one to use.

## What crosses the process boundary

Only SMILES go out. For the GAT arm a small graph comes back; for KPGT a graph
plus a 512-bit fingerprint and 200 descriptors, about 3 KiB a molecule. Both are
negligible against the protein tensor, which is ~144 MiB a batch and
deliberately never crosses -- it is read from the memory-mapped cache in the
parent.

Putting KPGT's descriptors **in the pool** is the whole point rather than an
incidental: they are the expensive part of that arm (~200 RDKit descriptors per
molecule, more than the graph build), they are a deterministic function of the
SMILES, and the pool is already there for exactly this kind of work. Whether the
pool is *enough* or a cache is still needed is a measurement, not a guess, and
it has not been made yet.

## The per-process generator

``RDKit2DNormalized`` is constructed once per worker process and reused, via a
module-level handle. Building it per molecule would add setup to every call;
building it once in the parent would not help, because the children do the work.
Each process gets its own, which is correct -- it is a calculator, not shared
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

#: Built lazily, once per worker process. See the module docstring.
_DESCRIPTORS: Any = None


def _descriptor_generator() -> Any:
    global _DESCRIPTORS
    if _DESCRIPTORS is None:
        from bindra.encoders.kpgt_descriptors import RDKit2DNormalized

        _DESCRIPTORS = RDKit2DNormalized()
    return _DESCRIPTORS


@dataclass(frozen=True)
class KPGTMoleculeParts:
    """One molecule's KPGT inputs: the triplet graph and both knowledge vectors.

    Returned as one object so a graph cannot be separated from its own
    fingerprint and descriptors on the way back from a worker. Holds numpy
    arrays rather than tensors: torch tensors pickle through a shared-memory
    path that is slower for small payloads and needs the file-descriptor
    strategy to survive a pool.
    """

    graph: Any  # KPGTGraph
    fingerprint: Any  # np.ndarray [512] float32
    descriptors: Any  # np.ndarray [200] float32


def featurize_gat_one(smiles: str) -> Any:
    """One molecule's atom-level graph. Picklable; maps in the pool."""
    from bindra.encoders.ligand_graph import featurize_molecule

    return featurize_molecule(smiles)


def featurize_kpgt_one(smiles: str) -> KPGTMoleculeParts:
    """One molecule's triplet graph plus both knowledge vectors.

    NaNs from the descriptor generator are zeroed here rather than downstream:
    a NaN reaching the knowledge node propagates through every attention weight
    in the batch, and the molecule that caused it would be unidentifiable by
    then.
    """
    import numpy as np

    from bindra.encoders.kpgt_descriptors import N_DESCRIPTORS
    from bindra.encoders.kpgt_graph import (
        DEFAULT_POLICY,
        featurize_kpgt,
        rdk_fingerprint,
    )
    from bindra.encoders.ligand_arm import LigandArmError

    values = np.asarray(_descriptor_generator().process(smiles)[1:], dtype=np.float32)
    if values.shape[0] != N_DESCRIPTORS:
        raise LigandArmError(
            f"descriptor generator produced {values.shape[0]} values for "
            f"{smiles!r}, expected {N_DESCRIPTORS}"
        )
    return KPGTMoleculeParts(
        graph=featurize_kpgt(smiles, DEFAULT_POLICY),
        fingerprint=rdk_fingerprint(smiles, DEFAULT_POLICY),
        descriptors=np.nan_to_num(values),
    )


def _collate_gat(parts: Sequence[Any]) -> Any:
    from bindra.encoders.ligand_graph import collate_graphs

    return collate_graphs(list(parts))


def _collate_kpgt(parts: Sequence[KPGTMoleculeParts]) -> Any:
    import numpy as np
    import torch

    from bindra.encoders.kpgt_graph import collate_kpgt
    from bindra.encoders.ligand_kpgt_arm import KPGTLigandBatch

    return KPGTLigandBatch(
        graphs=collate_kpgt([p.graph for p in parts]),
        fingerprints=torch.from_numpy(
            np.stack([p.fingerprint for p in parts])
        ).to(torch.float32),
        descriptors=torch.from_numpy(
            np.stack([p.descriptors for p in parts])
        ).to(torch.float32),
    )


@dataclass(frozen=True)
class LigandBatchBuilder:
    """Which arm's featurizer and collate to use. A string, so it pickles.

    Holding the arm *name* rather than the functions keeps this trivially
    picklable and keeps the dispatch in one place. ``for_config`` derives it from
    the same config object the model is built from, so a run cannot featurize for
    one arm and encode with the other.
    """

    arm: str = "gat"

    def __post_init__(self) -> None:
        if self.arm not in ("gat", "kpgt"):
            raise ValueError(
                f"unknown ligand arm {self.arm!r}; expected 'gat' or 'kpgt'"
            )

    @classmethod
    def for_config(cls, config: object) -> "LigandBatchBuilder":
        """Derive the arm from a ``GATConfig`` or ``KPGTConfig``.

        Dispatches on type rather than on a flag so that the model and the data
        pipeline cannot disagree about which arm a run uses -- they are handed
        the same object.
        """
        from bindra.encoders.ligand_gat import GATConfig

        if isinstance(config, GATConfig):
            return cls("gat")
        try:
            from bindra.encoders.ligand_kpgt import KPGTConfig
        except ImportError:  # pragma: no cover - KPGT is optional
            KPGTConfig = ()  # type: ignore[assignment]
        if KPGTConfig and isinstance(config, KPGTConfig):
            return cls("kpgt")
        raise ValueError(
            f"no ligand featurizer for config type {type(config).__name__}; "
            "expected GATConfig or KPGTConfig"
        )

    @property
    def featurize(self) -> Callable[[str], Any]:
        """The per-molecule function, as a picklable module-level reference."""
        return featurize_gat_one if self.arm == "gat" else featurize_kpgt_one

    def collate(self, parts: Sequence[Any]) -> Any:
        return _collate_gat(parts) if self.arm == "gat" else _collate_kpgt(parts)

    @property
    def featurization_errors(self) -> tuple[type[BaseException], ...]:
        """What a bad molecule raises, per arm, so the caller can say which.

        The two arms raise different exception types from different featurizers;
        catching only the GAT one would let a KPGT failure escape as an opaque
        traceback from inside a worker process.
        """
        from bindra.encoders.ligand_graph import LigandGraphError

        if self.arm == "gat":
            return (LigandGraphError,)
        from bindra.encoders.kpgt_graph import KPGTGraphError
        from bindra.encoders.ligand_arm import LigandArmError

        return (KPGTGraphError, LigandArmError, LigandGraphError)


__all__ = [
    "KPGTMoleculeParts",
    "LigandBatchBuilder",
    "featurize_gat_one",
    "featurize_kpgt_one",
]
