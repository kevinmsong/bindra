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

#: Per-process handle on the knowledge cache, opened on first use. A memmap is
#: cheap to open per worker and the pool is long-lived, so this costs one open
#: per process rather than one per batch.
_KNOWLEDGE: Any = None
_KNOWLEDGE_DIR: Any = None


def _knowledge_cache(directory: str):
    global _KNOWLEDGE, _KNOWLEDGE_DIR
    if _KNOWLEDGE is None or _KNOWLEDGE_DIR != directory:
        from bindra.features.knowledge_cache import open_cache

        _KNOWLEDGE = open_cache(directory)
        _KNOWLEDGE_DIR = directory
    return _KNOWLEDGE


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


def featurize_kpgt_one(smiles: str, cache_dir: str | None = None) -> KPGTMoleculeParts:
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

    # The cached half, when a cache is configured and holds this molecule.
    # Descriptors are 81% of this function; the graph below is the rest.
    if cache_dir:
        from bindra.features.knowledge_cache import (
            knowledge_cache_key,
            policy_digest,
            unpack,
        )

        key = knowledge_cache_key(smiles, policy_digest(DEFAULT_POLICY))
        cache = _knowledge_cache(cache_dir)
        if key in cache:
            fingerprint, descriptors = unpack(cache.get(key))
            return KPGTMoleculeParts(
                graph=featurize_kpgt(smiles, DEFAULT_POLICY),
                fingerprint=fingerprint.numpy(),
                descriptors=descriptors.numpy(),
            )

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

    arm: str = "kpgt"
    #: Directory of the precomputed fingerprint/descriptor cache, or None to
    #: compute them live. A plain string, so the builder stays picklable.
    knowledge_cache: str | None = None

    def __post_init__(self) -> None:
        if self.arm != "kpgt":
            raise ValueError(f"unknown ligand arm {self.arm!r}; expected 'kpgt'")

    @classmethod
    def for_config(cls, config: object) -> "LigandBatchBuilder":
        """Derive the arm from a ``KPGTConfig``.

        Dispatches on type rather than on a flag so that the model and the data
        pipeline cannot disagree about which arm a run uses -- they are handed
        the same object.
        """
        from bindra.encoders.ligand_kpgt import KPGTConfig

        if isinstance(config, KPGTConfig):
            return cls("kpgt")
        raise ValueError(
            f"no ligand featurizer for config type {type(config).__name__}; "
            "expected KPGTConfig"
        )

    @property
    def featurize(self) -> Callable[[str], Any]:
        """The per-molecule function, picklable for the pool.

        ``functools.partial`` over a module-level function with a string
        argument pickles by qualified name, so the cache path survives the trip
        into a worker where a closure would not.
        """
        if self.knowledge_cache:
            import functools

            return functools.partial(featurize_kpgt_one, cache_dir=self.knowledge_cache)
        return featurize_kpgt_one

    def collate(self, parts: Sequence[Any]) -> Any:
        return _collate_kpgt(parts)

    @property
    def featurization_errors(self) -> tuple[type[BaseException], ...]:
        """What a bad molecule raises, per arm, so the caller can say which.

        The two arms raise different exception types from different featurizers;
        catching only the GAT one would let a KPGT failure escape as an opaque
        traceback from inside a worker process.
        """
        from bindra.encoders.kpgt_graph import KPGTGraphError
        from bindra.encoders.ligand_arm import LigandArmError

        return (KPGTGraphError, LigandArmError)


__all__ = [
    "KPGTMoleculeParts",
    "LigandBatchBuilder",
    "featurize_kpgt_one",
]
