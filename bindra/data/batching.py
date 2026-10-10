"""Pair batching: cached protein features, ligand graphs and censored targets.

The whole design is forced by one measurement. Protein lengths in this corpus
span a median of 454 residues, a p99 of 2,780 and a **maximum of 34,350**. A
fixed batch size padded to the longest member is therefore unusable: 64 pairs
padded to 34,350 residues is **5 GiB of protein features alone**, before any
activation. So batches are formed under a **residue budget**
(``batch_size x padded_length``) rather than a pair count, which bounds the
padded tensor exactly regardless of what lands in a batch.

To stop that budget being spent on padding, records are sorted by protein length
inside a shuffle window before batches are cut. The window is what keeps this
from becoming length-ordered training: records are shuffled, chunked, sorted
*within* each chunk, and the resulting batches shuffled again, so a batch is
length-homogeneous while the epoch is not length-ordered.

## What this module does not do

It does not group by target. That is deliberate and worth stating, because a
trainable protein encoder would *require* it: re-encoding a distinct protein per
observation costs 54.6 hours per epoch, against 0.4 with one target per batch.
With the encoder frozen and its features cached, a lookup is a memmap read, so
random batching is free and the statistical cost of target-grouped batches --
correlated gradients, interaction with the comparability groups -- is avoided
entirely. If ESM-C is ever unfrozen, this module needs a target-grouped sampler
and that trade-off has to be faced.

## Nothing here invents a label

Targets are built through :class:`~bindra.training.losses.affinity.AffinityTargets`,
which refuses a censored row with no bound, a row with both bounds, and an exact
row carrying one. Tier B and other non-primary rows travel with
``is_primary=False`` and reach only the auxiliary term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from bindra.data.assay import ASSAY_COLUMNS, UNKNOWN, encode_assay
from bindra.encoders.ligand_featurizers import LigandBatchBuilder
from bindra.features.cache import ProteinFeatureCache
from bindra.features.types import EncoderMetadata
from bindra.training.losses.affinity import AffinityTargets

#: Columns the loader needs from the curated observations table.
OBSERVATION_COLUMNS = (
    "observation_id",
    "ligand_ref",
    "sequence_sha256",
    "endpoint",
    "p_value",
    "is_exact",
    "lower_bound",
    "upper_bound",
    "ic50_tier",
    "primary_regression_eligible",
    # Read for the assay-condition conditioning. Descriptive only: see
    # bindra.data.assay.EXCLUDED_AS_LEAKAGE for what is deliberately absent.
    "assay_condition_bucket",
    "metadata_confidence",
    "curation_source_class",
    "comparability_scope",
    "has_ph",
    "has_temperature",
    # Read for the within-assay ranking term. BindingDB has no assay
    # identifier, so `comparability_group_id` is the curation layer's proxy and
    # the only thing that distinguishes a within-assay comparison from a
    # cross-experiment one.
    "comparability_group_id",
    "ranking_eligible",
)


class BatchingError(ValueError):
    """The batch could not be formed from what the corpus and caches hold."""


@dataclass(frozen=True)
class BatchingPolicy:
    """Everything that decides batch composition, declared rather than inline."""

    #: ``batch_pairs x padded_residues``. 65,536 is 64 pairs at 1,024 residues,
    #: which covers the 90th percentile of protein length at a useful batch size
    #: and bounds the padded protein tensor at ~144 MiB in bf16.
    residue_budget: int = 65_536
    #: Upper bound on pairs per batch, for short proteins where the residue
    #: budget alone would allow a batch so large the ligand side dominates.
    max_pairs: int = 256
    #: Records are sorted by length inside a window of this many, so batches are
    #: length-homogeneous without the epoch becoming length-ordered.
    shuffle_window: int = 8_192
    seed: int = 20261001
    #: Padded protein length is rounded up to a multiple of this. The point is
    #: allocator fragmentation, not padding: without it every batch has its own
    #: ``padded_residues``, so the run produces thousands of distinct tensor
    #: shapes, the CUDA caching allocator reserves a segment per shape and
    #: eventually cannot serve a large request while still showing free memory.
    #: That is what killed the first real run, 63% into epoch 0, after 14,000
    #: batches. ``expandable_segments:True``, the usual remedy, is **not
    #: supported on Windows**.
    #:
    #: 128 turns ~34,000 possible lengths into ~268 and costs ~8% more padding.
    #: 0 disables the rounding.
    length_multiple: int = 128
    #: Proteins longer than this are **reported and excluded**, never truncated.
    #: ``None`` keeps every construct, which is the default: a 34,350-residue
    #: protein simply gets a batch to itself.
    max_protein_length: int | None = None
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.residue_budget < 1:
            raise BatchingError(f"residue_budget must be >= 1, got {self.residue_budget}")
        if self.max_pairs < 1:
            raise BatchingError(f"max_pairs must be >= 1, got {self.max_pairs}")
        if self.shuffle_window < 1:
            raise BatchingError(f"shuffle_window must be >= 1, got {self.shuffle_window}")
        if self.length_multiple < 0:
            raise BatchingError(
                f"length_multiple must be >= 0, got {self.length_multiple}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "residue_budget": self.residue_budget,
            "max_pairs": self.max_pairs,
            "shuffle_window": self.shuffle_window,
            "seed": self.seed,
            "length_multiple": self.length_multiple,
            "max_protein_length": self.max_protein_length,
            "drop_last": self.drop_last,
        }


def _round_up(length: int, multiple: int) -> int:
    """Round a padded length up to a multiple, to bound shape diversity."""
    if multiple <= 1:
        return int(length)
    return int(-(-int(length) // multiple) * multiple)


def length_bucketed_batches(
    lengths: Sequence[int],
    policy: BatchingPolicy,
    *,
    epoch: int = 0,
    shuffle: bool = True,
) -> list[list[int]]:
    """Group record indices into batches that respect the residue budget.

    Returns lists of indices into ``lengths``. Every index appears exactly once:
    a record longer than the whole budget is given a batch of its own rather
    than dropped, because silently discarding the longest proteins would remove
    exactly the targets least represented elsewhere.

    ``epoch`` advances the shuffle so successive epochs see different groupings
    while each remains reproducible from ``(seed, epoch)``.
    """
    count = len(lengths)
    if count == 0:
        return []
    order = np.arange(count)
    if shuffle:
        np.random.default_rng([policy.seed, epoch]).shuffle(order)

    batches: list[list[int]] = []
    for start in range(0, count, policy.shuffle_window):
        window = order[start : start + policy.shuffle_window]
        # Sort within the window so a batch's members have similar lengths and
        # the budget buys real residues instead of padding.
        window = window[np.argsort([lengths[i] for i in window], kind="stable")]

        current: list[int] = []
        longest = 0
        for index in window:
            candidate = _round_up(max(longest, lengths[index]), policy.length_multiple)
            fits = (
                len(current) + 1 <= policy.max_pairs
                and candidate * (len(current) + 1) <= policy.residue_budget
            )
            if current and not fits:
                batches.append(current)
                current, longest = [int(index)], lengths[index]
            else:
                current.append(int(index))
                longest = candidate
        if current:
            batches.append(current)

    if shuffle:
        np.random.default_rng([policy.seed, epoch, 1]).shuffle(batches)  # type: ignore[arg-type]
    if policy.drop_last and len(batches) > 1:
        batches = batches[:-1]
    return batches


@dataclass(frozen=True)
class PairBatch:
    """One training batch. Protein features are cached; ligand graphs are live."""

    protein_features: torch.Tensor  # [B, L, dp] bf16 on the host, float32 on device
    protein_mask: torch.Tensor  # [B, L] bool, True = real residue
    #: Whatever the configured arm collates: a ``KPGTLigandBatch`` today.
    #: Typed loosely because the batch type *is*
    #: what differs between arms; a consumer must go through the arm rather
    #: than assume atoms. Both expose ``num_graphs`` and ``to``.
    ligands: object
    targets: AffinityTargets
    observation_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    #: ``[B, ASSAY_DIM]`` recorded conditions, or None when the run does not
    #: condition on them. None rather than zeros: a zero vector is a valid
    #: encoding and would read as a real set of conditions.
    assay: "torch.Tensor | None" = None

    @property
    def size(self) -> int:
        return int(self.protein_features.shape[0])

    @property
    def padded_residues(self) -> int:
        return int(self.protein_features.shape[1])

    @property
    def padding_fraction(self) -> float:
        """Share of the protein tensor that is padding. A batching diagnostic."""
        total = self.size * self.padded_residues
        return 0.0 if not total else 1.0 - float(self.protein_mask.sum()) / total

    def to(
        self, device: torch.device | str, *, dtype: torch.dtype = torch.float32
    ) -> "PairBatch":
        """Move to device, upcasting the protein stream there rather than on the CPU."""
        return PairBatch(
            protein_features=self.protein_features.to(device, dtype=dtype),
            protein_mask=self.protein_mask.to(device),
            ligands=self.ligands.to(device),  # type: ignore[attr-defined]
            targets=self.targets.to(device),
            observation_ids=self.observation_ids,
            smiles=self.smiles,
            assay=None if self.assay is None else self.assay.to(device, dtype=dtype),
        )



def dataset_cache_key(
    processed: "Path | str",
    split_directory: "Path | str",
    partition: str,
    metadata: "EncoderMetadata",
    limit: "int | None",
) -> str:
    """Content digest of everything the assembled dataset depends on.

    Every input is a CONTENT hash, not a path or a timestamp: the combined
    sha256 of the observation shards, the split manifest's own sha256, and the
    encoder contract. A rebuilt corpus or a rebuilt split changes the key and
    the cache is simply missed, rather than serving rows that no longer match.

    The encoder metadata is in here because ``_unique_keys`` is derived from
    ``metadata.cache_key(sequence)`` -- a different protein cache means
    different keys for the same observations.
    """
    import hashlib
    import json as _json

    payload = _json.loads(
        (Path(processed) / "qc_report.json").read_text(encoding="utf-8")
    )
    observations = payload["artifacts"]["observations"]
    manifest = _json.loads(
        (Path(split_directory) / "manifest.json").read_text(encoding="utf-8")
    )
    parts = {
        "schema": "pair-dataset-v1",
        "observations_sha256": observations.get("combined_sha256"),
        "observation_rows": observations.get("rows"),
        "columns": list(OBSERVATION_COLUMNS),
        "split_manifest_sha256": manifest.get("manifest_sha256"),
        "split_id": manifest.get("split_id"),
        "partition": partition,
        "limit": limit,
        "encoder": metadata.to_dict() if hasattr(metadata, "to_dict") else str(metadata),
    }
    return hashlib.sha256(
        _json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


class PairDataset:
    """The curated observations of one split partition, joined to their inputs.

    Loads once into memory: 1.9M training rows of the ten columns needed, plus
    SMILES and sequence hashes, is a few hundred MB. The 14 GiB of protein
    features stays on disk and is memory-mapped per batch.
    """

    def __init__(
        self,
        processed: Path | str,
        split_directory: Path | str,
        partition: str,
        *,
        protein_cache: ProteinFeatureCache,
        limit: int | None = None,
        length_multiple: int = 128,
        ligand_builder: LigandBatchBuilder | None = None,
        with_assay: bool = False,
        cache_dir: "Path | str | None" = None,
    ) -> None:
        import pandas as pd

        #: Whether each batch carries its recorded assay conditions. Off by
        #: default so an existing checkpoint's loader is unchanged. Named
        #: differently from ``encode_assay`` on purpose: the flag and the
        #: function it gates would otherwise shadow each other.
        self.with_assay = with_assay

        #: Must match the policy the loader uses, or the budget check and the
        #: actual padding disagree.
        self.length_multiple = length_multiple
        #: Which arm's featurizer the pool maps. The default is the only arm
        #: that still exists; passing the model's own ligand config through
        #: ``LigandBatchBuilder.for_config`` is what keeps a run consistent, and
        #: is also where the knowledge cache is attached -- without it the 200
        #: descriptors are recomputed per molecule, which is 81% of the work.
        self.ligand_builder = ligand_builder or LigandBatchBuilder()

        from bindra.data.splits import read_split
        from bindra.data.tables import ShardedTableRef
        import json

        self.processed = Path(processed)
        self.partition = partition
        self.protein_cache = protein_cache
        self._metadata: EncoderMetadata = protein_cache.metadata

        # The assembled dataset is a pure function of the corpus, the split,
        # the partition, the limit and the encoder contract, and assembling it
        # means a full pass over 2.7M rows plus two joins -- about ten minutes,
        # repeated identically on every run. Keyed on content hashes, so a
        # rebuilt corpus or split misses the cache rather than serving stale
        # rows.
        self._cache_key = None
        if cache_dir is not None:
            self._cache_key = dataset_cache_key(
                self.processed, split_directory, partition, self._metadata, limit
            )
            if self._load_cached(Path(cache_dir) / self._cache_key):
                return

        _, assignment = read_split(Path(split_directory))
        wanted = {o for o, p in assignment.items() if p == partition}
        if not wanted:
            raise BatchingError(
                f"partition {partition!r} is empty in {split_directory}"
            )

        payload = json.loads(
            (self.processed / "qc_report.json").read_text(encoding="utf-8")
        )
        reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
        frames = []
        for frame in reference.iter_frames(columns=list(OBSERVATION_COLUMNS)):
            frames.append(frame[frame["observation_id"].isin(wanted)])
        observations = pd.concat(frames, ignore_index=True)
        if limit is not None:
            observations = observations.head(limit)

        ligands = pd.concat(
            [
                pd.read_parquet(path, columns=["ligand_ref", "standardized_smiles", "status"])
                for path in sorted((self.processed / "ligands").glob("*.parquet"))
            ],
            ignore_index=True,
        )
        ligands = ligands[ligands["status"] == "OK"].set_index("ligand_ref")
        observations = observations.join(
            ligands["standardized_smiles"], on="ligand_ref", how="inner"
        )

        constructs = pd.read_parquet(
            self.processed / "target_constructs.parquet",
            columns=["sequence_ref", "sequence_sha256", "status"],
        )
        constructs = constructs[constructs["status"] == "OK"]
        sequences = pd.read_parquet(
            self.processed / "sequences_raw.parquet",
            columns=["sequence_ref", "sequence", "sequence_length"],
        ).set_index("sequence_ref")
        constructs = constructs.join(sequences, on="sequence_ref").drop_duplicates(
            "sequence_sha256"
        )
        # Join only the *length* per observation, never the sequence itself.
        # There are 10,820 distinct constructs behind 2.7M observations, and a
        # construct averages ~500 residues: broadcasting the sequence string
        # across the observation axis would carry roughly a gigabyte of
        # duplicated text for information that 10,820 rows already hold.
        observations = observations.merge(
            constructs[["sequence_sha256", "sequence_length"]],
            on="sequence_sha256",
            how="inner",
        )
        self.frame = observations.reset_index(drop=True)
        self.lengths = self.frame["sequence_length"].to_numpy().astype(int)

        # One cache key per distinct construct, then an integer per observation
        # pointing at it. Hashing is over the sequence and the encoder contract,
        # so it is far too expensive to redo per batch and pointless to redo per
        # observation.
        unique = constructs.drop_duplicates("sequence_sha256")
        key_by_hash = {
            digest: self._metadata.cache_key(sequence)
            for digest, sequence in zip(unique["sequence_sha256"], unique["sequence"])
        }
        self._unique_keys = list(dict.fromkeys(key_by_hash.values()))
        position = {key: index for index, key in enumerate(self._unique_keys)}
        self._key_index = np.fromiter(
            (position[key_by_hash[d]] for d in self.frame["sequence_sha256"]),
            dtype=np.int32,
            count=len(self.frame),
        )
        if cache_dir is not None and self._cache_key:
            self._save_cached(Path(cache_dir) / self._cache_key)

    # -- the disk cache --------------------------------------------------

    def _load_cached(self, directory: Path) -> bool:
        """Restore a previously assembled dataset. False if there is none.

        A failure to read is treated as a miss, never as an error: the
        expensive path still works, and a half-written cache from a killed run
        must not stop training.
        """
        import json

        import pandas as pd

        marker = directory / "dataset.json"
        if not marker.exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.frame = pd.read_parquet(directory / "frame.parquet")
            self.lengths = self.frame["sequence_length"].to_numpy().astype(int)
            self._unique_keys = payload["unique_keys"]
            self._key_index = np.load(directory / "key_index.npy")
            if len(self._key_index) != len(self.frame):
                raise ValueError("key index and frame disagree on length")
        except Exception as exc:  # noqa: BLE001 - a bad cache is a miss, not a crash
            print(
                f"[ warn  ] dataset cache at {directory.name} unreadable "
                f"({type(exc).__name__}); rebuilding",
                flush=True,
            )
            return False
        print(
            f"[  ok   ] dataset cache hit: {len(self.frame):,} {self.partition} "
            f"rows restored without a corpus pass",
            flush=True,
        )
        return True

    def _save_cached(self, directory: Path) -> None:
        """Write the assembled dataset. A failure here must not stop a run."""
        import json

        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.frame.to_parquet(directory / "frame.parquet", index=False)
            np.save(directory / "key_index.npy", self._key_index)
            # Written LAST: its presence is what marks the cache complete, so
            # a run killed mid-write leaves a miss rather than a torn entry.
            (directory / "dataset.json").write_text(
                json.dumps({
                    "partition": self.partition,
                    "rows": int(len(self.frame)),
                    "unique_keys": list(self._unique_keys),
                }),
                encoding="utf-8",
            )
            print(
                f"[  ok   ] dataset cached to {directory.name} "
                f"({len(self.frame):,} rows); later runs skip the corpus pass",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ warn  ] could not write the dataset cache "
                f"({type(exc).__name__}: {exc}); continuing",
                flush=True,
            )

    @classmethod
    def from_queries(
        cls,
        processed: Path | str,
        queries,
        *,
        protein_cache: ProteinFeatureCache,
        length_multiple: int = 128,
        ligand_builder: LigandBatchBuilder | None = None,
        with_assay: bool = False,
    ) -> "PairDataset":
        """Build from a benchmark panel's **query** table -- no labels involved.

        A panel's queries carry ``identity_key``, ``sequence_sha256`` and
        ``endpoint``, which is everything needed to predict. The p-values sit in
        the panel's sealed directory and are not touched here; the batch's
        target fields are filled with placeholders that the prediction path
        never reads, purely so the batch keeps one shape throughout the codebase.
        """
        import numpy as np
        import pandas as pd

        processed = Path(processed)
        instance = cls.__new__(cls)
        instance.processed = processed
        instance.partition = "panel"
        instance.protein_cache = protein_cache
        instance._metadata = protein_cache.metadata
        instance.length_multiple = length_multiple
        # Set here rather than left to __init__, which this path bypasses via
        # __new__. Without it `build_batch` raises AttributeError on its first
        # batch -- latent until now only because no panel had ever been scored.
        instance.ligand_builder = ligand_builder or LigandBatchBuilder()
        instance.with_assay = with_assay

        ligands = pd.concat(
            [
                pd.read_parquet(
                    path, columns=["identity_key", "standardized_smiles", "status"]
                )
                for path in sorted((processed / "ligands").glob("*.parquet"))
            ],
            ignore_index=True,
        )
        ligands = ligands[ligands["status"] == "OK"].drop_duplicates("identity_key")
        frame = queries.merge(
            ligands[["identity_key", "standardized_smiles"]],
            on="identity_key",
            how="inner",
        )

        constructs = pd.read_parquet(
            processed / "target_constructs.parquet",
            columns=["sequence_ref", "sequence_sha256", "status"],
        )
        constructs = constructs[constructs["status"] == "OK"]
        sequences = pd.read_parquet(
            processed / "sequences_raw.parquet",
            columns=["sequence_ref", "sequence", "sequence_length"],
        ).set_index("sequence_ref")
        constructs = constructs.join(sequences, on="sequence_ref").drop_duplicates(
            "sequence_sha256"
        )
        frame = frame.merge(
            constructs[["sequence_sha256", "sequence_length"]],
            on="sequence_sha256",
            how="inner",
        )

        # Placeholders. `endpoint` is a query column and is real; the rest exist
        # only so AffinityTargets can be constructed, and the prediction path
        # reads nothing but endpoint_index.
        frame = frame.assign(
            p_value=0.0,
            is_exact=True,
            lower_bound=np.nan,
            upper_bound=np.nan,
            ic50_tier="na",
            primary_regression_eligible=True,
        )
        # A panel query records no assay conditions. Every assay column the
        # encoder needs is filled with a value outside its vocabulary, which
        # encodes as the explicit UNKNOWN category -- not as a plausible
        # condition the measurement never had.
        for column in ASSAY_COLUMNS:
            if column not in frame.columns:
                frame = frame.assign(**{column: UNKNOWN})
        if "observation_id" not in frame.columns:
            frame = frame.assign(
                observation_id=[f"q{i}" for i in range(len(frame))]
            )

        instance.frame = frame.reset_index(drop=True)
        instance.lengths = instance.frame["sequence_length"].to_numpy().astype(int)
        unique = constructs.drop_duplicates("sequence_sha256")
        key_by_hash = {
            digest: instance._metadata.cache_key(sequence)
            for digest, sequence in zip(unique["sequence_sha256"], unique["sequence"])
        }
        instance._unique_keys = list(dict.fromkeys(key_by_hash.values()))
        position = {key: index for index, key in enumerate(instance._unique_keys)}
        instance._key_index = np.fromiter(
            (position[key_by_hash[d]] for d in instance.frame["sequence_sha256"]),
            dtype=np.int32,
            count=len(instance.frame),
        )
        return instance

    def cache_key(self, row: int) -> str:
        """Protein cache key for one observation row."""
        return self._unique_keys[int(self._key_index[row])]

    def __len__(self) -> int:
        return len(self.frame)

    def attrition(self, requested: int) -> dict[str, int]:
        """What the joins dropped, so a silent shrink is visible."""
        return {"requested": requested, "retained": len(self.frame)}

    def build_batch(self, indices: Sequence[int], executor=None) -> PairBatch:
        """Assemble one batch. Raises rather than skipping a bad record.

        ``executor`` is a process pool used **only** for ligand featurization,
        which is 72% of this function and is GIL-bound Python inside RDKit's
        wrappers -- threads measured at 1.1x over serial, so the machine's eight
        cores went unused. Only the SMILES go out and small graph arrays come
        back (~1 MiB a batch for GAT, ~3 KiB a molecule for KPGT, whose
        descriptors are computed out there too); the protein tensor, which is
        ~144 MiB, never crosses a process boundary.

        Which featurizer the pool maps comes from ``self.ligand_builder``, so
        the arm is a configuration rather than a branch here.
        """
        if not indices:
            raise BatchingError("cannot build an empty batch")
        rows = self.frame.iloc[list(indices)]
        smiles = list(rows["standardized_smiles"])
        builder = self.ligand_builder
        featurize = builder.featurize

        try:
            if executor is not None and len(smiles) > 8:
                chunk = max(1, len(smiles) // (4 * (executor._max_workers or 1)))
                parts = list(executor.map(featurize, smiles, chunksize=chunk))
            else:
                parts = [featurize(s) for s in smiles]
        except builder.featurization_errors as exc:
            raise BatchingError(
                f"a curated molecule will not featurize for the "
                f"{builder.arm!r} arm: {exc}"
            ) from exc

        features = [
            self.protein_cache.get(self.cache_key(i)).residue_features for i in indices
        ]
        width = _round_up(
            max(int(f.shape[0]) for f in features), self.length_multiple
        )
        dim = int(features[0].shape[1])
        # Padded in the cache's own bf16, not upcast here. Converting on the CPU
        # cost 48 ms a batch and doubled the host-to-device transfer, since a
        # padded protein tensor at the residue budget is ~300 MiB in bf16 and
        # ~600 in fp32. The model's first op on this stream is a LayerNorm, which
        # promotes under autocast anyway, so the upcast happens on the GPU for
        # free. ``PairBatch.to`` is where it lands on device.
        padded = torch.zeros(len(features), width, dim, dtype=torch.bfloat16)
        mask = torch.zeros(len(features), width, dtype=torch.bool)
        for position, tensor in enumerate(features):
            length = int(tensor.shape[0])
            padded[position, :length] = tensor
            mask[position, :length] = True

        return PairBatch(
            protein_features=padded,
            protein_mask=mask,
            ligands=builder.collate(parts),
            targets=AffinityTargets.from_frame(rows),
            observation_ids=tuple(rows["observation_id"]),
            smiles=tuple(rows["standardized_smiles"]),
            assay=(
                torch.from_numpy(encode_assay(rows)) if self.with_assay else None
            ),
        )


class PairLoader:
    """Iterates a :class:`PairDataset` in residue-budgeted, length-bucketed batches."""

    def __init__(
        self,
        dataset: PairDataset,
        policy: BatchingPolicy | None = None,
        *,
        shuffle: bool = True,
        prefetch_workers: int = 3,
        featurize_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.policy = policy or BatchingPolicy()
        # The dataset pads and the policy budgets; they must round identically.
        self.dataset.length_multiple = self.policy.length_multiple
        self.shuffle = shuffle
        #: Processes featurizing ligands. Separate from ``prefetch_workers``
        #: because the two solve different problems: threads overlap the protein
        #: reads and the GPU, processes get past the GIL. 0 keeps it in-process.
        self.featurize_workers = featurize_workers
        #: First batch to actually build. A resumed epoch sets this so the
        #: batches already consumed are **skipped, not built and discarded** --
        #: assembling 25,504 of them to throw away costs about three hours.
        self.start_at = 0
        self._pool = None
        #: Threads building batches ahead of the consumer. 0 disables, which is
        #: what the tests use so failures surface in the calling thread.
        self.prefetch_workers = prefetch_workers
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the shuffle. Reproducible from ``(policy.seed, epoch)``."""
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        return length_bucketed_batches(
            self.dataset.lengths, self.policy, epoch=self.epoch, shuffle=self.shuffle
        )

    def __len__(self) -> int:
        return len(self.batches())

    def _executor(self):
        """Lazily start the featurization pool; reused across epochs."""
        if self.featurize_workers <= 0:
            return None
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(max_workers=self.featurize_workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __iter__(self) -> Iterator[PairBatch]:
        executor = self._executor()
        pending_batches = self.batches()[self.start_at :]
        if self.prefetch_workers <= 0:
            for indices in pending_batches:
                yield self.dataset.build_batch(indices, executor)
            return
        yield from self._prefetched(executor, pending_batches)

    def _prefetched(self, executor=None, batches=None) -> Iterator[PairBatch]:
        """Build batches on worker threads, a few ahead of the consumer.

        Threads, not processes, because a batch's padded protein tensor is
        ~144 MiB and pickling that across a pipe per step costs more than the
        overlap saves; processes would also reopen the 14 GiB memory-mapped
        cache per worker.

        **Measured honestly: the gain here is small.** An earlier version of this
        docstring asserted that RDKit featurization releases the GIL, so threads
        would overlap it fully. That was wrong — measured at 165 to 182 pairs/s
        for 0 to 2 workers, and *worse* above 2. Most of the build is
        GIL-bound Python in the per-atom feature loop.

        What actually fixed the imbalance was making the build cheaper: 438 ms
        to 275 ms per batch, by taking the per-bond tensor writes and the
        ring-size scan out of the hot path and leaving the protein stream in
        bf16 until it reaches the device. The CPU now produces 483 pairs/s
        against the GPU's 312, so the build is no longer the bottleneck and
        these threads only have to cover jitter.
        """
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque

        indices = self.batches() if batches is None else batches
        depth = max(2, self.prefetch_workers * 2)
        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            pending: deque = deque()
            upcoming = iter(indices)
            for batch_indices in upcoming:
                pending.append(
                    pool.submit(self.dataset.build_batch, batch_indices, executor)
                )
                if len(pending) >= depth:
                    break
            for batch_indices in upcoming:
                yield pending.popleft().result()
                pending.append(
                    pool.submit(self.dataset.build_batch, batch_indices, executor)
                )
            while pending:
                yield pending.popleft().result()

    def describe(self) -> dict[str, object]:
        """Batch-shape statistics, for the run log and for sanity."""
        batches = self.batches()
        sizes = [len(b) for b in batches]
        padded = [max(self.dataset.lengths[i] for i in b) for b in batches]
        real = [sum(self.dataset.lengths[i] for i in b) for b in batches]
        slots = [s * p for s, p in zip(sizes, padded)]
        return {
            "n_batches": len(batches),
            "n_pairs": int(sum(sizes)),
            "pairs_per_batch_mean": round(float(np.mean(sizes)), 1),
            "pairs_per_batch_min": int(min(sizes)),
            "pairs_per_batch_max": int(max(sizes)),
            "padded_residues_max": int(max(padded)),
            "residue_slots_max": int(max(slots)),
            "padding_fraction": round(1.0 - sum(real) / max(sum(slots), 1), 4),
            "policy": self.policy.to_dict(),
        }


__all__ = [
    "OBSERVATION_COLUMNS",
    "BatchingError",
    "BatchingPolicy",
    "PairBatch",
    "PairDataset",
    "PairLoader",
    "length_bucketed_batches",
]
