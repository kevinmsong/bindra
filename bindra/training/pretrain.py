"""Knowledge-guided pre-training for the LiGhT arm.

This is the missing letter. KPGT is *Knowledge-guided Pre-training of Graph
Transformer*; the port gave us the architecture and the knowledge inputs, and
``base.pth`` is unobtainable, so the only route to a pretrained LiGhT is to
pretrain one. The evidence that it matters is already on the board: LiGhT-mini
led the GAT arm for three epochs and then lost to it, which is what a
transformer with no pre-training looks like against a well-matched inductive
bias.

## The objective

Mask exactly one of a molecule's two knowledge nodes and reconstruct it from
the graph **plus the other**:

* fingerprint masked -> predict its 512 bits from the graph and the descriptors
* descriptors masked -> predict their 200 values from the graph and the
  fingerprint

Masking is the whole trick. The knowledge nodes are *inputs* during affinity
training, so predicting one while it is still visible is copying, not learning.
The masked node is replaced with the model's own ``mask_emb`` -- an embedding
this port has always carried and never used, frozen until now under
``PRETRAINING_ONLY`` because nothing optimised the objective it exists for.

**Why one and not both.** Masking both leaves ``triplet_emb.fp_proj`` and
``md_proj`` with no gradient whatsoever: their output is overwritten before it
reaches the stack. Masking only a fraction of molecules does not fix it either,
because the unmasked molecules contribute nothing to the loss. Those two
projections are exactly how knowledge enters the model during affinity
training, so a checkpoint that left them at initialisation would defeat the
point of pre-training. Cross-prediction keeps both live. This was found by
checking which parameters received gradient, not by reading the paper.

The targets cost nothing: they are what the knowledge cache already holds for
all 1,433,489 molecules.

## What may be pretrained on, and what may not

The specification is explicit that *"only training partitions may supply ...
chemistry pre-training"*. Self-supervision leaks no labels, but it does leak
**structure**: an encoder that has seen a test molecule's graph has seen the
test set. :func:`train_only_molecules` keeps only molecules appearing in
``train`` and in no other partition. Measured on split
``random_pair-5ef33512``:

* **767,870** ligand references appear in ``train``
* **220,241** of those also appear in val, cal or test
* a further **6,212** appear in ``quarantine``, where the locked benchmark
  panels' 337,191 observations are reserved
* **541,417** references survive, which collapse to **540,166
  distinct standardized SMILES** -- what this function returns, and what an
  epoch iterates

That last exclusion was missing at first, and it is the one worth naming: 6,212
molecules is only 1.1% of the set, and every one of them sits in a panel the
project will not score twice.

That set is regime-specific, and the consequence is worth stating plainly: a
checkpoint pretrained for ``random_pair`` is **not** legitimate for
``cold_drug``, which splits molecules differently. One pretrained encoder per
regime, or the leak is real.

## Speed, measured rather than projected

An earlier version of this section claimed ~2,000 mol/s and **4.6 minutes an
epoch**. That was wrong by a factor of 3.8 and it was a projection, not a
measurement. The run is **featurization-bound**, and the measured numbers on
this machine (8 physical cores, RTX 5060 Ti) are:

==================================  ===========
stage                               throughput
==================================  ===========
featurization, one core             148 mol/s
featurization, 8 worker processes   **525 mol/s**
parent-side collate and unpickle    11,964 mol/s
the GPU step alone, batch in VRAM   1,135 mol/s
==================================  ===========

So 540,166 molecules is **~17.1 minutes an epoch**, and thirty epochs is
**~8.6 hours**, not 2.3. The schedule is a real cost, not a free parameter.

Two things that follow from those numbers rather than from preference:

* **The pool does not scale.** One core does 148 mol/s; eight do 525, which is
  3.5x rather than 8x. Measured flat from 6 to 16 workers and from batch 512 to
  2048, so there is nothing to win by turning those knobs -- 8 workers, batch
  512 and 2 prefetch threads is already the best of eight combinations tried.
* **The GPU idles about 54% of the time**, because 525 of a possible 1,135
  mol/s reach it. That is where any remaining speed is, and it is behind
  RDKit's per-molecule cost rather than behind anything configurable here.

This is still far cheaper than affinity training, for the structural reason it
always was: no protein side, so no 144 MiB residue tensors and batches an order
of magnitude larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

# Written into every checkpoint this module produces and checked on load. A
# published ``base.pth`` carries no such marker, which is how the two are told
# apart without inspecting a filename. Declared beside the freezing policy
# because the two describe the same file.
from bindra.encoders.ligand_kpgt import PRETRAIN_CHECKPOINT_KIND


@dataclass(frozen=True)
class PretrainConfig:
    """Schedule and objective weights for the knowledge-denoising run."""

    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    #: Relative weight of the descriptor regression against the fingerprint
    #: term. The fingerprint is 512 sparse binary targets and the descriptors
    #: 200 dense continuous ones, so their raw magnitudes are not comparable.
    descriptor_weight: float = 1.0
    #: Share of molecules whose **fingerprint** node is the masked one; the
    #: rest have their **descriptor** node masked instead. Exactly one is
    #: masked per molecule -- see the module docstring for why both is wrong.
    fingerprint_mask_share: float = 0.5
    seed: int = 20261001
    amp_dtype: str = "bfloat16"
    #: Share of the train-only molecules held back to select a checkpoint.
    #:
    #: Held back from the **train** molecules, not taken from the affinity
    #: split's ``val`` partition. Selecting on val molecules would put their
    #: structures inside the encoder through the selection itself, which is the
    #: same leak :func:`train_only_molecules` exists to prevent, arriving by a
    #: quieter route. 2% of 547,629 is ~11,000 molecules, which is plenty to
    #: rank thirty checkpoints.
    val_fraction: float = 0.02
    #: Stop when the held-out loss has not improved for this many epochs.
    patience: int = 5
    log_every: int = 50
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError(
                f"val_fraction must be in (0, 0.5), got {self.val_fraction}"
            )
        if not 0.0 < self.fingerprint_mask_share < 1.0:
            raise ValueError(
                "fingerprint_mask_share must be in (0, 1); at 0 or 1 one of the "
                f"two loss terms has no rows, got {self.fingerprint_mask_share}"
            )
        if self.amp_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unknown amp_dtype {self.amp_dtype!r}")

    @property
    def torch_amp_dtype(self):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.amp_dtype]

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_fraction": self.warmup_fraction,
            "grad_clip": self.grad_clip,
            "descriptor_weight": self.descriptor_weight,
            "fingerprint_mask_share": self.fingerprint_mask_share,
            "val_fraction": self.val_fraction,
            "patience": self.patience,
            "seed": self.seed,
            "amp_dtype": self.amp_dtype,
            "objective": "cross-prediction of masked knowledge nodes",
        }


@dataclass(frozen=True)
class PretrainLoss:
    """The two terms, kept apart so each can be read on its own."""

    total: torch.Tensor
    fingerprint: torch.Tensor
    descriptors: torch.Tensor
    #: Share of fingerprint bits predicted correctly at a 0.5 threshold.
    #: Meaningless alone, so it is always carried with the baseline below.
    fingerprint_accuracy: float = 0.0
    #: Accuracy of predicting the **majority bit value** for this batch.
    #:
    #: Not the all-zeros rate, which is what this was first measured against on
    #: the assumption that a fingerprint is sparse. It is not: on 4,000 real
    #: curated molecules only 16.1% of the 512 bits are zero, so all-zeros
    #: scores 0.161 and the model's 0.582 looked like a large gain when the
    #: constant predictor available to it scores 0.839. The majority class is
    #: the honest floor whichever way the density falls.
    fingerprint_baseline: float = 0.0


def train_only_molecules(
    processed: Path | str,
    splits: Path | str,
    regime: str = "random_pair",
) -> list[str]:
    """Molecules in ``train`` and in no other partition, per the spec's rule.

    A molecule recurring in val, cal, test **or quarantine** is excluded
    entirely: pre-training on its graph would put an evaluation structure
    inside the encoder. Quarantine is not an afterthought in that list -- it is
    where the locked benchmark panels' observations are reserved, so a molecule
    leaking through it contaminates the one evaluation the project treats as
    sealed.
    """
    import gzip

    import pandas as pd

    processed = Path(processed)
    splits = Path(splits)

    frames = [
        pd.read_parquet(path, columns=["observation_id", "ligand_ref"])
        for path in sorted((processed / "observations").glob("*.parquet"))
    ]
    observations = pd.concat(frames)
    observations["observation_id"] = observations["observation_id"].astype(str)

    ids_dir = splits / regime / "ids"
    per_partition: dict[str, set[str]] = {}
    # `quarantine` belongs in this list and was missing from it. It holds the
    # observations reserved out of every partition so the locked benchmark
    # panels can be scored -- 337,191 of them, covering 192,871 molecules.
    # Omitting it put 6,212 sealed-panel molecules (1.1% of the pre-training
    # set) inside the encoder, which is the contamination the benchmark lock
    # exists to refuse, arriving through pre-training rather than through
    # training. Measured, not supposed: see the test that pins the count.
    for name in ("train", "val", "cal", "test", "quarantine"):
        path = ids_dir / f"{name}.txt.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as handle:
            wanted = {line.strip() for line in handle if line.strip()}
        rows = observations[observations["observation_id"].isin(wanted)]
        per_partition[name] = set(rows["ligand_ref"].astype(str))

    held_out: set[str] = set()
    for name, refs in per_partition.items():
        if name != "train":
            held_out |= refs
    safe = per_partition.get("train", set()) - held_out

    ligands = [
        pd.read_parquet(path, columns=["ligand_ref", "standardized_smiles", "status"])
        for path in sorted((processed / "ligands").glob("*.parquet"))
    ]
    table = pd.concat(ligands)
    table = table[
        (table["status"] == "OK") & (table["ligand_ref"].astype(str).isin(safe))
    ]
    return sorted({s for s in table["standardized_smiles"] if isinstance(s, str)})


def sample_mask(batch_size: int, config: PretrainConfig, generator=None) -> torch.Tensor:
    """Which molecules have their fingerprint masked rather than descriptors.

    Forced to contain both values. An all-true or all-false draw leaves one of
    the two loss terms with no rows, and a mean over an empty tensor is NaN --
    unlikely at a batch of 512, and not worth a silent NaN at step 300,000.
    """
    mask = torch.rand(batch_size, generator=generator) < config.fingerprint_mask_share
    if batch_size >= 2:
        if not bool(mask.any()):
            mask[0] = True
        if bool(mask.all()):
            mask[-1] = False
    return mask


def freeze_unused_heads(module: nn.Module) -> list[str]:
    """Freeze what this objective structurally cannot reach, and say what.

    ``node_predictor`` belongs to KPGT's masked-node objective, which this does
    not run, and with ``n_node_types=1`` it is degenerate anyway.
    ``model.mask_emb`` is LiGhT's internal copy; the masking here uses the
    predictor's. Left trainable they sit in the optimizer accruing state that
    never updates, and inflate any reported trainable-parameter count.
    """
    frozen: list[str] = []
    for name, parameter in module.named_parameters():
        if name.startswith("node_predictor") or name == "model.mask_emb.weight":
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def masked_knowledge_forward(
    module: nn.Module,
    batch,
    fingerprints: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    mask_fp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct each molecule's masked knowledge node from the other.

    ``mask_fp`` is per molecule, ``[B]`` boolean: True masks the fingerprint
    node and leaves the descriptor node visible, False does the reverse.

    Returns ``(fp_logits, md_pred)`` for the molecules whose corresponding node
    was masked, so the two have different row counts summing to the batch.
    """
    indicators = batch.vavn
    node_h = module.node_emb(batch.begin_end, indicators)
    edge_h = module.edge_emb(batch.edge_feats, indicators)
    triplet_h = module.triplet_emb(
        node_h, edge_h, fingerprints, descriptors, indicators
    ).clone()

    # A knowledge node belongs to a molecule; `graph_index` carries ownership.
    per_node = mask_fp.to(triplet_h.device)[batch.graph_index]
    hide_fp = (indicators == 1) & per_node
    hide_md = (indicators == 2) & ~per_node
    triplet_h[hide_fp | hide_md] = module.mask_emb.weight.to(triplet_h.dtype)

    states = module.model(batch, triplet_h)
    return module.fp_predictor(states[hide_fp]), module.md_predictor(states[hide_md])


def knowledge_loss(
    fp_logits: torch.Tensor,
    md_pred: torch.Tensor,
    fp_target: torch.Tensor,
    md_target: torch.Tensor,
    config: PretrainConfig,
) -> PretrainLoss:
    """Binary cross-entropy on the fingerprint, squared error on descriptors.

    Targets must already be restricted to the molecules whose corresponding
    node was masked, matching what :func:`masked_knowledge_forward` returned.
    """
    fingerprint = nn.functional.binary_cross_entropy_with_logits(
        fp_logits.float(), fp_target.float()
    )
    descriptors = nn.functional.mse_loss(md_pred.float(), md_target.float())
    with torch.no_grad():
        predicted = (fp_logits.float() > 0).float()
        accuracy = float((predicted == fp_target.float()).float().mean())
        # The majority-class rate, not the all-zeros rate. These fingerprints
        # are dense -- 83.9% of bits set on real curated molecules -- so
        # all-zeros is a weak baseline that makes any model look skilful.
        # max(p, 1 - p) is the best constant predictor either way.
        share_set = float(fp_target.float().mean())
        baseline = max(share_set, 1.0 - share_set)
    return PretrainLoss(
        total=fingerprint + config.descriptor_weight * descriptors,
        fingerprint=fingerprint,
        descriptors=descriptors,
        fingerprint_accuracy=accuracy,
        fingerprint_baseline=baseline,
    )


# --- the loop --------------------------------------------------------------
#
# Everything above is the objective; everything below runs it. They are kept in
# one module because the masking rule and the loop that applies it are the same
# idea, and splitting them invites a second masking rule.



class PretrainError(RuntimeError):
    """The pre-training run cannot proceed as configured."""


def prepare_for_pretraining(predictor: nn.Module) -> list[str]:
    """Unfreeze what this objective needs, then re-freeze what it cannot reach.

    ``apply_kpgt_freezing`` freezes everything matching ``PRETRAINING_ONLY`` --
    ``fp_predictor``, ``md_predictor`` and both ``mask_emb`` -- because no
    *affinity* forward pass reaches them. This objective reaches three of those
    five, and they are the three it exists to train: the two predictors and the
    embedding a masked node is replaced with. Running the loop without this
    leaves all three at initialisation while the loss still falls, because the
    stack underneath compensates for them.

    Returns the names left frozen, so a caller can print them rather than trust
    that the right ones were.
    """
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    return freeze_unused_heads(predictor)


def split_molecules(
    smiles: Sequence[str], *, val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Hold back a slice of the train-only molecules to select a checkpoint.

    The slice comes out of the **training** molecules. See
    :attr:`PretrainConfig.val_fraction` for why it may not come from the
    affinity split's ``val`` partition.
    """
    import numpy as np

    ordered = sorted(set(smiles))
    if len(ordered) < 2:
        raise PretrainError(
            f"{len(ordered)} molecule(s) to pretrain on; nothing to hold back"
        )
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(ordered))
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    n_val = min(n_val, len(ordered) - 1)
    validation = [ordered[i] for i in permutation[:n_val]]
    train = [ordered[i] for i in permutation[n_val:]]
    return train, validation


class MoleculeLoader:
    """Batches of molecules, featurized in a process pool.

    Not :class:`~bindra.data.batching.PairLoader`: there is no protein here, so
    none of what that class exists for applies. No residue budget, no length
    bucketing, no 144 MiB tensor to keep out of the pipe -- just fixed-size
    batches of SMILES, which is why a pre-training batch can be 512 where an
    affinity batch averages 87 pairs.

    A molecule that will not featurize is **dropped and counted**, not raised
    on. The corpus is curated and this should be empty; if it is not, a
    thirty-epoch run should not die at epoch 12 over one molecule, and the
    report prints the count so a drop cannot pass unnoticed.
    """

    def __init__(
        self,
        smiles: Sequence[str],
        builder,
        *,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int = 20261001,
        featurize_workers: int = 0,
        prefetch_workers: int = 2,
    ) -> None:
        if batch_size < 2:
            raise PretrainError(
                f"batch_size must be at least 2, got {batch_size}; sample_mask "
                "cannot place both mask kinds in a batch of one"
            )
        self.smiles = list(smiles)
        self.builder = builder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.featurize_workers = featurize_workers
        self.prefetch_workers = prefetch_workers
        self.epoch = 0
        self._pool = None
        #: Molecules dropped to a featurization failure, cumulative.
        self.dropped: list[str] = []

    def __len__(self) -> int:
        """Batches this loader yields, counting the merged trailing singleton."""
        count = (len(self.smiles) + self.batch_size - 1) // self.batch_size
        if count > 1 and len(self.smiles) % self.batch_size == 1:
            count -= 1
        return count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _executor(self):
        if self.featurize_workers <= 0:
            return None
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(max_workers=self.featurize_workers)
        return self._pool

    def _order(self) -> list[int]:
        import numpy as np

        if not self.shuffle:
            return list(range(len(self.smiles)))
        # Reproducible from (seed, epoch), like the affinity loader: a resumed
        # run must see the order the interrupted one would have seen.
        generator = np.random.default_rng((self.seed, self.epoch))
        return [int(i) for i in generator.permutation(len(self.smiles))]

    def _featurize(self, chunk: list[str], executor) -> list:
        featurize = self.builder.featurize
        errors = self.builder.featurization_errors
        try:
            if executor is not None and len(chunk) > 8:
                size = max(1, len(chunk) // (4 * (executor._max_workers or 1)))
                return list(executor.map(featurize, chunk, chunksize=size))
            return [featurize(text) for text in chunk]
        except errors:
            # One molecule in the chunk failed and `map` does not say which, so
            # the chunk is redone serially to find it. Rare by construction, and
            # the cost when it happens is one batch.
            parts = []
            for text in chunk:
                try:
                    parts.append(featurize(text))
                except errors:
                    self.dropped.append(text)
            return parts

    def _build(self, chunk: list[str], executor):
        parts = self._featurize(chunk, executor)
        if len(parts) < 2:
            return None
        return self.builder.collate(parts)

    def __iter__(self):
        executor = self._executor()
        order = self._order()
        chunks = [
            [self.smiles[i] for i in order[start : start + self.batch_size]]
            for start in range(0, len(order), self.batch_size)
        ]
        # A trailing chunk of one cannot be used: sample_mask needs two rows to
        # place both mask kinds, so `_build` would return None and the molecule
        # would vanish without appearing in `dropped`. Merged into the previous
        # chunk instead, at the cost of one batch being one larger.
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        if self.prefetch_workers <= 0:
            for chunk in chunks:
                batch = self._build(chunk, executor)
                if batch is not None:
                    yield batch
            return

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            pending: deque = deque()
            upcoming = iter(chunks)
            for _ in range(self.prefetch_workers + 1):
                following = next(upcoming, None)
                if following is None:
                    break
                pending.append(pool.submit(self._build, following, executor))
            while pending:
                batch = pending.popleft().result()
                following = next(upcoming, None)
                if following is not None:
                    pending.append(pool.submit(self._build, following, executor))
                if batch is not None:
                    yield batch


@dataclass
class PretrainReport:
    """One epoch over one partition."""

    epoch: int
    split: str
    loss: float
    fingerprint: float
    descriptors: float
    fingerprint_accuracy: float
    fingerprint_baseline: float
    molecules: int
    seconds: float
    dropped: int = 0

    @property
    def molecules_per_second(self) -> float:
        return self.molecules / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "split": self.split,
            "loss": self.loss,
            "fingerprint_bce": self.fingerprint,
            "descriptor_mse": self.descriptors,
            "fingerprint_accuracy": self.fingerprint_accuracy,
            "fingerprint_baseline": self.fingerprint_baseline,
            "molecules": self.molecules,
            "seconds": self.seconds,
            "molecules_per_second": self.molecules_per_second,
            "dropped": self.dropped,
        }

    def summary(self) -> str:
        lift = self.fingerprint_accuracy - self.fingerprint_baseline
        return (
            f"epoch {self.epoch:>3} {self.split:<5} loss {self.loss:.4f} "
            f"(fp {self.fingerprint:.4f}, md {self.descriptors:.4f}) | "
            f"fp acc {self.fingerprint_accuracy:.4f} vs majority-bit "
            f"{self.fingerprint_baseline:.4f} ({lift:+.4f}) | "
            f"{self.molecules:,} mol in {self.seconds:.0f}s "
            f"({self.molecules_per_second:.0f} mol/s)"
            + (f" | {self.dropped} dropped" if self.dropped else "")
        )


class PretrainRunner:
    """Runs the knowledge-denoising objective over a molecule loader.

    Deliberately not :class:`~bindra.training.engine.Trainer`. That class
    budgets residues, accumulates micro-batches, keeps an EMA and refuses every
    partition but train and val -- all of which exist for the affinity loop and
    none of which applies to a single-tower self-supervised pass. Reusing it
    would mean a second set of branches inside the one object that must never
    be wrong about which partition it is reading.

    What is shared is what is worth sharing: the cosine schedule with warmup,
    the atomic checkpoint write, and the rule that a locked file never ends a
    run.
    """

    def __init__(
        self,
        arm,
        config: PretrainConfig | None = None,
        *,
        device="cpu",
        run_directory: Path | str | None = None,
        provenance: dict | None = None,
    ) -> None:
        self.arm = arm
        #: The LiGhT predictor itself. The objective reads ``fp_predictor``,
        #: ``md_predictor`` and ``mask_emb`` off it by name, and those names are
        #: the published checkpoint's, which is what lets the result load
        #: strictly into an arm later.
        self.predictor = arm.module
        self.config = config or PretrainConfig()
        self.device = torch.device(device)
        self.predictor.to(self.device)
        self.run_directory = Path(run_directory) if run_directory else None
        if self.run_directory:
            self.run_directory.mkdir(parents=True, exist_ok=True)
        self.provenance = dict(provenance or {})

        self.frozen = prepare_for_pretraining(self.predictor)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.predictor.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = None
        self.epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None
        self.history: list[dict] = []

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)

    # -- the objective, one batch -----------------------------------------

    def _mask_generator(self, step: int):
        """A CPU generator seeded from ``(seed, step)``, fresh each step.

        Its own stream rather than the global one, so the mask a step sees does
        not depend on how much else consumed randomness before it. That is what
        makes a resumed run see the masks the interrupted one would have.
        """
        generator = torch.Generator()
        generator.manual_seed((self.config.seed * 1_000_003 + step) % (2**31 - 1))
        return generator

    def batch_loss(self, batch, step: int) -> PretrainLoss:
        """Mask one knowledge node per molecule and reconstruct it."""
        mask_fp = sample_mask(
            batch.num_graphs, self.config, generator=self._mask_generator(step)
        )
        fp_logits, md_pred = masked_knowledge_forward(
            self.predictor,
            batch.graphs,
            batch.fingerprints,
            batch.descriptors,
            mask_fp=mask_fp,
        )
        on_device = mask_fp.to(batch.fingerprints.device)
        fp_target = batch.fingerprints[on_device]
        md_target = batch.descriptors[~on_device]
        # The two row counts arrive from different places -- a boolean index
        # over nodes against one over molecules -- and a mismatch would train
        # against another molecule's knowledge while still producing finite
        # numbers and a falling loss.
        if fp_logits.shape[0] != fp_target.shape[0]:
            raise PretrainError(
                f"{fp_logits.shape[0]} masked fingerprint nodes against "
                f"{fp_target.shape[0]} molecules with the fingerprint masked"
            )
        if md_pred.shape[0] != md_target.shape[0]:
            raise PretrainError(
                f"{md_pred.shape[0]} masked descriptor nodes against "
                f"{md_target.shape[0]} molecules with the descriptors masked"
            )
        return knowledge_loss(fp_logits, md_pred, fp_target, md_target, self.config)

    # -- epochs ------------------------------------------------------------

    def _run_epoch(self, loader: MoleculeLoader, *, train: bool) -> PretrainReport:
        import time

        self.predictor.train(train)
        loader.set_epoch(self.epoch)
        before_dropped = len(loader.dropped)
        totals = {"loss": 0.0, "fp": 0.0, "md": 0.0, "acc": 0.0, "base": 0.0}
        molecules = 0
        batches = 0
        norm = torch.zeros(())
        started = time.perf_counter()
        use_amp = self.device.type == "cuda" and self.config.amp_dtype != "float32"

        for batch in loader:
            batch = batch.to(self.device)
            # Validation masks are keyed to the batch's position in the epoch,
            # not to global_step, so the held-out number is comparable across
            # epochs. Keyed to the step, an epoch that happened to mask easier
            # nodes would read as progress.
            step = self.global_step if train else -(batches + 1)
            with torch.autocast(
                self.device.type, dtype=self.config.torch_amp_dtype, enabled=use_amp
            ):
                loss = self.batch_loss(batch, step)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.predictor.parameters() if p.requires_grad],
                    self.config.grad_clip,
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
            batches += 1
            molecules += batch.num_graphs
            totals["loss"] += float(loss.total.detach())
            totals["fp"] += float(loss.fingerprint.detach())
            totals["md"] += float(loss.descriptors.detach())
            totals["acc"] += loss.fingerprint_accuracy
            totals["base"] += loss.fingerprint_baseline
            if train and self.config.verbose and batches % self.config.log_every == 0:
                rate = molecules / max(1e-9, time.perf_counter() - started)
                print(
                    f"    step {self.global_step:>6} | "
                    f"loss {float(loss.total.detach()):.4f} "
                    f"fp {float(loss.fingerprint.detach()):.4f} "
                    f"md {float(loss.descriptors.detach()):.4f} "
                    f"| grad {float(norm):.3f} "
                    f"| lr {self.optimizer.param_groups[0]['lr']:.2e} "
                    f"| {rate:.0f} mol/s",
                    flush=True,
                )

        if batches == 0:
            raise PretrainError(
                f"the {'train' if train else 'val'} loader produced no batch; "
                f"{len(loader.smiles):,} molecules at batch size "
                f"{loader.batch_size}"
            )
        divisor = float(batches)
        return PretrainReport(
            epoch=self.epoch,
            split="train" if train else "val",
            loss=totals["loss"] / divisor,
            fingerprint=totals["fp"] / divisor,
            descriptors=totals["md"] / divisor,
            fingerprint_accuracy=totals["acc"] / divisor,
            fingerprint_baseline=totals["base"] / divisor,
            molecules=molecules,
            seconds=time.perf_counter() - started,
            dropped=len(loader.dropped) - before_dropped,
        )

    def evaluate(self, loader: MoleculeLoader) -> PretrainReport:
        """Held-out loss, gradients off and masks fixed by batch position."""
        with torch.no_grad():
            return self._run_epoch(loader, train=False)

    def fit(
        self, train_loader: MoleculeLoader, val_loader: MoleculeLoader | None = None
    ) -> list[dict]:
        from bindra.training.engine import cosine_schedule_with_warmup

        total_steps = max(1, len(train_loader)) * self.config.epochs
        warmup = int(total_steps * self.config.warmup_fraction)
        # Rebuilt and fast-forwarded rather than restored blind, so a resumed
        # run continues the same curve even if the schedule length changed.
        self.scheduler = cosine_schedule_with_warmup(self.optimizer, total_steps, warmup)
        # torch warns that the scheduler stepped before the optimizer did. That
        # warning is about losing the schedule's first value, and it does not
        # apply here: this is a deliberate fast-forward of a LambdaLR whose rate
        # is a pure function of the step index, verified to land on the same
        # learning rate as an uninterrupted run. Suppressed narrowly so a real
        # ordering mistake elsewhere would still be reported.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*before `optimizer\.step\(\)`.*"
            )
            for _ in range(self.global_step):
                self.scheduler.step()

        stale = 0
        while self.epoch < self.config.epochs:
            train_report = self._run_epoch(train_loader, train=True)
            print(f"  {train_report.summary()}", flush=True)
            self.history.append(train_report.to_dict())

            selected = train_report.loss
            if val_loader is not None:
                val_report = self.evaluate(val_loader)
                print(f"  {val_report.summary()}", flush=True)
                self.history.append(val_report.to_dict())
                selected = val_report.loss

            improved = self.best_metric is None or selected < self.best_metric
            if improved:
                self.best_metric = selected
                stale = 0
            else:
                stale += 1
            self.epoch += 1
            self.save_checkpoint("last.pt")
            if improved:
                self.save_checkpoint("best.pt")
            self._write_history()
            if self.config.patience and stale >= self.config.patience:
                print(
                    f"  [stop] {stale} epochs without improvement on the "
                    f"held-out molecules; best {self.best_metric:.4f}",
                    flush=True,
                )
                break
        return self.history

    # -- checkpoints -------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            # Marks this as *our* pre-training output rather than the published
            # base.pth. The loader checks this instead of inspecting a filename.
            "kind": PRETRAIN_CHECKPOINT_KIND,
            # Named `model_state_dict` because that is the key the published
            # checkpoint uses, so one loader reads either file.
            "model_state_dict": self.predictor.state_dict(),
            "model_kwargs": self.arm.config.to_model_kwargs(),
            "ligand_config": self.arm.config.to_dict(),
            "pretrain_config": self.config.to_dict(),
            "provenance": self.provenance,
            "frozen_parameters": self.frozen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "history": self.history,
            "torch_rng": torch.get_rng_state(),
        }

    def save_checkpoint(self, name: str = "last.pt", *, attempts: int = 6) -> Path | None:
        """Atomic write, retried. A locked file never ends the run.

        Same judgement as the affinity engine's, for the same reason: the
        project tree lives under OneDrive, a sync client opens a freshly written
        file to upload it, and the rename then fails. Losing one checkpoint
        costs minutes; raising here costs hours.
        """
        import time

        if self.run_directory is None:
            return None
        path = self.run_directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(self.state_dict(), temporary)
        except OSError as exc:
            print(f"  [ckpt] could not write {temporary.name}: {exc}", flush=True)
            return None
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return path
            except PermissionError:
                if attempt == attempts - 1:
                    break
                time.sleep(0.5 * 2**attempt)
        print(
            f"  [ckpt] {path.name} is locked by another process (cloud sync?); "
            f"kept {temporary.name} and continued.",
            flush=True,
        )
        return None

    def load_checkpoint(self, path: Path | str) -> None:
        """Resume a pre-training run, optimizer and schedule included."""
        path = Path(path)
        if not path.exists():
            raise PretrainError(
                f"no checkpoint at {path}. Refusing to start from random weights "
                "when a checkpoint was requested."
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("kind") != PRETRAIN_CHECKPOINT_KIND:
            raise PretrainError(
                f"{path} is not a {PRETRAIN_CHECKPOINT_KIND} checkpoint "
                f"(kind={payload.get('kind')!r}), so it cannot resume this run"
            )
        self.predictor.load_state_dict(payload["model_state_dict"], strict=True)
        # A state-dict load does not restore requires_grad, and the arm's own
        # policy freezes three parameters this objective must train.
        self.frozen = prepare_for_pretraining(self.predictor)
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            if self.scheduler is None:
                from bindra.training.engine import cosine_schedule_with_warmup

                self.scheduler = cosine_schedule_with_warmup(self.optimizer, 1, 0)
            self.scheduler.load_state_dict(payload["scheduler"])
        self.epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        self.best_metric = payload.get("best_metric")
        self.history = list(payload.get("history", []))
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))

    def _write_history(self) -> None:
        if self.run_directory is None:
            return
        import json
        import time

        path = self.run_directory / "history.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "kind": PRETRAIN_CHECKPOINT_KIND,
                        "config": self.config.to_dict(),
                        "ligand_config": self.arm.config.to_dict(),
                        "provenance": self.provenance,
                        "frozen_parameters": self.frozen,
                        "trainable_parameters": self.n_trainable,
                        "epochs": self.history,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            for attempt in range(4):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * 2**attempt)
        except OSError as exc:
            print(
                f"  [hist] could not update history.json ({exc}); continuing",
                flush=True,
            )


__all__ = [
    "PRETRAIN_CHECKPOINT_KIND",
    "MoleculeLoader",
    "PretrainConfig",
    "PretrainError",
    "PretrainLoss",
    "PretrainReport",
    "PretrainRunner",
    "freeze_unused_heads",
    "knowledge_loss",
    "masked_knowledge_forward",
    "prepare_for_pretraining",
    "sample_mask",
    "split_molecules",
    "train_only_molecules",
]
