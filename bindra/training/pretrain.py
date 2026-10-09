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
``train`` and in no other partition -- **547,629 of 1,070,856 for
``random_pair``, so 49% of the corpus is excluded.**

That set is regime-specific, and the consequence is worth stating plainly: a
checkpoint pretrained for ``random_pair`` is **not** legitimate for
``cold_drug``, which splits molecules differently. One pretrained encoder per
regime, or the leak is real.

## Speed

Measured on the RTX 5060 Ti at ``d_g_feats=96``, 8 layers: ~2,000 mol/s on the
GPU, flat from batch 256 to 1024, so the step is compute-bound rather than
launch-bound. Featurization with the knowledge cache runs ~1,990 mol/s across
14 workers, which balances it. 547,629 molecules is therefore **~4.6 minutes an
epoch** -- cheap enough that the schedule, not the hardware, is the constraint.

There is no protein side here, which is why this is so much cheaper than
affinity training: no 144 MiB residue tensors, and batches an order of
magnitude larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


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
            "seed": self.seed,
            "objective": "cross-prediction of masked knowledge nodes",
        }


@dataclass(frozen=True)
class PretrainLoss:
    """The two terms, kept apart so each can be read on its own."""

    total: torch.Tensor
    fingerprint: torch.Tensor
    descriptors: torch.Tensor
    #: Share of fingerprint bits predicted correctly at a 0.5 threshold. The
    #: fingerprint is sparse, so this sits near the all-zeros baseline early
    #: and is reported beside it rather than alone.
    fingerprint_accuracy: float = 0.0
    fingerprint_baseline: float = 0.0


def train_only_molecules(
    processed: Path | str,
    splits: Path | str,
    regime: str = "random_pair",
) -> list[str]:
    """Molecules in ``train`` and in no other partition, per the spec's rule.

    A molecule recurring in val, cal or test is excluded entirely: pre-training
    on its graph would put the evaluation set's structures inside the encoder.
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
    for name in ("train", "val", "cal", "test"):
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
        # A sparse fingerprint makes "predict all zeros" a strong-looking
        # baseline, so it is reported alongside and never on its own.
        baseline = float((fp_target.float() == 0).float().mean())
    return PretrainLoss(
        total=fingerprint + config.descriptor_weight * descriptors,
        fingerprint=fingerprint,
        descriptors=descriptors,
        fingerprint_accuracy=accuracy,
        fingerprint_baseline=baseline,
    )


__all__ = [
    "PretrainConfig",
    "PretrainLoss",
    "freeze_unused_heads",
    "knowledge_loss",
    "masked_knowledge_forward",
    "sample_mask",
    "train_only_molecules",
]
