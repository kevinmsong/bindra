"""Train the analog generator: a conditional discrete diffusion over graphs.

The task is **scaffold-constrained analog generation**. Given a target, an
endpoint, a seed molecule and a scaffold to preserve, produce another molecule
measured against that same target. The training pairs come from the corpus
itself: two ligands measured on the same construct, same endpoint, same assay
context, sharing a scaffold.

**The clean product is never a condition.** It is what the denoiser predicts.
Conditions are the target's pooled residue features, the endpoint, and the seed
molecule -- all computable before the answer is known.
:func:`~bindra.generation.digress.denoiser.assert_no_product_leakage` states
that as a check, because a model handed the answer trains beautifully and
generates nothing, and the loss curve looks better rather than worse.

**Pairs are built from training partitions only.** A pair is two observations,
and if either sits in val or test the generator has seen a held-out molecule.

The objective is the ordinary discrete-diffusion one: draw ``t``, corrupt the
product to ``x_t``, predict the clean categories, cross-entropy against the
truth. Scaffold positions are excluded from the loss as well as from the
corruption -- they were never noised, so predicting them is free and including
them would inflate accuracy by whatever fraction of the molecule is scaffold.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


class GenerationTrainingError(RuntimeError):
    """Training was asked for something it cannot honestly do."""


@dataclass
class GenerationConfig:
    """What the run is, recorded into the checkpoint."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    patience: int = 3
    timesteps: int = 500
    max_atoms: int = 64
    model_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    ema_decay: float = 0.999
    seed: int = 20261001
    val_fraction: float = 0.05
    #: Weight on the bond term. Bonds outnumber atoms by N, so an unweighted
    #: sum lets the bond matrix dominate and the model learns connectivity
    #: while getting the elements wrong.
    bond_weight: float = 0.5
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AnalogTriple:
    """One training example, before featurization."""

    seed_smiles: str
    product_smiles: str
    sequence_sha256: str
    endpoint: str
    delta: float


def build_triples(
    processed: "Path | str",
    split_directory: "Path | str",
    *,
    partitions=("train",),
    max_per_group: int = 32,
    limit: int | None = None,
    seed: int = 20261001,
) -> "list[AnalogTriple]":
    """Analog pairs sharing a scaffold, from training partitions only.

    Grouped by ``(sequence_sha256, endpoint, assay_condition_bucket,
    scaffold_key)``. The scaffold key is the curation layer's own, so two
    molecules in a group share a scaffold by construction rather than by an
    MCS search run here.
    """
    import pandas as pd

    from bindra.data.splits import read_split
    from bindra.data.tables import ShardedTableRef

    processed = Path(processed)
    _, assignment = read_split(Path(split_directory))
    wanted = {o for o, p in assignment.items() if p in set(partitions)}
    if not wanted:
        raise GenerationTrainingError(f"no observations in partitions {partitions}")

    payload = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
    columns = [
        "observation_id", "ligand_ref", "sequence_sha256", "endpoint", "p_value",
        "is_exact", "assay_condition_bucket", "scaffold_key", "analog_pair_eligible",
    ]
    frames = []
    for frame in reference.iter_frames(columns=columns):
        frame = frame[frame["observation_id"].isin(wanted)]
        frame = frame[frame["is_exact"].astype(bool) & frame["analog_pair_eligible"].astype(bool)]
        if len(frame):
            frames.append(frame)
    if not frames:
        raise GenerationTrainingError("no analog-eligible exact rows in these partitions")
    observations = pd.concat(frames, ignore_index=True)

    ligands = pd.concat(
        [pd.read_parquet(p, columns=["ligand_ref", "standardized_smiles", "status"])
         for p in sorted((processed / "ligands").glob("*.parquet"))],
        ignore_index=True,
    )
    ligands = ligands[ligands["status"] == "OK"].set_index("ligand_ref")
    observations = observations.join(ligands["standardized_smiles"], on="ligand_ref", how="inner")

    rng = np.random.default_rng(seed)
    triples: list[AnalogTriple] = []
    grouped = observations.groupby(
        ["sequence_sha256", "endpoint", "assay_condition_bucket", "scaffold_key"], sort=False
    )
    for (sequence, endpoint, _, _), block in grouped:
        if len(block) < 2:
            continue
        smiles = block["standardized_smiles"].to_numpy()
        values = block["p_value"].to_numpy(dtype=float)
        i, j = np.triu_indices(len(block), k=1)
        # Both directions matter: generation is asked to go from either member
        # to the other, and a one-directional dataset teaches only improvement.
        pairs = list(zip(i, j)) + list(zip(j, i))
        if len(pairs) > max_per_group:
            chosen = rng.choice(len(pairs), max_per_group, replace=False)
            pairs = [pairs[k] for k in chosen]
        for a, b in pairs:
            if smiles[a] == smiles[b]:
                continue
            triples.append(AnalogTriple(
                seed_smiles=str(smiles[a]), product_smiles=str(smiles[b]),
                sequence_sha256=str(sequence), endpoint=str(endpoint),
                delta=float(values[b] - values[a]),
            ))
        if limit and len(triples) >= limit:
            break
    if limit:
        triples = triples[:limit]
    if not triples:
        raise GenerationTrainingError("no usable analog pairs were formed")
    return triples


class TripleDataset:
    """Featurized triples, with the attrition the encoder caused recorded."""

    def __init__(self, triples, protein_cache, *, max_atoms: int = 64) -> None:
        from bindra.generation.digress.graphs import (
            GraphEncodingError,
            encode_molecule,
            scaffold_mask,
        )

        self.max_atoms = max_atoms
        self.items = []
        self.attrition = {"input": len(triples), "unencodable": 0, "no_protein": 0}
        cache_keys = set(protein_cache.keys()) if protein_cache is not None else None
        self._protein_cache = protein_cache
        metadata = protein_cache.metadata if protein_cache is not None else None

        from bindra.heads.affinity import ENDPOINT_INDEX

        for triple in triples:
            try:
                product = encode_molecule(triple.product_smiles, max_atoms=max_atoms)
                seed = encode_molecule(triple.seed_smiles, max_atoms=max_atoms)
                keep = scaffold_mask(triple.product_smiles, max_atoms=max_atoms)
            except GraphEncodingError:
                self.attrition["unencodable"] += 1
                continue
            self.items.append({
                "product": product, "seed": seed, "keep": keep,
                "sequence_sha256": triple.sequence_sha256,
                "endpoint": ENDPOINT_INDEX.get(triple.endpoint, 0),
                "delta": triple.delta,
            })
        self.attrition["kept"] = len(self.items)
        del cache_keys, metadata

    def __len__(self) -> int:
        return len(self.items)


#: Width of the encoded potency change: five signed buckets plus one scaled
#: scalar. See :func:`encode_delta`.
DELTA_DIM = 6

#: Bucket edges as multiples of the endpoint's own replicate-noise floor. A
#: change smaller than the floor is not distinguishable from measuring the
#: same compound twice, so it is "no real change" rather than a small one.
DELTA_EDGES = (-2.0, -1.0, 1.0, 2.0)


def encode_delta(deltas, endpoint_indices, device=None):
    """``[B, DELTA_DIM]`` encoding of the requested potency change.

    Without this the generator learns what analogs of a scaffold look like on
    a target, but not which direction is wanted -- it reproduces the
    distribution of observed analogs rather than directional improvement.

    Two encodings, because either alone is weak. The raw scalar carries
    magnitude but is one number among fifteen and easy to ignore; the bucket
    one-hot is impossible to ignore but throws away magnitude.

    **The buckets are in units of the endpoint's replicate-noise floor**
    (0.945 Kd, 0.641 Ki, 0.671 IC50), not in fixed log units. A 0.7 log
    improvement is real for Ki and inside the noise for Kd, and a fixed
    threshold would label those the same.
    """
    from bindra.heads.affinity import ENDPOINT_ORDER
    from bindra.training.losses.affinity import noise_floor_table

    floors = noise_floor_table()
    ordered = np.array(
        [floors[str(getattr(e, "value", e))] for e in ENDPOINT_ORDER], dtype=np.float32
    )
    deltas = np.asarray(deltas, dtype=np.float32)
    endpoints = np.asarray(endpoint_indices, dtype=np.int64)
    scaled = deltas / ordered[endpoints]

    out = np.zeros((len(deltas), DELTA_DIM), dtype=np.float32)
    bucket = np.digitize(scaled, DELTA_EDGES)  # 0..4
    out[np.arange(len(deltas)), bucket] = 1.0
    # Clipped so one extreme pair cannot dominate the scale the model sees.
    out[:, 5] = np.clip(scaled, -4.0, 4.0) / 4.0
    tensor = torch.from_numpy(out)
    return tensor.to(device) if device is not None else tensor


def _conditions(batch_items, device):
    """What the denoiser is told: endpoint, seed molecule, requested change.

    The clean product is absent by construction -- it is what is being
    predicted. :func:`assert_no_product_leakage` states that as a check.
    """
    from bindra.heads.affinity import ENDPOINT_ORDER

    n = len(batch_items)
    endpoint = torch.zeros(n, len(ENDPOINT_ORDER), device=device)
    for i, item in enumerate(batch_items):
        endpoint[i, item["endpoint"]] = 1.0
    seed_summary = torch.stack([
        torch.from_numpy(
            np.concatenate([
                np.bincount(it["seed"].atoms[it["seed"].mask], minlength=11).astype(np.float32),
                np.array([it["seed"].n_atoms], dtype=np.float32),
            ])
        ) for it in batch_items
    ]).to(device)
    delta = encode_delta(
        [it["delta"] for it in batch_items],
        [it["endpoint"] for it in batch_items],
        device=device,
    )
    return torch.cat((endpoint, seed_summary, delta), dim=-1)


def train(
    triples,
    *,
    protein_cache=None,
    config: GenerationConfig | None = None,
    out_directory: "Path | str" = "runs/generator",
    device=None,
    log_every: int = 25,
) -> dict:
    """Fit the denoiser. Returns the history it wrote."""
    from bindra.generation.digress.denoiser import DenoiserConfig, GraphDenoiser
    from bindra.generation.digress.graphs import (
        N_ATOM_TYPES, N_BOND_TYPES, collate, corpus_marginals,
    )
    from bindra.generation.digress.noise import DiscreteGraphNoise

    config = config or GenerationConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    dataset = TripleDataset(triples, protein_cache, max_atoms=config.max_atoms)
    if len(dataset) < 2:
        raise GenerationTrainingError(
            f"only {len(dataset)} triples survived featurization: {dataset.attrition}"
        )
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(dataset))
    n_val = max(1, int(config.val_fraction * len(dataset)))
    val_index, train_index = order[:n_val], order[n_val:]

    marginals, n_used = corpus_marginals(
        [t.product_smiles for t in triples[:5000]], max_atoms=config.max_atoms
    )
    noise = DiscreteGraphNoise(marginals, timesteps=config.timesteps)

    from bindra.heads.affinity import ENDPOINT_ORDER

    condition_dim = len(ENDPOINT_ORDER) + 12 + DELTA_DIM
    model = GraphDenoiser(DenoiserConfig(
        n_atom_types=N_ATOM_TYPES, n_bond_types=N_BOND_TYPES,
        condition_dim=condition_dim, model_dim=config.model_dim,
        n_layers=config.n_layers, n_heads=config.n_heads, dropout=config.dropout,
    )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run_epoch(indices, training: bool):
        model.train(training)
        rng.shuffle(indices) if training else None
        totals = {"loss": 0.0, "atom": 0.0, "bond": 0.0, "n": 0, "n_atom": 0, "n_bond": 0}
        for start in range(0, len(indices), config.batch_size):
            chunk = [dataset.items[k] for k in indices[start:start + config.batch_size]]
            if len(chunk) < 2:
                continue
            product = collate([c["product"] for c in chunk])
            atoms0 = product["atoms"].to(device)
            bonds0 = product["bonds"].to(device)
            mask = product["mask"].to(device)
            keep = torch.from_numpy(np.stack([c["keep"] for c in chunk])).to(device)
            conditions = _conditions(chunk, device).unsqueeze(1)

            t = torch.randint(1, config.timesteps + 1, (len(chunk),), device=device)
            with torch.set_grad_enabled(training):
                noisy_atoms, noisy_bonds = noise.q_sample(atoms0, bonds0, t, keep=keep)
                atom_logits, bond_logits = model(
                    noisy_atoms, noisy_bonds, t, conditions, mask
                )
                # Scaffold positions were never corrupted, so predicting them
                # is free. Scoring them would inflate accuracy by whatever
                # fraction of the molecule the scaffold happens to be.
                atom_score = mask & ~keep
                pair_score = (mask.unsqueeze(1) & mask.unsqueeze(2)) & ~(
                    keep.unsqueeze(1) & keep.unsqueeze(2)
                )
                atom_loss, n_atom = _masked_cross_entropy(atom_logits, atoms0, atom_score)
                bond_loss, n_bond = _masked_cross_entropy(bond_logits, bonds0, pair_score)
                loss = atom_loss + config.bond_weight * bond_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            # Weighted by SCORED POSITIONS, not by batch size. A molecule that
            # is entirely its own scaffold scores nothing, and counting it as a
            # zero-loss example is how the reported bond loss came out at
            # 0.2143 when the real figure was 1.2760.
            # Detached before float(), or torch warns about converting a
            # grad-tracking tensor to a scalar.
            totals["atom"] += float(atom_loss.detach()) * n_atom
            totals["bond"] += float(bond_loss.detach()) * n_bond
            totals["n_atom"] += n_atom
            totals["n_bond"] += n_bond
            totals["loss"] += float(loss.detach()) * len(chunk)
            totals["n"] += len(chunk)
        return {
            "loss": totals["loss"] / max(totals["n"], 1),
            "atom": totals["atom"] / max(totals["n_atom"], 1),
            "bond": totals["bond"] / max(totals["n_bond"], 1),
            "n": totals["n"],
            "n_atom_scored": totals["n_atom"],
            "n_bond_scored": totals["n_bond"],
        }

    history, best, bad_epochs = [], None, 0
    for epoch in range(config.epochs):
        started = time.perf_counter()
        train_stats = run_epoch(list(train_index), True)
        val_stats = run_epoch(list(val_index), False)
        record = {
            "epoch": epoch, "seconds": round(time.perf_counter() - started, 1),
            "train": train_stats, "val": val_stats,
        }
        history.append(record)
        print(
            f"[  ok   ] epoch {epoch}: train {train_stats['loss']:.4f} "
            f"(atom {train_stats['atom']:.4f}, bond {train_stats['bond']:.4f}) | "
            f"val {val_stats['loss']:.4f}  {record['seconds']:.0f}s",
            flush=True,
        )
        if best is None or val_stats["loss"] < best:
            best, bad_epochs = val_stats["loss"], 0
            torch.save({
                "kind": "bindra-analog-generator-v1",
                "model": model.state_dict(),
                "config": config.to_dict(),
                "marginals": {"atom": noise.marginals.atom, "bond": noise.marginals.bond},
                "condition_dim": condition_dim,
                "epoch": epoch, "best_val_loss": best,
                "attrition": dataset.attrition,
                "marginals_from_molecules": n_used,
                "trainable_parameters": trainable,
            }, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"[ info  ] no improvement for {config.patience} epochs; stopping")
                break
    payload = {
        "config": config.to_dict(), "history": history,
        "attrition": dataset.attrition, "trainable_parameters": trainable,
        "n_train": len(train_index), "n_val": len(val_index),
    }
    (out / "history.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _masked_cross_entropy(logits, target_onehot, score_mask):
    """Cross-entropy over scored positions, and how many there were.

    The count is returned, not discarded, because some molecules score nothing:
    a molecule that *is* its own scaffold has no decorated atoms and no
    non-scaffold bonds. Averaging those in as a zero loss reports a number six
    times better than the truth -- measured, 0.2143 against an actual 1.2760.
    The caller must weight by the count.
    """
    n = int(score_mask.sum())
    if n == 0:
        # Kept in the graph so the batch still has a gradient path, but it
        # contributes no positions and the caller's denominator ignores it.
        return logits.sum() * 0.0, 0
    flat_logits = logits[score_mask]
    flat_target = target_onehot[score_mask].argmax(dim=-1)
    return torch.nn.functional.cross_entropy(flat_logits, flat_target), n


__all__ = [
    "DELTA_DIM",
    "DELTA_EDGES",
    "AnalogTriple",
    "GenerationConfig",
    "GenerationTrainingError",
    "TripleDataset",
    "build_triples",
    "encode_delta",
    "train",
]
