"""``bindra plot-ibam``: export iBAM matrices and publication figures.

What this command produces, per explained pair:

``ibam_matrices.npz`` / ``ibam_metadata.json``
    The numeric artifact of record, including the **unaveraged** per-layer,
    per-head attention. Every figure is re-derivable from these two files.
``interaction_panel.{pdf,svg,png}``
    The composite figure: whole-sequence profile, a zoomed residue x atom
    matrix, the molecule coloured by per-atom contribution, and ranked tables.
``residue_profile``, ``head_agreement``, ``residue_atom``, ``residue_token``
    The individual panels, for when one is wanted on its own.

## Selection is a stated rule, never an eye

Which pair gets explained changes what the figure appears to show, so the rule
is prespecified, recorded in ``selection.json`` beside the figures, and printed
on the figure itself.

There is deliberately **no** "pick the pairs the model predicts best" rule.
Selecting examples by the model's own accuracy and then showing their attention
is the standard way an interpretability figure flatters a model: it reports the
attention of the easy cases as though it were the attention of the method. If
you want to know whether the attention is informative, ``--select random`` with
a recorded seed is the honest instrument, and the prediction error is printed on
every panel either way.

## What the figures do not show

The disclaimer is burned into every figure because this is the point where a
reader is most likely to over-read: attention is not a contact map, not a
binding-site prediction, and not a mechanism. ``residue_atom`` in particular is
a **derived** aggregation -- the model attends to tokens, and the mapping onto
atoms is ours. The figures say so on their face.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
WARN = " warn  "
FAIL = " FAIL  "

#: Prespecified. See the module docstring on why accuracy-based selection is absent.
SELECTION_RULES = {
    "strongest": "highest measured pX among exact, primary-eligible observations",
    "random": "uniform random among exact, primary-eligible observations, seeded",
    "diverse": "the median-potency observation of each of the most-measured "
    "distinct targets",
}


def _select(frame, rule: str, count: int, seed: int, max_length: int):
    """Apply a prespecified selection rule. Returns the chosen rows.

    One pair per distinct target: five figures of the same protein would look
    like five pieces of evidence and be one.
    """
    import numpy as np

    eligible = frame[
        frame["is_exact"]
        & frame["primary_regression_eligible"]
        & (frame["sequence_length"] <= max_length)
    ]
    if eligible.empty:
        raise SystemExit(
            f"no exact, primary-eligible observation has a sequence of "
            f"{max_length} residues or fewer; raise --max-length"
        )
    if rule == "strongest":
        ordered = eligible.sort_values("p_value", ascending=False)
    elif rule == "random":
        ordered = eligible.sample(
            n=min(len(eligible), count * 50), random_state=seed
        )
    elif rule == "diverse":
        # The *median* observation of each target, not the strongest. Sorting by
        # potency puts pKi 13.85 at the top -- 0.14 picomolar, which is past what
        # the assays behind this corpus resolve and is far more likely a unit or
        # transcription artifact than a real measurement. Explaining those makes
        # the figures a tour of the worst-predicted outliers in the set.
        popular = eligible["sequence_sha256"].value_counts().index[: count * 4]
        pool = eligible[eligible["sequence_sha256"].isin(set(popular))].copy()
        middle = pool.groupby("sequence_sha256")["p_value"].transform("median")
        pool["_distance"] = (pool["p_value"] - middle).abs()
        order = {digest: rank for rank, digest in enumerate(popular)}
        pool["_target_rank"] = pool["sequence_sha256"].map(order)
        ordered = pool.sort_values(["_target_rank", "_distance"])
    else:
        raise SystemExit(f"unknown --select {rule!r}; use {', '.join(SELECTION_RULES)}")
    # One per target, in the rule's order.
    chosen = ordered.drop_duplicates("sequence_sha256").head(count)
    return chosen.reset_index(drop=True)


def _sequences_for(processed: Path, digests) -> dict[str, str]:
    """``sequence_sha256 -> sequence``, for the handful of chosen constructs only."""
    import pandas as pd

    constructs = pd.read_parquet(
        processed / "target_constructs.parquet",
        columns=["sequence_ref", "sequence_sha256", "status"],
    )
    constructs = constructs[
        (constructs["status"] == "OK") & constructs["sequence_sha256"].isin(set(digests))
    ]
    sequences = pd.read_parquet(
        processed / "sequences_raw.parquet", columns=["sequence_ref", "sequence"]
    ).set_index("sequence_ref")
    joined = constructs.join(sequences, on="sequence_ref").drop_duplicates(
        "sequence_sha256"
    )
    return dict(zip(joined["sequence_sha256"], joined["sequence"]))


def _explain_one(
    model,
    cache,
    sequence: str,
    smiles: str,
    *,
    device,
    policy,
    checkpoint_name: str,
):
    """One pair through the model with attention on, then an export.

    Returns ``(IBAMExport, predictions)``. The export is built from the same
    forward pass that produced the predictions, so the figure and the number
    printed on it cannot describe different runs of the model.
    """
    import torch

    from bindra.encoders.ligand_gat import GATLigandEncoder
    from bindra.encoders.ligand_graph import collate_graphs, featurize_molecule
    from bindra.heads.affinity import ENDPOINT_ORDER
    from bindra.ibam.export import export_ibam

    features = cache.get(cache.metadata.cache_key(sequence))
    protein = features.residue_features.to(device, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, protein.shape[1], dtype=torch.bool, device=device)

    graph = featurize_molecule(smiles)
    batch = collate_graphs([graph]).to(device)
    with torch.no_grad():
        output = model(protein, mask, batch, need_attention=True)

    # The trained module behind the encoder-shaped wrapper, so the token
    # metadata and the attention come from the same weights.
    wrapper = GATLigandEncoder(module=model.ligand_encoder, checkpoint=checkpoint_name)
    ligand_features = wrapper.encode(smiles)

    export = export_ibam(
        output.fusion,
        ligand_features,
        index=0,
        residue_positions=features.residue_indices,
        policy=policy,
        model_checkpoint=checkpoint_name,
        protein_encoder=cache.metadata.name,
    )
    predictions = {
        name.value: float(output.prediction.values[0, column])
        for column, name in enumerate(ENDPOINT_ORDER)
    }
    return export, predictions


def cmd_plot_ibam(args) -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[{FAIL}] no checkpoint at {checkpoint}")
        return 1

    import torch

    from bindra.cli.evaluate import _load_model
    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
    )
    from bindra.features.cache import ProteinFeatureCache
    from bindra.ibam.export import AggregationPolicy
    from bindra.ibam.plots import PUBLICATION, plot_all

    from dataclasses import replace as _replace

    device = torch.device(args.device)
    model, payload, config, used_ema = _load_model(checkpoint, device)
    print(
        f"[{OK}] checkpoint epoch {payload.get('epoch')}, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{'EMA' if used_ema else 'raw'} weights"
    )

    policy = AggregationPolicy(
        head_reduction=args.head_reduction,
        layer_reduction=args.layer_reduction,
        renormalize_physical=args.renormalize,
    )
    print(f"[{INFO}] aggregation: {policy.describe()} [{policy.policy_hash}]")

    metadata = ESMCProteinEncoder(n_trainable_layers=0).metadata(
        EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
    )
    cache = ProteinFeatureCache(args.protein_cache, metadata=metadata)

    style = _replace(PUBLICATION, png_dpi=args.dpi, write_png=not args.no_png)
    out = Path(args.out)

    # --- which pairs ------------------------------------------------------
    if args.sequence and args.smiles:
        chosen = [
            {
                "sequence": args.sequence,
                "standardized_smiles": args.smiles,
                "label": "explicit",
                "p_value": None,
                "endpoint": None,
                "observation_id": "explicit",
            }
        ]
        rule = "explicit --sequence/--smiles supplied on the command line"
    else:
        from bindra.data.batching import PairDataset

        print(f"[{INFO}] loading the {args.partition} partition of {args.split}")
        dataset = PairDataset(
            args.processed,
            args.split,
            args.partition,
            protein_cache=cache,
            limit=args.limit,
        )
        print(f"[{OK}] {len(dataset.frame):,} candidate observations")
        selected = _select(
            dataset.frame, args.select, args.count, args.seed, args.max_length
        )
        sequences = _sequences_for(
            Path(args.processed), selected["sequence_sha256"].tolist()
        )
        chosen = []
        for _, row in selected.iterrows():
            chosen.append(
                {
                    "sequence": sequences[row["sequence_sha256"]],
                    "standardized_smiles": row["standardized_smiles"],
                    "label": str(row["observation_id"]),
                    "p_value": float(row["p_value"]),
                    "endpoint": str(row["endpoint"]),
                    "observation_id": str(row["observation_id"]),
                }
            )
        rule = SELECTION_RULES[args.select]
        if args.select == "random":
            rule += f" (seed {args.seed})"
        rule += (
            f"; one pair per distinct target; sequences of at most "
            f"{args.max_length} residues; drawn from "
            f"{len(dataset.frame):,} observations in the {args.partition} partition"
        )

    print(f"[{INFO}] selection rule: {rule}")

    written: list[dict[str, object]] = []
    for index, pair in enumerate(chosen):
        name = f"{index:02d}_{pair['label']}"
        directory = out / name
        try:
            export, predictions = _explain_one(
                model,
                cache,
                pair["sequence"],
                pair["standardized_smiles"],
                device=device,
                policy=policy,
                checkpoint_name=str(checkpoint),
            )
        except Exception as error:
            print(f"[{WARN}] {name}: {type(error).__name__}: {error}")
            continue

        # Numbers first, figures second: the export is the artifact of record,
        # so a crash in matplotlib must not cost the matrices.
        paths = export.write(directory)

        measured = pair.get("p_value")
        endpoint = pair.get("endpoint")
        banner = (
            f"predicted pKd {predictions['Kd']:.2f} / pKi {predictions['Ki']:.2f} / "
            f"pIC50 {predictions['IC50']:.2f}"
        )
        if measured is not None and endpoint in predictions:
            banner += (
                f"   |   measured p{endpoint} {measured:.2f}, "
                f"error {predictions[endpoint] - measured:+.2f} log units"
            )
        banner += f"\nselection: {rule}"

        figures = plot_all(
            export,
            directory,
            smiles=pair["standardized_smiles"],
            sequence=pair["sequence"],
            style=style,
            prediction=banner,
        )
        (directory / "selection.json").write_text(
            json.dumps(
                {
                    "selection_rule": rule,
                    "select": args.select if not args.sequence else "explicit",
                    "seed": args.seed,
                    "observation_id": pair["observation_id"],
                    "sequence_length": len(pair["sequence"]),
                    "smiles": pair["standardized_smiles"],
                    "measured_p_value": measured,
                    "measured_endpoint": endpoint,
                    "predicted": predictions,
                    "checkpoint": str(checkpoint),
                    "used_ema_weights": used_ema,
                    "aggregation": policy.to_dict(),
                    "aggregation_hash": policy.policy_hash,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        count = sum(len(v) for v in figures.values())
        print(
            f"[{OK}] {name}: {export.sequence_length:,} residues x "
            f"{export.num_atoms} atoms, {count} figure files, matrices in "
            f"{paths['matrices'].name}"
        )
        written.append({"name": name, "directory": str(directory)})

    if not written:
        print(f"[{FAIL}] nothing was explained")
        return 1
    print(
        f"\n[{OK}] {len(written)} pair(s) explained under {out}"
        f"\n[{INFO}] attention is not a contact map, a binding site or a mechanism; "
        "the residue x atom view is a derived aggregation"
    )
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra plot-ibam",
        description="Export iBAM matrices and publication figures for a checkpoint.",
    )
    parser.add_argument("checkpoint", help="a trained checkpoint (best.pt / last.pt)")
    parser.add_argument("--out", default="docs/figures/ibam")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--processed", default="data/processed/202610")
    parser.add_argument("--split", default="data/splits/202610/random_pair")
    parser.add_argument("--partition", default="val", choices=["train", "val", "cal"])
    parser.add_argument("--protein-cache", default="data/cache/proteins")
    parser.add_argument(
        "--select",
        default="diverse",
        choices=sorted(SELECTION_RULES),
        help="prespecified selection rule; recorded beside the figures",
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--max-length",
        type=int,
        default=1200,
        help="skip constructs longer than this; the profile stays readable but "
        "the matrix panel becomes a window",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sequence", default=None, help="explain one explicit pair")
    parser.add_argument("--smiles", default=None, help="explain one explicit pair")
    parser.add_argument("--head-reduction", default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--layer-reduction", default="last", choices=["last", "mean", "first"]
    )
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-png", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.sequence) != bool(args.smiles):
        print(f"[{FAIL}] --sequence and --smiles must be given together")
        return 2
    return cmd_plot_ibam(args)


if __name__ == "__main__":
    raise SystemExit(main())
