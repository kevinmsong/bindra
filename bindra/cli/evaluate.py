"""``bindra evaluate``: score a checkpoint, on a partition or on the locked panels.

The ordering here is the point, and it is enforced rather than documented:
**predictions are computed and frozen before any sealed label is opened.** The
model never sees a held-out label, and the panel's access log records who opened
it and why. A run that crashes between prediction and scoring leaves the
predictions on disk, so re-scoring never requires re-predicting against labels
that are by then known.

Two kinds of evaluation:

``--partition val|cal|test``
    Standard regression and ranking metrics over a split partition. ``val`` is
    free to repeat. ``test`` is not: it is the held-out set, and scoring several
    models against it and keeping the best is how a test set is overfitted. The
    command says so and requires ``--i-mean-it`` for ``test``.

``--panels``
    The prespecified panels under ``data/benchmarks/``. Only the three temporal
    panels are affinity-shaped and scored here. Selectivity, fishing and the
    activity-cliff panel ask different questions -- a difference between two
    targets, a ranking against decoys, a matched-pair contrast -- and are
    **listed as unscored** rather than silently omitted, because a report that
    quietly covers three of seven panels invites the reader to assume seven.

Every number is reported beside the measured replicate-noise floor for its
endpoint, because an RMSE without that comparison invites the wrong reading.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
WARN = " warn  "
FAIL = " FAIL  "

#: Panels whose labels are a p-value per observation, so the affinity head can
#: be scored against them directly.
AFFINITY_PANELS = (
    "panel_strict_future",
    "panel_repeated_measurement",
    "panel_reimport_control",
)

#: Panels that need their own scorer. Named so the report can say what it did
#: not do, rather than leaving a reader to infer coverage from silence.
UNSCORED_PANELS = {
    "selectivity": "a difference between two targets, not a single affinity",
    "censored_selectivity": "an interval-valued difference between two targets",
    "fishing": "ranking actives against measured inactives and matched decoys",
    "activity_cliff": "a matched-pair contrast across a potency step",
}


def _load_model(checkpoint: Path, device):
    """Rebuild the model from the config stored in its own checkpoint."""
    import torch

    from bindra.encoders.ligand_kpgt import KPGTConfig
    from bindra.heads.affinity import AffinityHeadConfig
    from bindra.model import BindraConfig, BindraModel

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = payload.get("provenance", {}).get("model", {})
    ligand = stored.get("ligand", {})
    config = BindraConfig(
        protein_dim=stored.get("protein_dim", 1152),
        model_dim=stored.get("model_dim", 256),
        ligand=KPGTConfig.mini(
            d_g_feats=ligand.get("d_g_feats", 128),
            n_mol_layers=ligand.get("n_mol_layers", 4),
            n_trainable_layers=ligand.get("n_mol_layers", 4),
            n_heads=ligand.get("n_heads", 8),
            d_hpath_ratio=ligand.get("d_hpath_ratio", 4),
        ),
        cross_attention_layers=stored.get("cross_attention_layers", 2),
        cross_attention_heads=stored.get("cross_attention_heads", 8),
        dropout=stored.get("dropout", 0.1),
        # Both default to what the model was *before* these became options, not
        # to today's defaults. A checkpoint written before either existed holds
        # neither key, and rebuilding it under the current defaults would give a
        # different pair_dim and a state dict that will not load.
        feedforward_multiplier=stored.get("feedforward_multiplier", 4),
        split_graph_readout=stored.get("split_graph_readout", False),
        head=AffinityHeadConfig(hidden_dim=stored.get("model_dim", 256)),
    )
    model = BindraModel(config)
    model.load_state_dict(payload["model"])

    # Score under the EMA weights when they exist: those are what validation
    # reported, so scoring the raw ones would report a different model.
    shadow = (payload.get("ema") or {}).get("shadow", payload.get("ema"))
    used_ema = False
    if isinstance(shadow, dict) and shadow:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in shadow:
                    parameter.copy_(shadow[name].to(parameter.dtype))
        used_ema = True
    return model.to(device).eval(), payload, config, used_ema


def _predict(model, loader, device, amp: bool = True):
    """Run the model over a loader and return frozen arrays. No labels opened."""
    import numpy as np
    import torch

    truth, predicted, endpoint, exact, ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device.type if hasattr(device, "type") else "cuda",
                                dtype=torch.bfloat16, enabled=amp):
                out = model(batch.protein_features, batch.protein_mask, batch.ligands)
            values = out.prediction.observed(batch.targets.endpoint_index)
            predicted.append(values.float().cpu().numpy())
            truth.append(batch.targets.p_value.cpu().numpy())
            endpoint.append(batch.targets.endpoint_index.cpu().numpy())
            exact.append(batch.targets.is_exact.cpu().numpy())
            ids.extend(batch.observation_ids)
    return {
        "predicted": np.concatenate(predicted),
        "truth": np.concatenate(truth),
        "endpoint": np.concatenate(endpoint),
        "is_exact": np.concatenate(exact),
        "observation_id": ids,
    }


def _metrics(truth, predicted, endpoint, is_exact) -> dict:
    """Regression, ranking and the strong/weak call, per endpoint and overall."""
    import numpy as np
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    from bindra.heads.affinity import ENDPOINT_ORDER
    from bindra.training.losses.affinity import noise_floor_table

    floors = noise_floor_table()

    def block(y, p, label, floor=None):
        if len(y) < 30:
            return None
        tau = kendalltau(y, p).statistic
        entry = {
            "n": int(len(y)),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "mae": float(np.mean(np.abs(y - p))),
            # Concordance index: the chance a random pair is ordered correctly.
            "concordance_index": float((1 + tau) / 2),
            "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic),
        }
        if floor:
            entry["noise_floor"] = floor
            entry["rmse_over_floor"] = entry["rmse"] / floor
            # What the error means where a chemist works: a fold in concentration.
            entry["typical_fold_error"] = float(10 ** entry["rmse"])
        strong = y >= 7.0
        if 30 < strong.sum() < len(y) - 30:
            entry["auc_100nM_vs_rest"] = float(roc_auc_score(strong.astype(int), p))
        return entry

    keep = is_exact.astype(bool)
    report = {"overall": block(truth[keep], predicted[keep], "overall")}
    for column, name in enumerate(ENDPOINT_ORDER):
        mask = keep & (endpoint == column)
        entry = block(truth[mask], predicted[mask], name.value, floors[name.value])
        if entry:
            report[name.value] = entry

    censored = ~keep
    if censored.sum():
        report["censored"] = {
            "n": int(censored.sum()),
            "note": "bound violations need the bound direction; see training logs",
        }
    return report


def _render(title: str, report: dict) -> None:
    from bindra.training.losses.affinity import noise_floor_table

    floors = noise_floor_table()
    print(f"\n  +-- {title}")
    overall = report.get("overall")
    if overall:
        print(
            f"  |   overall  n={overall['n']:>8,}  RMSE {overall['rmse']:.4f}  "
            f"CI {overall['concordance_index']:.3f}  "
            f"Spearman {overall['spearman']:.3f}"
            + (f"  AUC {overall['auc_100nM_vs_rest']:.3f}"
               if "auc_100nM_vs_rest" in overall else "")
        )
    print(
        f"  |   {'endpoint':<8}{'n':>9}{'RMSE':>9}{'floor':>8}{'x floor':>9}"
        f"{'fold':>8}{'CI':>7}{'rho':>7}"
    )
    for name in ("Kd", "Ki", "IC50"):
        entry = report.get(name)
        if not entry:
            continue
        print(
            f"  |   {name:<8}{entry['n']:>9,}{entry['rmse']:>9.4f}"
            f"{floors[name]:>8.3f}{entry['rmse_over_floor']:>9.2f}"
            f"{entry['typical_fold_error']:>7.0f}x"
            f"{entry['concordance_index']:>7.3f}{entry['spearman']:>7.3f}"
        )


def _score_affinity_panel(name, directory, args, model, device, cache) -> dict:
    """Predict on a panel's queries, freeze, then open its sealed labels.

    The order is the discipline. Predictions are complete and written to disk
    before ``load_sealed_labels`` is called, so the model cannot have been
    influenced by a label, and the panel's access log records the read.
    """
    import numpy as np
    import pandas as pd

    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.eval.benchmarks.lock import load_locked, load_queries, load_sealed_labels

    manifest = load_locked(directory)
    queries = load_queries(directory)
    if args.panel_limit:
        queries = queries.head(args.panel_limit)
    print(f"[{INFO}] {name}: {len(queries):,} queries", flush=True)

    dataset = PairDataset.from_queries(
        args.processed, queries, protein_cache=cache,
        length_multiple=BatchingPolicy().length_multiple,
    )
    loader = PairLoader(
        dataset,
        BatchingPolicy(residue_budget=args.residue_budget, max_pairs=args.max_pairs),
        shuffle=False,
        prefetch_workers=args.workers,
        featurize_workers=args.featurize_workers,
    )
    frozen = _predict(model, loader, device, amp=not args.no_amp)
    loader.close()

    predictions = pd.DataFrame(
        {
            "observation_id": dataset.frame["observation_id"].to_numpy(),
            "endpoint": dataset.frame["endpoint"].to_numpy(),
            "predicted": frozen["predicted"],
        }
    )
    frozen_path = Path(args.out).parent / f"predictions_{name}.parquet"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(frozen_path, index=False)
    print(
        f"[{OK}] {name}: {len(predictions):,} predictions frozen to "
        f"{frozen_path.name}; opening sealed labels",
        flush=True,
    )

    # Only now.
    labels = load_sealed_labels(
        directory,
        allow_labels=True,
        reason=(
            f"scoring checkpoint {Path(args.checkpoint).name} "
            f"(epoch {args.checkpoint_epoch}) against panel {name}"
        ),
    )
    merged = predictions.merge(
        labels[["observation_id", "p_value", "is_exact"]],
        on="observation_id",
        how="inner",
    )
    from bindra.heads.affinity import ENDPOINT_INDEX

    endpoint_index = np.array(
        [ENDPOINT_INDEX[str(e)] for e in merged["endpoint"]], dtype=np.int64
    )
    report = _metrics(
        merged["p_value"].to_numpy(),
        merged["predicted"].to_numpy(),
        endpoint_index,
        merged["is_exact"].to_numpy().astype(bool),
    )
    _render(f"panel {name}", report)
    return {
        "status": "scored",
        "n_queries": int(len(queries)),
        "n_matched": int(len(merged)),
        "manifest_sha256": manifest.manifest_sha256,
        "predictions": str(frozen_path),
        "metrics": report,
    }


def cmd_evaluate(args) -> int:
    import os

    import torch

    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
    )
    from bindra.features.cache import ProteinFeatureCache
    from bindra.runtime.device import DeviceRequirement, select_device

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[{FAIL}] no checkpoint at {checkpoint}")
        return 1

    if args.partition == "test" and not args.i_mean_it:
        print(
            f"[{FAIL}] --partition test is the held-out set. Scoring several "
            "models against it and keeping the best overfits it, and it cannot "
            "be un-seen. Pass --i-mean-it if this is a finished model."
        )
        return 2

    device = torch.device(args.device) if args.device else select_device(
        DeviceRequirement(min_memory_gib=4.0)
    )
    model, payload, config, used_ema = _load_model(checkpoint, device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"[{OK}] device {device}", flush=True)
    print(
        f"[{OK}] checkpoint epoch {payload.get('epoch')}, {trainable:,} parameters"
        f"{', EMA weights' if used_ema else ', raw weights (no EMA in checkpoint)'}",
        flush=True,
    )
    args.checkpoint_epoch = payload.get("epoch")
    best = payload.get("best_metric")
    if best is not None:
        print(f"[{INFO}] recorded best val RMSE during training: {best:.4f}", flush=True)

    results = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "parameters": trainable,
        "used_ema_weights": used_ema,
        "provenance": payload.get("provenance", {}),
        "partition": {},
        "panels": {},
        "panels_not_scored": UNSCORED_PANELS,
    }

    metadata = ESMCProteinEncoder(n_trainable_layers=0).metadata(
        EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
    )
    cache = ProteinFeatureCache(args.protein_cache, metadata=metadata)

    # -- partition ---------------------------------------------------------
    split_directory = Path(args.splits) / args.regime
    print(
        f"[{INFO}] building {args.partition} loader for {args.regime} "
        "(one pass over the corpus)...",
        flush=True,
    )
    dataset = PairDataset(
        args.processed,
        split_directory,
        args.partition,
        protein_cache=cache,
        limit=args.limit,
    )
    loader = PairLoader(
        dataset,
        BatchingPolicy(residue_budget=args.residue_budget, max_pairs=args.max_pairs),
        shuffle=False,
        prefetch_workers=args.workers,
        featurize_workers=args.featurize_workers,
    )
    print(f"[{OK}] {len(dataset):,} pairs in {len(loader):,} batches", flush=True)

    import time

    started = time.perf_counter()
    frozen = _predict(model, loader, device, amp=not args.no_amp)
    loader.close()
    print(
        f"[{OK}] predicted {len(frozen['truth']):,} labels in "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )

    report = _metrics(
        frozen["truth"], frozen["predicted"], frozen["endpoint"], frozen["is_exact"]
    )
    results["partition"] = {
        "regime": args.regime,
        "name": args.partition,
        "metrics": report,
    }
    _render(f"{args.regime} / {args.partition}", report)

    # -- panels ------------------------------------------------------------
    if args.panels:
        print(f"\n[{INFO}] locked panels", flush=True)
        for name in AFFINITY_PANELS:
            directory = Path(args.benchmarks) / name
            if not (directory / "manifest.json").exists():
                print(f"[{WARN}] {name}: not built")
                continue
            try:
                results["panels"][name] = _score_affinity_panel(
                    name, directory, args, model, device, cache
                )
            except Exception as exc:  # noqa: BLE001 - one panel must not stop the rest
                print(f"[{FAIL}] {name}: {type(exc).__name__}: {exc}", flush=True)
                results["panels"][name] = {"status": "failed", "error": str(exc)}
        for name, why in UNSCORED_PANELS.items():
            print(f"[{INFO}] {name}: not scored -- {why}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[{OK}] wrote {out}", flush=True)
    if args.partition != "test":
        print(
            f"[{INFO}] the held-out test partition and the sealed panels were "
            "not opened."
        )
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra evaluate",
        description="Score a checkpoint on a split partition or the locked panels.",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--processed", default="data/processed/202610")
    parser.add_argument("--splits", default="data/splits/202610")
    parser.add_argument("--benchmarks", default="data/benchmarks/202610")
    parser.add_argument("--protein-cache", default="data/cache/proteins")
    parser.add_argument("--regime", default="random_pair")
    parser.add_argument(
        "--partition", default="val", choices=("val", "cal", "test"),
        help="val is repeatable; test is one-shot and needs --i-mean-it.",
    )
    parser.add_argument("--i-mean-it", action="store_true")
    parser.add_argument(
        "--panels", action="store_true",
        help="Score the locked temporal panels. OPENS SEALED LABELS and writes "
             "to each panel's access log.",
    )
    parser.add_argument("--panel-limit", type=int, default=None)
    parser.add_argument("--out", default="runs/evaluation.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--residue-budget", type=int, default=49152)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--featurize-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_evaluate(build_parser().parse_args(argv))


__all__ = ["build_parser", "cmd_evaluate", "main"]
