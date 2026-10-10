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


def _panel_contamination(args, trained_on: str | None) -> dict[str, float]:
    """Share of each affinity panel that is in this regime's TRAINING set.

    Measured against the split's own id files rather than assumed from the
    regime name, because the question is empirical and the answer decides
    whether a one-way door gets opened. A temporally-trained model returns
    nothing; ``random_pair`` overlapped the three panels by 60.2%, 100.0% and
    67.7% when this was first measured.
    """
    import gzip

    if not trained_on or trained_on == "temporal":
        return {}
    ids_path = Path(args.splits) / trained_on / "ids" / "train.txt.gz"
    if not ids_path.exists():
        return {}
    try:
        import pandas as pd

        with gzip.open(ids_path, "rt") as handle:
            train_ids = {line.strip() for line in handle if line.strip()}
        overlap: dict[str, float] = {}
        for name in AFFINITY_PANELS:
            queries = Path(args.benchmarks) / name / "queries.parquet"
            if not queries.exists():
                continue
            frame = pd.read_parquet(queries)
            column = next(
                (c for c in frame.columns if "observation" in c.lower()), frame.columns[0]
            )
            panel_ids = set(frame[column].astype(str))
            if panel_ids:
                share = len(panel_ids & train_ids) / len(panel_ids)
                if share > 0:
                    overlap[name] = share
        return overlap
    except Exception:  # noqa: BLE001 - a guard that crashes must not block scoring
        return {}


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
    """Run the model over a loader and return frozen arrays. No labels opened.

    The two bounds come back alongside the point label because a censored row
    has no point label at all: ``p_value`` is not a measurement there, and the
    only answerable question is whether the prediction respects the half-line
    the bound allows. Dropping them here is what left a fifth of the corpus
    unexamined in every earlier report.
    """
    import numpy as np
    import torch

    truth, predicted, endpoint, exact, ids = [], [], [], [], []
    lower, upper = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device.type if hasattr(device, "type") else "cuda",
                                dtype=torch.bfloat16, enabled=amp):
                out = model(
                    batch.protein_features,
                    batch.protein_mask,
                    batch.ligands,
                    assay=batch.assay,
                )
            values = out.prediction.observed(batch.targets.endpoint_index)
            predicted.append(values.float().cpu().numpy())
            truth.append(batch.targets.p_value.cpu().numpy())
            endpoint.append(batch.targets.endpoint_index.cpu().numpy())
            exact.append(batch.targets.is_exact.cpu().numpy())
            lower.append(batch.targets.lower_bound.float().cpu().numpy())
            upper.append(batch.targets.upper_bound.float().cpu().numpy())
            ids.extend(batch.observation_ids)
    return {
        "predicted": np.concatenate(predicted),
        "truth": np.concatenate(truth),
        "endpoint": np.concatenate(endpoint),
        "is_exact": np.concatenate(exact),
        "lower_bound": np.concatenate(lower),
        "upper_bound": np.concatenate(upper),
        "observation_id": ids,
    }


def _row_grouping(processed, dataset, observation_ids):
    """Protein component and construct id per predicted row.

    Two grouping keys come back, in the row order of the *prediction arrays*
    rather than the frame's:

    ``components``
        The cluster-bootstrap resampling unit. It has to be the protein
        component, not the observation: 10,474 targets carry only 572 effective
        independent units and the top 1% of targets hold 34.2% of the data, so
        a bootstrap over rows would treat correlated measurements as
        independent evidence and return an interval far too narrow. Components
        come from the curation layer's own clustering rather than being
        recomputed here.
    ``constructs``
        The per-target macro-average unit.

    Both are resolved with pandas reindex over an index built once. The obvious
    spelling -- a dict comprehension per row -- is quadratic on a partition this
    size: 235,473 rows each rebuilding a 235,473-entry mapping is 5.5e10
    operations, which does not finish.

    Returns ``(None, None, reason)`` rather than a guess when the mapping
    cannot be built. A wrong clustering is worse than an absent one, because it
    still looks like a clustered interval.
    """
    from pathlib import Path

    import pandas as pd

    path = Path(processed) / "protein_groups.parquet"
    if not path.exists():
        return None, None, f"no protein_groups.parquet under {processed}"
    try:
        order = pd.Index(observation_ids)
        frame = dataset.frame.drop_duplicates("observation_id").set_index("observation_id")
        constructs = frame["sequence_sha256"].reindex(order)

        groups = (
            pd.read_parquet(path, columns=["sequence_sha256", "protein_group"])
            .drop_duplicates("sequence_sha256")
            .set_index("sequence_sha256")["protein_group"]
        )
        components = groups.reindex(pd.Index(constructs.to_numpy()))

        missing = int(components.isna().sum())
        if missing:
            return None, None, f"{missing:,} of {len(order):,} rows have no protein component"
        return components.to_numpy(), constructs.to_numpy(), None
    except Exception as exc:  # noqa: BLE001 - a missing clustering must not stop scoring
        return None, None, f"{type(exc).__name__}: {exc}"


def _metrics(
    truth,
    predicted,
    endpoint,
    is_exact,
    *,
    clusters=None,
    lower_bound=None,
    upper_bound=None,
    targets=None,
    bootstrap: int = 0,
    bootstrap_seed: int = 20261010,
) -> dict:
    """Regression, ranking and the strong/weak call, per endpoint and overall.

    The four positional arguments are the whole contract; everything beyond
    them adds a section and is skipped when not supplied, so a caller that has
    only predictions and labels still gets the table it always got.

    ``clusters`` turns on the cluster-bootstrap intervals, ``targets`` the
    per-target macro averages, and the two bounds the censored section. Each is
    reported as absent rather than silently omitted -- a reader cannot tell the
    difference between "not computed" and "nothing to report" unless the report
    says which.
    """
    import numpy as np
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    from bindra.eval import metrics as M
    from bindra.heads.affinity import ENDPOINT_ORDER
    from bindra.training.losses.affinity import noise_floor_table

    floors = noise_floor_table()

    def block(y, p, label, floor=None, rows=None):
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

        # -- the sections that need more than labels and predictions --------
        entry["classification"] = M.classification_metrics(y, p)
        entry["calibration"] = M.calibration_curve(y, p)
        if bootstrap:
            entry["intervals"] = M.bootstrap_metrics(
                y,
                p,
                None if clusters is None else clusters[rows],
                n_resamples=bootstrap,
                seed=bootstrap_seed,
            )
        else:
            entry["intervals"] = M.undefined("bootstrap disabled (--no-bootstrap)")
        if targets is not None:
            entry["macro_by_target"] = M.macro_by_group(y, p, targets[rows])
        if clusters is not None:
            entry["macro_by_protein_component"] = M.macro_by_group(
                y, p, clusters[rows], label="component"
            )
        return entry

    keep = is_exact.astype(bool)
    report = {
        "overall": block(
            truth[keep], predicted[keep], "overall", rows=np.flatnonzero(keep)
        )
    }
    for column, name in enumerate(ENDPOINT_ORDER):
        mask = keep & (endpoint == column)
        entry = block(
            truth[mask],
            predicted[mask],
            name.value,
            floors[name.value],
            rows=np.flatnonzero(mask),
        )
        if entry:
            report[name.value] = entry

    censored = ~keep
    if censored.sum():
        if lower_bound is None or upper_bound is None:
            report["censored"] = {
                "n": int(censored.sum()),
                "note": "bounds were not supplied to this call, so no violation "
                        "rate could be computed",
            }
        else:
            report["censored"] = M.censored_metrics(
                predicted[censored], lower_bound[censored], upper_bound[censored]
            )
    return report


def _render(title: str, report: dict) -> None:
    """The terminal view. Detail belongs in the JSON, not here.

    Point estimates carry their clustered interval where one was computed,
    because a number without it reads as more settled than it is -- which is
    the entire reason the interval exists.
    """
    from bindra.training.losses.affinity import noise_floor_table

    floors = noise_floor_table()
    print(f"\n  +-- {title}")
    overall = report.get("overall")
    if not overall:
        print("  |   too few rows to score")
        return

    def interval(entry: dict, metric: str) -> str:
        block = (entry.get("intervals") or {}).get("metrics", {}).get(metric)
        bounds = (block or {}).get("cluster")
        if not bounds or bounds.get("status") == "undefined":
            return ""
        return f" [{bounds['low']:.3f}, {bounds['high']:.3f}]"

    print(
        f"  |   overall  n={overall['n']:>8,}  RMSE {overall['rmse']:.4f}"
        f"{interval(overall, 'rmse')}"
    )
    print(
        f"  |            CI {overall['concordance_index']:.3f}"
        f"{interval(overall, 'concordance_index')}"
        f"   Spearman {overall['spearman']:.3f}"
        f"{interval(overall, 'spearman')}"
    )
    classification = overall.get("classification") or {}
    if "auroc" in classification:
        print(
            f"  |            AUROC {classification['auroc']:.3f}"
            f"{interval(overall, 'auc_100nM_vs_rest')}"
            f"   AUPRC {classification['auprc']:.3f}"
            f" (baseline {classification['auprc_baseline']:.3f})"
        )
    intervals = overall.get("intervals") or {}
    if intervals.get("n_clusters"):
        print(
            f"  |            intervals: {intervals['interval']} over "
            f"{intervals['n_clusters']:,} protein components, "
            f"{intervals['n_resamples']:,} resamples"
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

    # The rows every regression metric above dropped.
    censored = report.get("censored")
    if censored and "overall" in censored and censored.get("overall"):
        block = censored["overall"]
        print(
            f"  |   censored n={censored['n']:>8,}  "
            f"bound violated {block['violation_rate']:.1%}  "
            f"mean miss {block['mean_violation_log_units']:.3f} log units"
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
        ligand_builder=getattr(args, "ligand_builder", None),
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
    import dataclasses
    import os

    import torch

    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.encoders.ligand_featurizers import LigandBatchBuilder
    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
        metadata_for_cache,
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

    # Shaped to whatever layer set the cache holds, so a multi-layer cache
    # opens here without the caller restating how it was built.
    metadata = metadata_for_cache(args.protein_cache)
    cache = ProteinFeatureCache(args.protein_cache, metadata=metadata)

    # The featurizer is derived from the model's own ligand config, exactly as
    # `bindra train` does it, so scoring cannot featurize for one arm while
    # encoding with another. Scoring under a different featurizer than training
    # used would be a silent wrong answer, not an error.
    builder = LigandBatchBuilder.for_config(config.ligand)
    if args.knowledge_cache:
        builder = dataclasses.replace(builder, knowledge_cache=args.knowledge_cache)
        print(f"[{OK}] knowledge cache {args.knowledge_cache}", flush=True)
    else:
        print(
            f"[{WARN}] no knowledge cache; descriptors will be computed live, "
            "which is roughly 81% of featurization",
            flush=True,
        )
    args.ligand_builder = builder

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
        ligand_builder=builder,
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

    import numpy as np

    clusters, targets, why_not = _row_grouping(
        args.processed, dataset, frozen["observation_id"]
    )
    if clusters is None:
        print(
            f"[{WARN}] no clustered intervals: {why_not}. Only the i.i.d. "
            "interval will be reported, and it understates the uncertainty here.",
            flush=True,
        )
    else:
        print(
            f"[{OK}] {len(np.unique(clusters)):,} protein components over "
            f"{len(clusters):,} rows (the bootstrap resamples components); "
            f"{len(np.unique(targets)):,} constructs for the macro averages",
            flush=True,
        )

    resamples = 0 if args.no_bootstrap else args.bootstrap
    if resamples:
        print(
            f"[{INFO}] bootstrapping {resamples:,} resamples; this is the slow "
            "part of the command",
            flush=True,
        )
    started = time.perf_counter()
    report = _metrics(
        frozen["truth"],
        frozen["predicted"],
        frozen["endpoint"],
        frozen["is_exact"],
        clusters=clusters,
        lower_bound=frozen["lower_bound"],
        upper_bound=frozen["upper_bound"],
        targets=targets,
        bootstrap=resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"[{OK}] scored in {time.perf_counter() - started:.0f}s", flush=True)
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
    parser.add_argument(
        "--knowledge-cache", default="data/cache/knowledge",
        help="Precomputed fingerprints and descriptors. Descriptors are 81%% of "
             "featurization, so without this a full partition takes hours "
             "rather than minutes. Pass an empty string to compute them live.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Resamples for the cluster bootstrap over protein components. "
             "The intervals are the expensive part of this command.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Point estimates only, for a quick look.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20261010)
    parser.add_argument("--residue-budget", type=int, default=49152)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--featurize-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_evaluate(build_parser().parse_args(argv))


__all__ = ["build_parser", "cmd_evaluate", "main"]
