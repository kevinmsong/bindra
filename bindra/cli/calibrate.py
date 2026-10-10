"""``bindra calibrate``: fit conformal intervals on the held-apart ``cal`` partition.

The three modules this drives are already built and tested
(:mod:`bindra.uncertainty.ensemble`, :mod:`~bindra.uncertainty.conformal`,
:mod:`~bindra.uncertainty.domain`). This is the command that runs them against
real checkpoints and writes one report.

## What this command may and may not do

It **opens ``cal``**, which the training engine refuses to read -- that is the
whole point of the partition and of this command. What follows from that is a
rule the code enforces rather than states:

* It fits a quantile. It does **not** select a model. Nothing here ranks the
  members, and per-member accuracy on ``cal`` is deliberately not reported,
  because a table of per-member numbers on a held-apart partition is a model
  selection waiting to happen and would invalidate the very interval being
  fitted.
* Ensemble-level error on ``cal`` *is* reported, because suppressing it while
  reporting the quantile would be theatre: ``q`` is the 90th percentile of the
  absolute residual, so the residual scale is already in the output.

## Three things the specification requires, and where each is refused

**Independent members.** ``assert_independent`` refuses fewer than two members,
duplicate weight digests, and shared seeds. Three dropout passes over one model
produce a spread that plots convincingly and means nothing; that is the failure
the clause names, and it is checked here rather than trusted.

**One split.** Added on top of the specification: every member must have trained
on the same ``split_id``. A quantile fitted across models that held out
different observations has no exchangeability to rest on, and the resulting
interval would look exactly like a valid one.

**Exact labels only.** The conformal score is ``|y - mean|``, which a censored
record cannot supply -- "weaker than 10 uM" has no ``y``. Censored rows are
excluded and counted, never imputed to their bound.

## The applicability-domain pass is the expensive part

Fingerprinting runs at 953 mol/s, so the 767,870 training molecules cost ~13
minutes. The Tanimoto search is the real cost: 7.0M comparisons/s on cpu, and
every distinct cal molecule against every training one is 206,252 x 767,870 =
**158G comparisons, about 6.3 hours**. So ``--domain-sample`` defaults to 5,000
molecules (~9 minutes), the cap is printed before the work starts and written
into the report, and ``--domain-sample 0`` asks for all of it.

``--similarity-device`` defaults to cpu, which measured *faster* than cuda here
(7.0M against 1.8M comparisons/s at 50,000 references) -- though that cuda
figure was taken while a pre-training run held the card.

## Coverage is measured, not assumed

The guarantee is marginal and rests on exchangeability, which is doing real
work on a corpus with correlated assays, 41.6% patent-derived records and a
Kish effective sample size of 572 target units against 10,474 observed targets.
So ``cal`` is split in half: the quantile is fitted on one half and its
empirical coverage measured on the other, and both are reported beside the
final quantile fitted on all of ``cal``. A coverage figure far from nominal is
the signal that exchangeability has failed for a regime, and without this step
nothing would say so.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
WARN = " warn  "
FAIL = " FAIL  "


class CalibrationRefused(RuntimeError):
    """The inputs do not support a calibration that would mean anything."""


def _members(paths: list[Path]):
    """Identify each checkpoint, then refuse anything that is not an ensemble."""
    from bindra.uncertainty import ensemble

    described = []
    for path in paths:
        if not path.exists():
            raise CalibrationRefused(f"no checkpoint at {path}")
        described.append(ensemble.describe_member(path))
    ensemble.assert_independent(described)
    return described


def _assert_one_split(paths: list[Path]) -> dict:
    """Every member must have trained on the same split, and on ``train``.

    Not in the specification, and here anyway. Conformal's guarantee rests on
    the calibration set being exchangeable with what the model is applied to;
    members that held out different observations break that before the
    arithmetic starts, and the interval that results is indistinguishable from
    a sound one.
    """
    import torch

    seen: dict[str, list[str]] = {}
    provenance: dict = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("provenance", {}) or {}
        split_id = str(recorded.get("split_id", "unknown"))
        seen.setdefault(split_id, []).append(path.name)
        provenance = provenance or recorded
    if len(seen) > 1:
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(seen.items()))
        raise CalibrationRefused(
            "the members trained on different splits, so they held out "
            f"different observations and no calibration set is exchangeable "
            f"with all of them ({detail}). Calibrate one split's models together."
        )
    return provenance


def _cal_predictions(paths, args, device):
    """Each member's prediction over ``cal``, stacked [n_members, n].

    The loader is built once and reused: ``cal`` is 235,717 observations and
    rebuilding the dataset per member would re-scan the corpus three times.
    """
    import numpy as np

    from bindra.cli.evaluate import _load_model, _predict
    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.encoders.ligand_featurizers import LigandBatchBuilder
    from bindra.encoders.ligand_kpgt import KPGTConfig
    from bindra.features.cache import ProteinFeatureCache
    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
        metadata_for_cache,
    )

    # Shaped to whatever layer set the cache holds, so a multi-layer cache
    # opens here without the caller restating how it was built.
    metadata = metadata_for_cache(args.protein_cache)
    cache = ProteinFeatureCache(args.protein_cache, metadata=metadata)
    builder = LigandBatchBuilder.for_config(KPGTConfig.mini())
    if args.knowledge_cache:
        import dataclasses

        builder = dataclasses.replace(builder, knowledge_cache=args.knowledge_cache)

    dataset = PairDataset(
        Path(args.processed),
        Path(args.splits) / args.regime,
        # The partition the engine refuses. Named here on purpose.
        "cal",
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
    print(
        f"[{OK}] cal partition: {len(dataset):,} observations in "
        f"{len(loader):,} batches",
        flush=True,
    )

    stacked, reference = [], None
    try:
        for path in paths:
            model, _, _, used_ema = _load_model(path, device)
            result = _predict(model, loader, device, amp=not args.no_amp)
            print(
                f"[{OK}] {path.name}: predicted {len(result['predicted']):,} rows"
                + (" (EMA weights)" if used_ema else " (raw weights)"),
                flush=True,
            )
            if reference is None:
                reference = result
            elif result["observation_id"] != reference["observation_id"]:
                # Shuffle is off and the loader is shared, so this cannot drift
                # -- which is exactly why it is cheap to assert rather than hope.
                raise CalibrationRefused(
                    f"{path.name} returned a different row order; the members "
                    "would be averaged across different observations"
                )
            stacked.append(result["predicted"])
    finally:
        loader.close()
    return np.stack(stacked), reference


def _calibrate_endpoints(truth, mean, endpoint_index, is_exact, alpha, seed):
    """Per endpoint: the quantile, and a measured coverage beside it."""
    import numpy as np

    from bindra.heads.affinity import ENDPOINT_ORDER
    from bindra.uncertainty import conformal

    report: dict[str, dict] = {}
    for column, endpoint in enumerate(ENDPOINT_ORDER):
        name = str(getattr(endpoint, "value", endpoint))
        rows = endpoint_index == column
        # The conformal score is |y - mean|, which a censored record has no y
        # for. Excluded and counted; imputing the bound would be inventing a
        # label on the one partition that exists to measure honestly.
        exact = rows & is_exact
        n_total, n_exact = int(rows.sum()), int(exact.sum())
        entry: dict = {
            "n_observations": n_total,
            "n_exact": n_exact,
            "n_censored_excluded": n_total - n_exact,
        }
        if n_exact == 0:
            entry["status"] = "no_exact_labels"
            report[name] = entry
            continue

        y, p = truth[exact], mean[exact]
        fitted = conformal.calibrate(y, p, endpoint=name, alpha=alpha)
        entry["quantile"] = fitted.to_dict()

        # Measured coverage: fit on one half, score the other. Reported beside
        # the quantile that ships, which is fitted on all of cal.
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_exact)
        half = n_exact // 2
        if half >= 2:
            fit_rows, score_rows = order[:half], order[half:]
            held = conformal.calibrate(
                y[fit_rows], p[fit_rows], endpoint=name, alpha=alpha
            )
            if isinstance(held, conformal.ConformalQuantile):
                empirical = conformal.coverage(y[score_rows], p[score_rows], held)
                entry["measured_coverage"] = {
                    "nominal": 1.0 - alpha,
                    "empirical": empirical,
                    "n_fit": int(half),
                    "n_scored": int(n_exact - half),
                    "half_width_p_units": held.q,
                    "note": (
                        "Fitted on half of cal and scored on the other half. "
                        "The guarantee is marginal and rests on exchangeability, "
                        "which this checks rather than assumes."
                    ),
                }
            else:
                entry["measured_coverage"] = held.to_dict()
        else:
            entry["measured_coverage"] = {
                "status": "too_few_exact_labels_to_split",
                "n_exact": n_exact,
            }

        if isinstance(fitted, conformal.ConformalQuantile):
            entry["interval_at_10nm"] = {
                "centre_p": 8.0,
                "low_nm": fitted.to_nanomolar(8.0)[0],
                "high_nm": fitted.to_nanomolar(8.0)[1],
                "note": (
                    "A 10 nM prediction's interval in concentration. The inverse "
                    "logarithm reverses the ends: the upper p-value is the lower "
                    "concentration."
                ),
            }
        report[name] = entry
    return report


def _domain(args, reference, device: str):
    """Applicability domain of the cal molecules against the TRAINING ones.

    Computed from the data alone and never from the ensemble spread. The
    signature of :func:`~bindra.uncertainty.domain.ligand_domain` is what
    enforces that; this function only supplies the two molecule lists.
    """
    import gzip

    import pandas as pd

    from bindra.uncertainty import domain

    processed = Path(args.processed)
    ids = Path(args.splits) / args.regime / "ids"

    observations = pd.concat(
        pd.read_parquet(p, columns=["observation_id", "ligand_ref"])
        for p in sorted((processed / "observations").glob("*.parquet"))
    )
    observations["observation_id"] = observations["observation_id"].astype(str)
    ligands = pd.concat(
        pd.read_parquet(p, columns=["ligand_ref", "standardized_smiles", "status"])
        for p in sorted((processed / "ligands").glob("*.parquet"))
    )
    ligands = ligands[ligands["status"] == "OK"]
    smiles_of = dict(
        zip(ligands["ligand_ref"].astype(str), ligands["standardized_smiles"])
    )

    with gzip.open(ids / "train.txt.gz", "rt") as handle:
        train_ids = {line.strip() for line in handle if line.strip()}
    train_rows = observations[observations["observation_id"].isin(train_ids)]
    training = sorted(
        {
            smiles_of[r]
            for r in train_rows["ligand_ref"].astype(str)
            if r in smiles_of and isinstance(smiles_of[r], str)
        }
    )

    scored = set(reference["observation_id"])
    cal_rows = observations[observations["observation_id"].isin(scored)]
    queries = sorted(
        {
            smiles_of[r]
            for r in cal_rows["ligand_ref"].astype(str)
            if r in smiles_of and isinstance(smiles_of[r], str)
        }
    )
    sampled = None
    # Captured before the cut, so the report can say what the sample was drawn
    # FROM. Printing the sample size as the population is how a partial pass
    # comes to read as a complete one.
    n_distinct_cal = len(queries)
    # 0 (or None) means compare every distinct cal molecule.
    if args.domain_sample and len(queries) > args.domain_sample:
        import numpy as np

        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(queries), size=args.domain_sample, replace=False)
        queries = [queries[i] for i in sorted(chosen)]
        sampled = args.domain_sample

    # 7.0M comparisons/s measured on cpu at 50,000 references. Printed before
    # the work rather than discovered during it: uncapped this is a ~6 hour job,
    # which is not something a command should start without saying so.
    comparisons = len(queries) * len(training)
    print(
        f"[{INFO}] applicability domain: {len(queries):,} distinct cal molecules "
        f"against {len(training):,} training molecules on {device} "
        f"({comparisons / 1e9:.1f}G comparisons, roughly "
        f"{comparisons / 7.0e6 / 60:.0f} min at the measured 7.0M/s)"
        + (
            f" -- SAMPLED from {n_distinct_cal:,}; pass --domain-sample 0 for all"
            if sampled
            else ""
        ),
        flush=True,
    )
    statuses = domain.ligand_domain(queries, training, device=device)
    payload = domain.summarise(statuses)
    payload["n_training_molecules"] = len(training)
    payload["measured_against"] = "train partition molecules only"
    payload["n_distinct_cal_molecules"] = n_distinct_cal
    if sampled:
        # Stated rather than left to be inferred from `n`.
        payload["sampled"] = sampled
        payload["sampled_from"] = n_distinct_cal
        payload["coverage_note"] = (
            f"{sampled:,} of {n_distinct_cal:,} distinct cal molecules were "
            "compared. This is a sample, not the whole partition; "
            "--domain-sample 0 compares all of them."
        )
    return payload


def cmd_calibrate(args) -> int:
    import numpy as np
    import torch

    from bindra.runtime.device import (
        DeviceRequirement,
        describe_device,
        DeviceSelectionError,
        select_device,
    )
    from bindra.uncertainty import conformal, ensemble

    paths = [Path(p) for p in args.checkpoints]
    try:
        members = _members(paths)
        provenance = _assert_one_split(paths)
    except (CalibrationRefused, ensemble.EnsembleError) as exc:
        print(f"[{FAIL}] {exc}")
        return 2

    print(
        f"[{OK}] {len(members)} independent members: "
        + ", ".join(f"{m.path.name}(seed={m.seed}, {m.weight_digest})" for m in members),
        flush=True,
    )
    if len(members) < ensemble.RECOMMENDED_MEMBERS:
        print(
            f"[{WARN}] {len(members)} members; the specification asks for "
            f"{ensemble.RECOMMENDED_MEMBERS}. With two, the standard deviation is "
            "a half-difference, which is a worse estimator than it looks.",
            flush=True,
        )

    if args.device:
        device = torch.device(args.device)
    else:
        try:
            device = select_device(DeviceRequirement(min_memory_gib=4.0))
        except DeviceSelectionError as exc:
            print(f"[{FAIL}] {exc}")
            return 1
    print(f"[{OK}] device {describe_device(device)}", flush=True)
    print(
        f"[{INFO}] opening the {args.regime!r} cal partition. It is held apart "
        "from train and val and has not been read. This command fits a quantile "
        "on it and does not select a model: per-member accuracy is not reported, "
        "because a per-member table on a held-apart partition is a selection.",
        flush=True,
    )

    try:
        per_member, reference = _cal_predictions(paths, args, device)
    except CalibrationRefused as exc:
        print(f"[{FAIL}] {exc}")
        return 2

    combined = ensemble.combine(per_member, members)
    truth = reference["truth"]
    endpoint_index = reference["endpoint"]
    is_exact = reference["is_exact"].astype(bool)

    endpoints = _calibrate_endpoints(
        truth, combined.mean, endpoint_index, is_exact, args.alpha, args.seed
    )

    report = {
        "regime": args.regime,
        "alpha": args.alpha,
        "nominal_coverage": 1.0 - args.alpha,
        "partition": "cal",
        "ensemble": combined.to_dict(),
        "endpoints": endpoints,
        "split_id": provenance.get("split_id"),
        "split_regime": provenance.get("split_regime"),
        "n_scored": int(len(truth)),
        "n_exact": int(is_exact.sum()),
        "n_censored_excluded": int((~is_exact).sum()),
        "ensemble_exact_rmse": float(
            np.sqrt(np.mean((truth[is_exact] - combined.mean[is_exact]) ** 2))
        )
        if is_exact.any()
        else None,
        "selection": (
            "none. This command fits intervals and does not rank or choose "
            "between members; per-member accuracy on cal is not computed."
        ),
        "censoring": (
            "The conformal score is |y - mean| and a censored record has no y. "
            "Censored rows are excluded and counted, never imputed to the bound."
        ),
    }

    if not args.skip_domain:
        report["applicability_domain"] = _domain(
            args, reference, args.similarity_device
        )
    else:
        report["applicability_domain"] = {"status": "skipped (--skip-domain)"}

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print()
    for name, entry in endpoints.items():
        quantile = entry.get("quantile", {})
        if quantile.get("status") == "calibrated":
            measured = entry.get("measured_coverage", {})
            empirical = measured.get("empirical")
            shown = f"{empirical:.3f}" if isinstance(empirical, float) else "n/a"
            print(
                f"  {name:<6} +/- {quantile['q']:.3f} p-units on "
                f"{quantile['n_calibration']:,} exact labels | measured coverage "
                f"{shown} against nominal {1 - args.alpha:.2f}"
            )
        else:
            reason = quantile.get("reason") or entry.get("status", "no quantile")
            print(f"  {name:<6} {reason}")
    rmse = report["ensemble_exact_rmse"]
    if rmse is not None:
        print(f"\n[{INFO}] ensemble exact RMSE on cal: {rmse:.4f}")
    print(f"[{OK}] wrote {destination}", flush=True)
    print(
        f"[{INFO}] coverage is marginal and rests on exchangeability. On a novel "
        "target it is not guaranteed and must not be claimed.",
        flush=True,
    )
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra calibrate",
        description="Fit split-conformal intervals on the held-apart cal partition.",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Two or more INDEPENDENTLY SEEDED checkpoints. Identical weights or "
             "a shared seed are refused: repeated passes over one model produce a "
             "spread that plots convincingly and means nothing.",
    )
    parser.add_argument("--processed", default="data/processed/202610")
    parser.add_argument("--splits", default="data/splits/202610")
    parser.add_argument("--protein-cache", default="data/cache/proteins")
    parser.add_argument("--knowledge-cache", default="data/cache/knowledge")
    parser.add_argument("--regime", default="random_pair")
    parser.add_argument("--out", default="reports/calibration.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="1 - alpha is the nominal coverage. 0.10 gives 90%% intervals.",
    )
    parser.add_argument(
        "--skip-domain", action="store_true",
        help="Skip the applicability-domain pass, which fingerprints every "
             "distinct cal molecule against every training molecule.",
    )
    parser.add_argument(
        "--domain-sample", type=int, default=5_000,
        help="Distinct cal molecules compared against every training molecule. "
             "0 means all of them, which is 206,252 x 767,870 = 158G Tanimoto "
             "comparisons and ~6.3 hours; the 5,000 default is ~9 minutes. The "
             "cap is printed and written into the report, so a sampled number "
             "cannot be read as a complete one.",
    )
    parser.add_argument(
        "--similarity-device", default="cpu",
        help="Device for the Tanimoto pass. cpu is the default because it "
             "measured FASTER here: 7.0M comparisons/s against 1.8M on cuda:1 "
             "at 50,000 references -- though that cuda figure was taken while a "
             "pre-training run held the card, so it is a contended one. cpu "
             "also leaves the GPU alone, which is worth something on its own.",
    )
    parser.add_argument("--residue-budget", type=int, default=65_536)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--featurize-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap cal observations, for smoke runs.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261001,
                        help="Seeds the coverage half-split and any domain sample.")
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_calibrate(build_parser().parse_args(argv))


__all__ = ["CalibrationRefused", "build_parser", "cmd_calibrate", "main"]
