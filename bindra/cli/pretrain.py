"""``bindra pretrain``: knowledge-guided pre-training for the LiGhT arm.

The objective and the loop live in :mod:`bindra.training.pretrain`; this is the
command that assembles them. What it adds is the part that can go silently
wrong: deciding **which molecules may be pretrained on**.

The specification allows only training partitions to supply chemistry
pre-training. Self-supervision leaks no labels, but it does leak structure, so
:func:`~bindra.training.pretrain.train_only_molecules` keeps molecules that
appear in ``train`` and in no other partition -- about half the corpus for
``random_pair``. That set is a property of one split, which is why the regime
and the split id are written into the checkpoint and checked when
``bindra train`` loads it.

The held-out slice used to pick a checkpoint comes out of those *training*
molecules, never from the affinity split's ``val`` partition. Selecting on val
molecules would put their structures inside the encoder by the back door, which
is the same leak arriving through the selection rather than the gradient.

## Order

Pre-training is cheap relative to affinity training -- no protein side, so no
144 MiB residue tensors and batches an order of magnitude larger -- and it runs
**before** the affinity run, not after. Running it afterwards means training the
affinity model twice.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
FAIL = " FAIL  "

#: Cached molecule list, written beside the run so a resumed run does not
#: re-scan the corpus. Keyed to the split that produced it; a different
#: ``split_id`` invalidates it, because the held-out molecules moved.
MOLECULE_CACHE = "molecules.txt.gz"
MOLECULE_CACHE_META = "molecules.json"


def _split_provenance(processed: Path, split_directory: Path) -> dict:
    """What this run depended on, recorded so the checkpoint can be traced."""
    from bindra.config.profiles import get_profile

    manifest = json.loads((split_directory / "manifest.json").read_text(encoding="utf-8"))
    profile = get_profile()
    payload = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    return {
        "snapshot_release": profile.snapshot_release,
        "curation_profile_hash": profile.profile_hash,
        "observations_combined_sha256": payload["artifacts"]["observations"][
            "combined_sha256"
        ],
        "split_regime": manifest["regime"],
        "split_id": manifest["split_id"],
        "split_manifest_sha256": manifest["manifest_sha256"],
        "split_seed": manifest["seed"],
    }


def _molecules(run_directory: Path, processed: Path, splits: Path, regime: str,
               split_id: str, refresh: bool = False) -> list[str]:
    """The train-only molecule list, computed once and cached beside the run.

    Scanning the observations and ligands tables costs minutes; a thirty-epoch
    run that is resumed four times should not pay it five times. The cache
    records the ``split_id`` it was built from and is ignored when that changes,
    because a rebuilt split holds out different molecules.
    """
    import gzip

    from bindra.training.pretrain import train_only_molecules

    cache = run_directory / MOLECULE_CACHE
    meta = run_directory / MOLECULE_CACHE_META
    if cache.exists() and meta.exists() and not refresh:
        recorded = json.loads(meta.read_text(encoding="utf-8"))
        if recorded.get("split_id") == split_id and recorded.get("regime") == regime:
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                molecules = [line.strip() for line in handle if line.strip()]
            print(
                f"[{OK}] {len(molecules):,} train-only molecules read from "
                f"{cache.name} (split {split_id})",
                flush=True,
            )
            return molecules
        print(
            f"[{INFO}] {cache.name} was built for split "
            f"{recorded.get('split_id')!r}, not {split_id!r}; rebuilding",
            flush=True,
        )

    print(
        f"[{INFO}] selecting molecules in {regime}/train and in no other "
        "partition (one pass over the corpus)...",
        flush=True,
    )
    molecules = train_only_molecules(processed, splits, regime)
    run_directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(molecules))
    meta.write_text(
        json.dumps(
            {"regime": regime, "split_id": split_id, "n_molecules": len(molecules)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return molecules


def cmd_pretrain(args) -> int:
    import dataclasses

    import torch

    from bindra.encoders.ligand_featurizers import LigandBatchBuilder
    from bindra.encoders.ligand_kpgt import KPGTConfig
    from bindra.encoders.ligand_kpgt_arm import KPGTLigandArmModule
    from bindra.runtime.device import (
        DeviceRequirement,
        describe_device,
        DeviceSelectionError,
        select_device,
    )
    from bindra.training.pretrain import (
        MoleculeLoader,
        PretrainConfig,
        PretrainError,
        PretrainRunner,
        split_molecules,
    )

    processed = Path(args.processed)
    splits = Path(args.splits)
    split_directory = splits / args.regime
    if not (split_directory / "manifest.json").exists():
        print(f"[{FAIL}] no split at {split_directory}; run `bindra make-splits build`")
        return 1

    name = f"pretrain-{args.regime}-{args.tag}" if args.tag else f"pretrain-{args.regime}"
    run_directory = Path(args.out) / name

    if args.device:
        device = torch.device(args.device)
    else:
        try:
            device = select_device(DeviceRequirement(min_memory_gib=4.0))
        except DeviceSelectionError as exc:
            print(f"[{FAIL}] {exc}")
            return 1
    print(f"[{OK}] device {describe_device(device)}", flush=True)

    provenance = _split_provenance(processed, split_directory)
    molecules = _molecules(
        run_directory, processed, splits, args.regime,
        provenance["split_id"], refresh=args.refresh_molecules,
    )
    if args.limit:
        molecules = molecules[: args.limit]
        print(f"[{INFO}] limited to {len(molecules):,} molecules", flush=True)
    if len(molecules) < 2:
        print(f"[{FAIL}] {len(molecules)} molecules is not a pre-training set")
        return 1

    ligand = KPGTConfig.mini()
    builder = LigandBatchBuilder.for_config(ligand)
    if args.knowledge_cache:
        directory = Path(args.knowledge_cache)
        if not (directory / "compact-keys.npy").exists():
            print(
                f"[{FAIL}] no compact index at {directory}. Pre-training reads "
                "fingerprints and descriptors for every molecule every epoch, and "
                "computing them live costs 57.74 ms each. Build it with "
                "`bindra cache knowledge --compact-index`, or pass "
                "--knowledge-cache '' to accept the cost.",
                flush=True,
            )
            return 2
        builder = dataclasses.replace(builder, knowledge_cache=str(directory))
        print(
            f"[{OK}] knowledge cache at {directory}: both targets read rather "
            "than recomputed",
            flush=True,
        )
    else:
        print(
            f"[{INFO}] no knowledge cache; descriptors computed live at ~57.74 "
            "ms/mol, which is 81% of featurization",
            flush=True,
        )

    config = PretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        descriptor_weight=args.descriptor_weight,
        fingerprint_mask_share=args.fingerprint_mask_share,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        log_every=args.log_every,
        verbose=args.verbose,
    )
    train_smiles, val_smiles = split_molecules(
        molecules, val_fraction=config.val_fraction, seed=config.seed
    )
    print(
        f"[{OK}] {len(train_smiles):,} molecules to pretrain on, "
        f"{len(val_smiles):,} held back to select a checkpoint. The held-back "
        "slice comes from train, never from the affinity split's val partition.",
        flush=True,
    )

    arm = KPGTLigandArmModule(ligand)
    provenance.update(
        {
            "n_molecules": len(molecules),
            "n_pretrain_train": len(train_smiles),
            "n_pretrain_val": len(val_smiles),
            "knowledge_cache": str(args.knowledge_cache) or None,
            "ligand_config": ligand.to_dict(),
        }
    )
    runner = PretrainRunner(
        arm, config, device=device, run_directory=run_directory,
        provenance=provenance,
    )
    print(
        f"[{OK}] LiGhT-mini: {runner.n_trainable:,} trainable parameters for this "
        f"objective ({len(runner.frozen)} left frozen: "
        f"{', '.join(runner.frozen) or 'none'})",
        flush=True,
    )

    train_loader = MoleculeLoader(
        train_smiles, builder, batch_size=config.batch_size, seed=config.seed,
        featurize_workers=args.featurize_workers,
        prefetch_workers=args.prefetch_workers,
    )
    val_loader = MoleculeLoader(
        val_smiles, builder, batch_size=config.batch_size, shuffle=False,
        seed=config.seed, featurize_workers=args.featurize_workers,
        prefetch_workers=args.prefetch_workers,
    )
    print(
        f"[{OK}] {len(train_loader):,} train batches, {len(val_loader):,} val "
        f"batches at batch size {config.batch_size}",
        flush=True,
    )

    if args.resume:
        runner.load_checkpoint(Path(args.resume))
        print(
            f"[{OK}] resumed from {args.resume} at epoch {runner.epoch}, "
            f"step {runner.global_step}",
            flush=True,
        )

    print(f"[{INFO}] writing to {run_directory}", flush=True)
    print(
        f"[{INFO}] objective: mask one knowledge node per molecule and predict it "
        "from the graph plus the other. Fingerprint accuracy is reported beside "
        "the majority-bit baseline, never alone: these fingerprints are dense "
        "(83.9% of bits set), so a constant predictor already scores 0.839.",
        flush=True,
    )
    try:
        runner.fit(train_loader, val_loader)
    except PretrainError as exc:
        print(f"[{FAIL}] {exc}")
        return 1
    finally:
        train_loader.close()
        val_loader.close()

    dropped = len(train_loader.dropped) + len(val_loader.dropped)
    if dropped:
        print(
            f"[{INFO}] {dropped} molecule(s) would not featurize and were dropped; "
            f"first: {(train_loader.dropped + val_loader.dropped)[:3]}",
            flush=True,
        )
    best = runner.best_metric
    print(
        f"[{OK}] done. best held-out loss {best:.4f}"
        if best is not None
        else f"[{OK}] done.",
        flush=True,
    )
    print(
        f"[{INFO}] train the affinity model from it with\n"
        f"         bindra train --regime {args.regime} "
        f"--ligand-checkpoint {run_directory / 'best.pt'}",
        flush=True,
    )
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra pretrain",
        description="Knowledge-guided pre-training for the LiGhT ligand arm.",
    )
    parser.add_argument("--processed", default="data/processed/202610")
    parser.add_argument("--splits", default="data/splits/202610")
    parser.add_argument("--out", default="runs")
    parser.add_argument(
        "--regime", default="random_pair",
        help="Which split defines the train-only molecules. A checkpoint "
             "pretrained for one regime is refused by another, because the two "
             "hold out different molecules.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Measured at ~525 mol/s, so ~17.1 min an epoch over the 540,166 "
             "train-only molecules of random_pair -- 30 epochs is ~8.6 hours. "
             "The cosine schedule is sized to this number, so lowering it is a "
             "different learning-rate curve, not a truncation of the same one.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Molecules per step. There is no protein side here, so this is an "
             "order of magnitude above an affinity batch; measured flat from 256 "
             "to 1024, so the step is compute-bound rather than launch-bound.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--val-fraction", type=float, default=0.02,
        help="Share of the train-only molecules held back to select a "
             "checkpoint. Taken from train, never from the affinity val set.",
    )
    parser.add_argument(
        "--descriptor-weight", type=float, default=1.0,
        help="Weight on the descriptor regression against the fingerprint term. "
             "512 sparse binary targets and 200 dense continuous ones are not "
             "comparable in raw magnitude.",
    )
    parser.add_argument(
        "--fingerprint-mask-share", type=float, default=0.5,
        help="Share of molecules whose fingerprint node is masked; the rest have "
             "their descriptor node masked. Exactly one is masked per molecule.",
    )
    parser.add_argument(
        "--knowledge-cache", default="data/cache/knowledge",
        help="Precomputed fingerprints and descriptors. Here they are both the "
             "input and the target, so the cache matters more than it does for "
             "affinity training. Pass '' to compute live.",
    )
    parser.add_argument(
        "--featurize-workers", type=int, default=8,
        help="Processes building triplet graphs; this is what the loop waits "
             "on. Raising it does not help: measured 444 mol/s at 6 workers, "
             "526 at 8, 490 at 16. The pool scales 3.5x across 8 physical "
             "cores, not 8x, and no batch size or prefetch depth moved it.",
    )
    parser.add_argument("--prefetch-workers", type=int, default=2,
                        help="Threads assembling batches ahead of the GPU.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap molecules, for smoke runs.")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--refresh-molecules", action="store_true",
        help="Re-scan the corpus instead of reading the cached molecule list.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_pretrain(build_parser().parse_args(argv))


__all__ = ["build_parser", "cmd_pretrain", "main"]
