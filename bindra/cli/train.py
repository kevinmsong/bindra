"""``bindra train``: train one model on one split regime.

Deliberately narrow. It builds the train and validation loaders for a named
split, constructs the model with widths read off the caches that will feed it,
collects the hashes that identify everything the run depended on, and hands the
rest to :class:`~bindra.training.engine.Trainer`.

It does not touch the test or calibration partitions, and the engine refuses
them independently. The locked benchmark panels stay sealed; scoring against
them is a separate command that does not exist yet, on purpose, because it
should be run once against a model that is finished rather than repeatedly
against one that is being tuned.

Provenance written into every checkpoint and into ``history.json``: the curation
profile hash, the observations table's combined sha256, the split id and its
manifest hash, the protein encoder's full metadata including its weights
checksum, the ligand encoder's metadata, and the batching and training configs.
A number produced by this command can be traced to the corpus, the split, the
weights and the settings that produced it, or it is not reportable.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
FAIL = " FAIL  "


def _protein_metadata(processed: Path):
    """Metadata of the frozen ESM-C contract that populated the cache."""
    from bindra.encoders.protein_esmc import (
        ESMC_600M_WEIGHTS_SHA256,
        EXPECTED_ESMC_600M_DIM,
        ESMCProteinEncoder,
    )

    return ESMCProteinEncoder(n_trainable_layers=0).metadata(
        EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
    )


def _provenance(processed: Path, split_directory: Path, protein_metadata, ligand_metadata):
    from bindra.config.profiles import get_profile

    payload = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (split_directory / "manifest.json").read_text(encoding="utf-8")
    )
    profile = get_profile()
    return {
        "snapshot_release": profile.snapshot_release,
        "curation_profile_hash": profile.profile_hash,
        "observations_combined_sha256": payload["artifacts"]["observations"][
            "combined_sha256"
        ],
        "observations_rows": payload["artifacts"]["observations"]["rows"],
        "split_regime": manifest["regime"],
        "split_id": manifest["split_id"],
        "split_manifest_sha256": manifest["manifest_sha256"],
        "split_seed": manifest["seed"],
        "protein_encoder": protein_metadata.to_dict(),
        "ligand_encoder": ligand_metadata.to_dict(),
    }


def cmd_fit(args) -> int:
    import torch

    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.encoders.ligand_gat import GATConfig, GATLigandEncoder
    from bindra.features.cache import ProteinFeatureCache
    from bindra.heads.affinity import AffinityHeadConfig
    from bindra.model import BindraConfig, BindraModel
    from bindra.runtime.device import DeviceRequirement, select_device
    from bindra.training.engine import Trainer, TrainingConfig
    from bindra.training.losses.affinity import noise_floor_table

    processed = Path(args.processed)
    split_directory = Path(args.splits) / args.regime
    run_directory = Path(args.out) / f"{args.regime}-{args.tag}" if args.tag else Path(args.out) / args.regime

    if not (split_directory / "manifest.json").exists():
        print(f"[{FAIL}] no split at {split_directory}; run `bindra make-splits build`")
        return 1

    device = (
        torch.device(args.device)
        if args.device
        else select_device(DeviceRequirement(min_memory_gib=8.0))
    )
    print(f"[{OK}] device {device}", flush=True)

    protein_metadata = _protein_metadata(processed)
    cache = ProteinFeatureCache(args.protein_cache, metadata=protein_metadata)
    print(
        f"[{OK}] protein cache: {len(cache):,} constructs, "
        f"{cache.total_bytes / 2**30:.3f} GiB, dim {protein_metadata.output_dim}",
        flush=True,
    )

    print(f"[{INFO}] building loaders (one pass over the corpus)...", flush=True)
    train = PairDataset(
        processed, split_directory, "train", protein_cache=cache, limit=args.limit
    )
    validation = PairDataset(
        processed,
        split_directory,
        "val",
        protein_cache=cache,
        limit=None if args.limit is None else max(1, args.limit // 8),
    )
    policy = BatchingPolicy(
        residue_budget=args.residue_budget, max_pairs=args.max_pairs, seed=args.seed
    )
    train_loader = PairLoader(train, policy, prefetch_workers=args.workers)
    val_loader = PairLoader(
        validation, policy, shuffle=False, prefetch_workers=args.workers
    )
    print(
        f"[{OK}] train {len(train):,} pairs in {len(train_loader):,} batches | "
        f"val {len(validation):,} pairs in {len(val_loader):,} batches",
        flush=True,
    )
    print(f"[{INFO}] batch shape: {train_loader.describe()}", flush=True)

    ligand = GATConfig(
        hidden_dim=args.ligand_dim, num_layers=args.ligand_layers, num_heads=8
    )
    config = BindraConfig(
        protein_dim=protein_metadata.output_dim,
        ligand=ligand,
        model_dim=args.model_dim,
        cross_attention_layers=args.fusion_layers,
        cross_attention_heads=8,
        dropout=args.dropout,
        head=AffinityHeadConfig(hidden_dim=args.model_dim, dropout=args.dropout),
    )
    model = BindraModel(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{OK}] model: {trainable:,} trainable parameters", flush=True)

    ligand_metadata = GATLigandEncoder(ligand, module=model.ligand_encoder).metadata()
    provenance = _provenance(processed, split_directory, protein_metadata, ligand_metadata)
    provenance["model"] = config.to_dict()
    provenance["batching"] = policy.to_dict()

    training = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        ema_decay=args.ema,
        seed=args.seed,
        log_every=args.log_every,
    )
    trainer = Trainer(
        model,
        training,
        device=device,
        run_directory=run_directory,
        provenance=provenance,
    )
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        print(f"[{OK}] resumed from {args.resume} at epoch {trainer.epoch}", flush=True)

    print(f"[{INFO}] noise floor (RMSE at which a model is not skilful): {noise_floor_table()}")
    print(f"[{INFO}] writing to {run_directory}", flush=True)
    trainer.fit(train_loader, val_loader)

    best = trainer.best_metric
    print(
        f"[{OK}] done. best val exact RMSE {best:.4f}"
        if best is not None
        else f"[{OK}] done.",
        flush=True,
    )
    print(
        f"[{INFO}] no benchmark panel was opened; scoring is a separate, "
        "once-only step against a finished model."
    )
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra train", description="Train one model on one split regime."
    )
    parser.add_argument("--processed", default="data/processed/202610")
    parser.add_argument("--splits", default="data/splits/202610")
    parser.add_argument("--protein-cache", default="data/cache/proteins")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--regime", default="random_pair")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default=None, help="Defaults to the largest CUDA device.")
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs, for smoke runs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--ligand-dim", type=int, default=256)
    parser.add_argument("--ligand-layers", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--residue-budget", type=int, default=65_536)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_fit(build_parser().parse_args(argv))


__all__ = ["build_parser", "cmd_fit", "main"]
