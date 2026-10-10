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


def _cache_layers(cache_directory):
    """Delegates to the encoder that owns the cache contract."""
    from bindra.encoders.protein_esmc import cache_declared_layers

    return cache_declared_layers(cache_directory)


def _protein_metadata(processed: Path, cache_directory=None):
    """ESM-C metadata shaped to the layer set the cache actually holds."""
    from bindra.encoders.protein_esmc import metadata_for_cache

    return metadata_for_cache(cache_directory or "data/cache/proteins")


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


def _ligand_config(args):
    """The arm's own config, from the arm-specific flags.

    Each arm reads only its own flags, so a GAT width silently applying to a
    KPGT run is not expressible. Both are built here rather than in ``cmd_fit``
    so the loaders and the model can be handed the same object.
    """
    from bindra.encoders.ligand_kpgt import KPGTConfig

    # LiGhT-mini, and only LiGhT-mini. The published 91.9M `base` shape was
    # selectable here and is not any more: without `base.pth` it can only be
    # partial fine-tuning over randomly initialised weights, which `cmd_fit`
    # refused anyway, so the flag's sole reachable outcome was an error. The
    # configuration itself remains expressible in library code -- `KPGTConfig()`
    # still defaults to the published shape, and `load_pretrained` still loads
    # the checkpoint with `strict=True` -- because that is what the port's
    # verification rests on, and it is what this returns to the moment the
    # weights arrive.
    return KPGTConfig.mini()


def _ligand_metadata(ligand, module):
    """Encoder identity for provenance, from whichever arm is installed.

    Routed through each arm's own encoder class so the recorded
    ``feature_schema_version``, ``revision`` and preprocessing policy are the
    arm's own rather than a generic summary. Every results table reads this to
    say which arm produced it.
    """
    from bindra.encoders.ligand_kpgt import KPGTLigandEncoder

    checkpoint = getattr(module, "pretrained_checkpoint", None) or "untrained"
    return KPGTLigandEncoder(
        ligand, module=module.module, checkpoint=checkpoint
    ).metadata()


def cmd_fit(args) -> int:
    import dataclasses
    import os

    # Set before torch initialises CUDA, or it has no effect.
    #
    # Batch shapes here swing from [1, 34350] to [256, 256] because the residue
    # budget trades pairs against protein length, and the default caching
    # allocator fragments badly under that: it reserves segments sized for one
    # shape and cannot serve the next, so a run dies hours in with free memory
    # still showing. Expandable segments let a reservation grow instead of
    # forcing a new one, which is exactly the variable-shape case it exists for.
    # Not supported on Windows -- it warns and does nothing there -- so it is
    # set only where it helps. On Windows the fragmentation is handled instead
    # by BatchingPolicy.length_multiple, which bounds how many distinct tensor
    # shapes a run can produce.
    import sys as _sys

    if not _sys.platform.startswith("win"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from bindra.data.batching import BatchingPolicy, PairDataset, PairLoader
    from bindra.encoders.ligand_featurizers import LigandBatchBuilder
    from bindra.features.cache import ProteinFeatureCache
    from bindra.heads.affinity import AffinityHeadConfig
    from bindra.model import BindraConfig, BindraModel
    from bindra.runtime.device import (
        DeviceRequirement,
        describe_device,
        DeviceSelectionError,
        select_device,
    )
    from bindra.training.engine import Trainer, TrainingConfig
    from bindra.training.losses.affinity import noise_floor_table

    processed = Path(args.processed)
    split_directory = Path(args.splits) / args.regime
    run_directory = Path(args.out) / f"{args.regime}-{args.tag}" if args.tag else Path(args.out) / args.regime

    if not (split_directory / "manifest.json").exists():
        print(f"[{FAIL}] no split at {split_directory}; run `bindra make-splits build`")
        return 1

    # An absent GPU is an ordinary condition here, not a crash: the training
    # card is external and gets unplugged. select_device already explains which
    # devices it saw and why it rejected each, so this only stops that
    # explanation arriving wrapped in a traceback.
    if args.device:
        device = torch.device(args.device)
    else:
        try:
            device = select_device(DeviceRequirement(min_memory_gib=8.0))
        except DeviceSelectionError as exc:
            print(f"[{FAIL}] {exc}")
            return 1
    print(f"[{OK}] device {describe_device(device)}", flush=True)

    cached_layers = _cache_layers(args.protein_cache)
    n_cached = len(cached_layers) if cached_layers else 1
    # --protein-layers is a cross-check, not the source of truth. Left at its
    # default it simply follows the cache; set to something the cache does not
    # hold it is an error here rather than a StaleCacheError two frames down.
    if args.protein_layers not in (1, n_cached):
        print(
            f"[{FAIL}] --protein-layers {args.protein_layers} but the cache at "
            f"{args.protein_cache} holds {n_cached} layer(s)"
            + (f" {list(cached_layers)}" if cached_layers else "")
            + ". Point at the matching cache, or drop the flag and let the "
            "cache decide.",
            flush=True,
        )
        return 2
    args.protein_layers = n_cached
    protein_metadata = _protein_metadata(processed, args.protein_cache)
    cache = ProteinFeatureCache(args.protein_cache, metadata=protein_metadata)
    print(
        f"[{OK}] protein cache: {len(cache):,} constructs, "
        f"{cache.total_bytes / 2**30:.3f} GiB, dim {protein_metadata.output_dim}"
        + (f" ({n_cached} layers {list(cached_layers)})" if cached_layers else ""),
        flush=True,
    )

    # The ligand config is built before the loaders, not after, because the
    # loaders featurize for an arm and must be handed the same config object the
    # model is built from. Deriving the featurizer from a separate flag is how a
    # run ends up encoding GAT graphs with a KPGT arm.
    ligand = _ligand_config(args)
    ligand_builder = LigandBatchBuilder.for_config(ligand)
    if args.knowledge_cache:
        from pathlib import Path as _Path

        if not (_Path(args.knowledge_cache) / "index.json").exists():
            print(
                f"[{FAIL}] no knowledge cache at {args.knowledge_cache}. Build it "
                "with `bindra cache knowledge`, or pass --knowledge-cache ''",
                flush=True,
            )
            return 2
        ligand_builder = dataclasses.replace(
            ligand_builder, knowledge_cache=args.knowledge_cache
        )
        print(
            f"[{OK}] knowledge cache at {args.knowledge_cache}: fingerprints and "
            "descriptors read rather than recomputed",
            flush=True,
        )
    print(
        f"[{INFO}] building loaders for the LiGhT ligand arm "
        "(one pass over the corpus)...",
        flush=True,
    )
    train = PairDataset(
        processed, split_directory, "train", protein_cache=cache, limit=args.limit,
        ligand_builder=ligand_builder, with_assay=args.assay_conditioning,
        cache_dir=args.dataset_cache or None,
    )
    validation = PairDataset(
        processed,
        split_directory,
        "val",
        protein_cache=cache,
        limit=None if args.limit is None else max(1, args.limit // 8),
        ligand_builder=ligand_builder,
        with_assay=args.assay_conditioning,
        cache_dir=args.dataset_cache or None,
    )
    policy = BatchingPolicy(
        residue_budget=args.residue_budget, max_pairs=args.max_pairs, seed=args.seed
    )
    train_loader = PairLoader(train, policy, prefetch_workers=args.workers,
                              featurize_workers=args.featurize_workers)
    val_loader = PairLoader(
        validation, policy, shuffle=False, prefetch_workers=args.workers,
        featurize_workers=args.featurize_workers,
    )
    print(
        f"[{OK}] train {len(train):,} pairs in {len(train_loader):,} batches | "
        f"val {len(validation):,} pairs in {len(val_loader):,} batches",
        flush=True,
    )
    print(f"[{INFO}] batch shape: {train_loader.describe()}", flush=True)

    # The cache stores `protein_layers` layers concatenated, so one layer is
    # that width divided by the count. Dividing here rather than in the model
    # keeps `protein_dim` meaning the same thing everywhere.
    cached_width = protein_metadata.output_dim
    if cached_width % args.protein_layers:
        print(
            f"[{FAIL}] cache width {cached_width} is not divisible by "
            f"--protein-layers {args.protein_layers}",
            flush=True,
        )
        return 2
    config = BindraConfig(
        protein_dim=cached_width // args.protein_layers,
        ligand=ligand,
        model_dim=args.model_dim,
        cross_attention_layers=args.fusion_layers,
        cross_attention_heads=8,
        dropout=args.dropout,
        feedforward_multiplier=args.feedforward_multiplier,
        attention_pooling=not args.no_attention_pooling,
        pool_heads=args.pool_heads,
        protein_layers=args.protein_layers,
        assay_conditioning=args.assay_conditioning,
        adapter_bottleneck=args.adapter_bottleneck,
        film_bottleneck=args.film_bottleneck or None,
        head=AffinityHeadConfig(
            hidden_dim=args.model_dim, dropout=args.dropout,
            share_trunk=not args.no_share_head_trunk,
        ),
    )
    model = BindraModel(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{OK}] model: {trainable:,} trainable parameters", flush=True)

    # Pretrained ligand weights, before anything reads the arm's identity.
    # `_ligand_metadata` reports `pretrained_checkpoint` into provenance, so
    # loading after that point would record the run as untrained while training
    # a pretrained encoder.
    if args.ligand_checkpoint:
        from bindra.encoders.ligand_arm import LigandArmError

        manifest = json.loads(
            (split_directory / "manifest.json").read_text(encoding="utf-8")
        )
        try:
            identity = model.ligand_encoder.load_ligand_checkpoint(
                Path(args.ligand_checkpoint),
                regime=manifest["regime"],
                split_id=manifest["split_id"],
            )
        except LigandArmError as exc:
            print(f"[{FAIL}] {exc}", flush=True)
            return 2
        print(f"[{OK}] ligand arm initialised from {identity}", flush=True)
    else:
        print(
            f"[{INFO}] no --ligand-checkpoint: the ligand arm starts from random "
            "initialisation. `bindra pretrain` produces one, and running it after "
            "this means training the affinity model twice.",
            flush=True,
        )

    # Neither guard is reachable from the flags any more: the CLI builds only
    # LiGhT-mini, which trains every weight it has. They are kept because a
    # config built in library code still can reach them, and because the second
    # one describes a failure that trains and reports a normal loss curve.
    if getattr(model.ligand_encoder, "frozen_and_untrained", lambda: False)():
        print(
            f"[{FAIL}] the {ligand_builder.arm!r} arm has no trainable layers and "
            "no pretrained weights, so the ligand side is a fixed random "
            "projection and cannot learn. Build it with KPGTConfig.mini(), or "
            "load base.pth (see docs/STATUS.md)",
            flush=True,
        )
        return 2

    pinned = getattr(model.ligand_encoder, "partially_frozen_at_random", lambda: 0)()
    if pinned:
        print(
            f"[{FAIL}] {pinned:,} parameters of the {ligand_builder.arm!r} arm are "
            "frozen at RANDOM initialisation, because partial fine-tuning was "
            "requested without a pretrained checkpoint. The trailing blocks "
            "would spend training compensating for a fixed random embedding, "
            "and the run would look normal throughout. Build it with "
            "KPGTConfig.mini(), or load base.pth (see docs/STATUS.md)",
            flush=True,
        )
        return 2

    ligand_metadata = _ligand_metadata(ligand, model.ligand_encoder)
    provenance = _provenance(processed, split_directory, protein_metadata, ligand_metadata)
    provenance["model"] = config.to_dict()
    provenance["batching"] = policy.to_dict()

    from bindra.training.losses.ranking import RankingLossConfig, margin_per_endpoint

    ranking = RankingLossConfig(
        weight=args.ranking_weight,
        margin_scale=args.ranking_margin_scale,
        margin_override=args.ranking_margin,
        max_pairs=args.ranking_max_pairs,
    )
    if ranking.weight:
        print(
            f"[{OK}] within-assay ranking term at weight {ranking.weight}; "
            f"margins {margin_per_endpoint(ranking)}",
            flush=True,
        )
        print(
            f"[{INFO}] the margin defaults to each endpoint's measured replicate "
            "noise floor. The specification's 0.5 is below the floor of every "
            "endpoint, so it would rank pairs the laboratory could not reproduce; "
            "pass --ranking-margin 0.5 to reproduce it anyway.",
            flush=True,
        )
    else:
        print(
            f"[{INFO}] no ranking term (--ranking-weight 0). The objective is "
            "affinity alone, which is what every recorded number so far used.",
            flush=True,
        )

    training = TrainingConfig(
        ranking=ranking,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        accumulate_steps=args.accumulate,
        ema_decay=args.ema,
        seed=args.seed,
        # --verbose means every step unless a cadence was asked for explicitly.
        log_every=1 if args.verbose and args.log_every == 100 else args.log_every,
        verbose=args.verbose,
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
    parser.add_argument(
        "--dataset-cache", default="data/cache/datasets",
        help="Where the assembled split partition is cached. Building it is a "
             "full pass over 2.7M rows plus two joins, about ten minutes, and "
             "it is identical every run. Keyed on the corpus and split content "
             "hashes, so a rebuild misses rather than serving stale rows. "
             "Pass '' to rebuild every time.",
    )
    parser.add_argument("--out", default="runs")
    parser.add_argument("--regime", default="random_pair")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default=None, help="Defaults to the largest CUDA device.")
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs, for smoke runs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model-dim", type=int, default=160,
        help="160 is the only width with a measured result on this split "
             "(val RMSE 1.1718, and still falling when that run stopped). "
             "96 was v1's and scored 1.2432.",
    )
    parser.add_argument(
        "--film-bottleneck", type=int, default=128,
        help="Hidden width in the FiLM projection. Without it the projection "
             "is the largest non-fusion block in the model (788,992 at "
             "model-dim 256) for what is a scale and a shift, not a "
             "representation. 0 restores the full map.",
    )
    parser.add_argument(
        "--no-share-head-trunk", action="store_true",
        help="Give each endpoint its own whole MLP instead of one trunk with "
             "three outputs. Costs 918,528 at model-dim 256 for three "
             "near-identical networks over three correlated endpoints. The "
             "endpoints stay separate either way: each keeps its own output "
             "layer and nothing is mixed across them.",
    )
    parser.add_argument(
        "--adapter-bottleneck", type=int, default=None,
        help="Hidden width inside each feature adapter. Default None is a "
             "single linear map, which at --model-dim 96 compressed ESM-C's "
             "1152 by 12x in one step. A value here makes it a two-layer MLP.",
    )
    parser.add_argument(
        "--pool-heads", type=int, default=2,
        help="Attention-pooling heads over residues and ligand tokens.",
    )
    parser.add_argument(
        "--no-attention-pooling", action="store_true",
        help="Pool with a plain masked mean instead of learned relevance. The "
             "two are bit-identical at initialisation, so this is a clean "
             "ablation rather than a different model.",
    )
    parser.add_argument(
        "--protein-layers", type=int, default=1,
        help="How many ESM-C layers the protein cache holds. Must match the "
             "cache: `bindra cache proteins --layers` writes it, and the cache "
             "key refuses a mismatch rather than mis-slicing.",
    )
    parser.add_argument(
        "--assay-conditioning", action="store_true",
        help="Condition on each measurement's recorded assay conditions (pH/T "
             "bucket, IC50 tier, source class, metadata confidence). IC50 is "
             "72%% of the corpus and is not a constant of the pair. Identity "
             "columns are excluded; see bindra.data.assay.EXCLUDED_AS_LEAKAGE.",
    )
    parser.add_argument(
        "--knowledge-cache", default="data/cache/knowledge",
        help="Precomputed KPGT fingerprints/descriptors. They are 81%% of ligand "
             "featurization and featurization is what the loop waits on; cached "
             "they read in 2.03 ms/mol against 37.01 live. Pass '' to compute live.",
    )
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument(
        "--feedforward-multiplier", type=int, default=2,
        help="Fusion feed-forward width, as a multiple of --model-dim. At the "
             "default 4 the fusion stack is 52.9%% of trainable parameters "
             "against the ligand arm's 23.9%%; 2 gives 42.8%%/29.1%% and 1 gives "
             "36.0%%/32.5%%. Lowering it keeps an arm-vs-arm comparison valid but "
             "invalidates the recorded 1.0809 baseline.",
    )
    parser.add_argument(
        "--accumulate", type=int, default=1,
        help="Micro-batches per optimizer step. Raises the effective batch at "
             "no memory cost; mean micro-batch is ~87 pairs.",
    )
    parser.add_argument("--residue-budget", type=int, default=65_536)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3,
                        help="Threads overlapping batch assembly with the GPU.")
    parser.add_argument("--featurize-workers", type=int, default=8,
                        help="Processes featurizing ligands. RDKit is GIL-bound, "
                             "so threads cannot use the extra cores; 0 disables.")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Per-step detail: loss terms, gradient norm, batch shape, data wait. "
             "Implies --log-every 1 unless you set it explicitly.",
    )
    parser.add_argument(
        "--ligand-checkpoint", default=None,
        help="Pretrained LiGhT weights from `bindra pretrain`. The regime and "
             "split id recorded in it must match this run: pre-training leaks no "
             "labels but it does leak structure, so an encoder pretrained against "
             "another split has already seen molecules this one evaluates on. A "
             "published base.pth is accepted here too and loaded strictly.",
    )
    parser.add_argument(
        "--ranking-weight", type=float, default=0.0,
        help="Weight on the within-assay pairwise ranking term. 0 disables "
             "it, which is the default and what every recorded run used: "
             "adding a term changes the objective, so it turns on "
             "deliberately rather than by upgrading.",
    )
    parser.add_argument(
        "--ranking-margin", type=float, default=None,
        help="One margin in p-units for all endpoints, replacing the "
             "floor-derived per-endpoint margins. The specification starts "
             "at 0.5, which is BELOW the replicate noise floor of every "
             "endpoint (0.945/0.641/0.671), so it selects pairs whose "
             "difference the laboratory could not reproduce.",
    )
    parser.add_argument(
        "--ranking-margin-scale", type=float, default=1.0,
        help="Multiplies each endpoint's noise floor to get its margin. At "
             "1.0 a pair must differ by more than replicate scatter.",
    )
    parser.add_argument(
        "--ranking-max-pairs", type=int, default=4096,
        help="Pairs scored per batch, subsampled at random. One "
             "comparability group of 2,295 rows would otherwise contribute "
             "2.6M pairs and dominate the term.",
    )
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return cmd_fit(build_parser().parse_args(argv))


__all__ = ["build_parser", "cmd_fit", "main"]
