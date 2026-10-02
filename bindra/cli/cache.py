"""``bindra cache`` -- populate and verify frozen encoder feature caches.

Protein features only for now; the ligand half waits on the KPGT environment.

Three things this command does deliberately:

**Selects the GPU by capability, never by index.** This machine has two CUDA
devices and CUDA's default ordering is not ``nvidia-smi``'s, so a hardcoded
``cuda:1`` lands on the 4 GB T1200 under one ordering and the 16 GB 5060 Ti under
the other. ``bindra.runtime.device`` picks by compute capability and memory, and
refuses to fall back to CPU unless asked.

**Pilots before committing.** The specification requires estimating storage and
runtime from a representative pilot before encoding the whole corpus. The pilot
sample is drawn across the measured length distribution and deliberately includes
the longest construct and several of the 261 that exceed the encoder's
2,046-residue window, so the windowing and merge path is exercised rather than
assumed.

**Verifies cached against live.** ``bindra cache verify`` re-encodes a random
sample and compares bit-for-bit, which is the specification's cached/live feature
agreement requirement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

OK = "  ok   "
FAIL = " FAIL  "
INFO = " info  "


def _load_constructs(processed_dir: Path):
    import pandas as pd

    constructs = pd.read_parquet(processed_dir / "target_constructs.parquet")
    constructs = constructs[constructs["status"] == "OK"]
    sequences = pd.read_parquet(processed_dir / "sequences_raw.parquet").set_index(
        "sequence_ref"
    )
    return constructs.join(sequences["sequence"], on="sequence_ref").reset_index(drop=True)


def _pilot_selection(frame, n: int):
    """A length-stratified sample that always includes the hard cases.

    An encode run that only ever saw median-length proteins would tell us nothing
    about the windowed path, which is where the residue-numbering bugs live.
    """
    import numpy as np

    frame = frame.sort_values("sequence_length").reset_index(drop=True)
    windowed = frame[frame["sequence_length"] > 2046]
    picks: list[int] = []

    # Always the longest construct, then a spread of over-window ones.
    picks.append(int(frame["sequence_length"].idxmax()))
    if len(windowed):
        step = max(1, len(windowed) // 8)
        picks.extend(int(i) for i in windowed.index[::step][:8])

    remaining = max(n - len(set(picks)), 0)
    if remaining:
        quantiles = np.linspace(0, len(frame) - 1, remaining).astype(int)
        picks.extend(int(i) for i in quantiles)

    seen: list[int] = []
    for index in picks:
        if index not in seen:
            seen.append(index)
    return frame.loc[seen[:n]].reset_index(drop=True)


def _encoder(device, *, batch_hint: str = "") -> tuple:
    from bindra.encoders.protein_esmc import EXPECTED_ESMC_600M_DIM, ESMCProteinEncoder

    # Frozen is pinned here rather than inherited from the default. Caching
    # requires it -- a cached feature is only valid because it is a
    # deterministic function of the sequence -- so this path states the
    # requirement instead of depending on a default it does not control.
    encoder = ESMCProteinEncoder(device=device, n_trainable_layers=0)
    print(f"[{INFO}] loading {encoder.checkpoint} @ {encoder.revision}{batch_hint}", flush=True)
    checksum = encoder.weights_checksum()
    metadata = encoder.metadata(EXPECTED_ESMC_600M_DIM, weights_checksum=checksum)
    print(f"[{OK}] weights checksum {checksum[:16]}...", flush=True)
    return encoder, metadata


def _encode_frame(
    frame,
    encoder,
    metadata,
    cache,
    *,
    label: str,
    progress_every: int = 250,
) -> dict:
    import torch

    # A cached feature is valid only because it is a deterministic function of
    # the sequence. One trainable block ends that, and the cache would be stale
    # from the first optimizer step with nothing to signal it.
    if hasattr(encoder, "assert_cacheable"):
        encoder.assert_cacheable()

    device = encoder.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Length-sorted so VRAM use rises smoothly instead of thrashing between a
    # 7-residue peptide and a 34,350-residue construct.
    frame = frame.sort_values("sequence_length").reset_index(drop=True)

    started = time.perf_counter()
    encoded = skipped = 0
    residues = 0
    windowed = 0
    per_sequence: list[float] = []

    for position, row in enumerate(frame.itertuples(index=False), start=1):
        sequence = row.sequence
        key = metadata.cache_key(sequence)
        if key in cache:
            skipped += 1
            continue
        tick = time.perf_counter()
        features = encoder.encode(sequence)
        per_sequence.append(time.perf_counter() - tick)
        if features.residue_features.shape[0] != len(sequence):
            raise RuntimeError(
                f"{row.construct_id}: encoder returned "
                f"{features.residue_features.shape[0]} rows for {len(sequence)} residues"
            )
        cache.put(key, features)
        encoded += 1
        residues += len(sequence)
        if len(sequence) > encoder.window_size:
            windowed += 1
        if progress_every and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = residues / max(elapsed, 1e-9)
            print(
                f"    {label}: {position:,}/{len(frame):,} "
                f"({encoded:,} encoded, {skipped:,} cached) "
                f"{rate:,.0f} residues/s",
                flush=True,
            )
            cache.flush()

    cache.flush()
    elapsed = time.perf_counter() - started
    peak = 0
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))

    per_sequence.sort()
    return {
        "label": label,
        "n_considered": int(len(frame)),
        "n_encoded": encoded,
        "n_already_cached": skipped,
        "n_windowed": windowed,
        "residues_encoded": residues,
        "elapsed_seconds": round(elapsed, 2),
        "residues_per_second": round(residues / elapsed, 1) if elapsed else None,
        "sequences_per_second": round(encoded / elapsed, 3) if elapsed else None,
        "seconds_per_sequence_median": (
            round(per_sequence[len(per_sequence) // 2], 4) if per_sequence else None
        ),
        "seconds_per_sequence_max": round(per_sequence[-1], 4) if per_sequence else None,
        "peak_vram_bytes": peak,
        "peak_vram_gib": round(peak / 1024**3, 3) if peak else None,
        "bytes_per_residue": round(residues and cache.total_bytes / cache.total_residues, 3),
        "device": str(device),
    }


def cmd_proteins(args) -> int:
    from bindra.features.cache import ProteinFeatureCache, default_cache_root
    from bindra.runtime.device import DeviceRequirement, DeviceSelectionError, select_device

    processed = Path(args.processed)
    try:
        device = select_device(DeviceRequirement(allow_cpu=args.allow_cpu))
    except DeviceSelectionError as exc:
        print(f"[{FAIL}] device selection\n{exc}")
        return 1
    print(f"[{OK}] device {device}")

    frame = _load_constructs(processed)
    print(
        f"[{INFO}] {len(frame):,} valid constructs, "
        f"{int(frame['sequence_length'].sum()):,} residues, "
        f"{int((frame['sequence_length'] > 2046).sum()):,} over the encoder window"
    )

    out = Path(args.out) if args.out else default_cache_root() / "proteins"
    encoder, metadata = _encoder(device)

    selection = _pilot_selection(frame, args.pilot) if args.pilot else frame
    label = f"pilot({args.pilot})" if args.pilot else "full"
    if args.pilot:
        print(
            f"[{INFO}] pilot of {len(selection)} constructs, lengths "
            f"{int(selection['sequence_length'].min())}-"
            f"{int(selection['sequence_length'].max())}, "
            f"{int((selection['sequence_length'] > 2046).sum())} windowed"
        )

    with ProteinFeatureCache(out, metadata=metadata, mode="w") as cache:
        summary = _encode_frame(selection, encoder, metadata, cache, label=label)
        stats = cache.stats()

    summary["cache"] = stats
    if args.pilot:
        total_residues = int(frame["sequence_length"].sum())
        rate = summary["residues_per_second"] or 1.0
        summary["projection_full_corpus"] = {
            "total_residues": total_residues,
            "estimated_hours": round(total_residues / rate / 3600, 2),
            "estimated_gib": round(total_residues * metadata.output_dim * 2 / 1024**3, 2),
        }

    report = out / ("pilot.json" if args.pilot else "encode_report.json")
    report.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"[{OK}] encoded {summary['n_encoded']:,} ({summary['n_already_cached']:,} already cached)")
    print(f"[{OK}] {summary['residues_per_second']:,} residues/s, peak VRAM {summary['peak_vram_gib']} GiB")
    print(f"[{OK}] cache holds {stats['n_entries']:,} entries, {stats['total_gib']} GiB across {stats['n_shards']} shards")
    if args.pilot:
        projection = summary["projection_full_corpus"]
        print(
            f"[{INFO}] projection for all {projection['total_residues']:,} residues: "
            f"{projection['estimated_hours']} h, {projection['estimated_gib']} GiB"
        )
    print(f"[{OK}] wrote {report}")
    return 0


def cmd_verify(args) -> int:
    """Re-encode a random sample and compare with the cache, bit for bit."""
    import random

    import torch

    from bindra.features.cache import ProteinFeatureCache, default_cache_root
    from bindra.runtime.device import DeviceRequirement, DeviceSelectionError, select_device

    processed = Path(args.processed)
    out = Path(args.out) if args.out else default_cache_root() / "proteins"
    try:
        device = select_device(DeviceRequirement(allow_cpu=args.allow_cpu))
    except DeviceSelectionError as exc:
        print(f"[{FAIL}] {exc}")
        return 1

    frame = _load_constructs(processed)
    encoder, metadata = _encoder(device)
    cache = ProteinFeatureCache(out, metadata=metadata, mode="r")
    print(f"[{OK}] cache holds {len(cache):,} entries")

    population = [
        row for row in frame.itertuples(index=False) if metadata.cache_key(row.sequence) in cache
    ]
    if not population:
        print(f"[{FAIL}] no cached constructs to verify")
        return 1
    count = max(1, int(len(population) * args.sample))
    random.Random(args.seed).shuffle(population)
    chosen = population[:count]

    print(f"[{INFO}] re-encoding {len(chosen):,} of {len(population):,} cached constructs")
    exact = 0
    worst = 0.0
    for row in chosen:
        key = metadata.cache_key(row.sequence)
        cached = cache.get(key).residue_features
        live = encoder.encode(row.sequence).residue_features
        if cached.shape != live.shape:
            print(f"[{FAIL}] {row.construct_id}: shape {tuple(cached.shape)} vs {tuple(live.shape)}")
            return 1
        if torch.equal(cached, live):
            exact += 1
        else:
            difference = (cached.float() - live.float()).abs().max().item()
            worst = max(worst, difference)

    print()
    if exact == len(chosen):
        print(f"[{OK}] all {exact:,} re-encoded constructs are bit-identical to the cache")
        return 0
    print(
        f"[{FAIL}] {len(chosen) - exact:,} of {len(chosen):,} differ; "
        f"max absolute difference {worst:.3e}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bindra cache",
        description="Populate and verify frozen encoder feature caches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    proteins = subparsers.add_parser("proteins", help="Encode protein constructs with ESM-C.")
    proteins.add_argument("--processed", default="data/processed/202610")
    proteins.add_argument("--out", default=None, help="Cache directory (default: $BINDRA_CACHE_ROOT/proteins).")
    proteins.add_argument(
        "--pilot",
        type=int,
        default=0,
        metavar="N",
        help="Encode a length-stratified pilot of N constructs and project the full run.",
    )
    proteins.add_argument("--allow-cpu", action="store_true", help="Permit CPU (very slow).")
    proteins.set_defaults(func=cmd_proteins)

    verify = subparsers.add_parser("verify", help="Compare cached features against live encoding.")
    verify.add_argument("--processed", default="data/processed/202610")
    verify.add_argument("--out", default=None)
    verify.add_argument("--sample", type=float, default=0.01, help="Fraction to re-encode.")
    verify.add_argument("--seed", type=int, default=20261001)
    verify.add_argument("--allow-cpu", action="store_true")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
