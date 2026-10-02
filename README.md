# BINDRA

**B**inding **IN**ference, **D**esign, **R**etrieval, and **A**ttention.

A sequence-based platform for protein–ligand affinity prediction, attention-based
explanation (iBAM), target-conditioned retrieval, and diffusion-based analog
generation. Successor to [DeepDTA-iBAM](https://github.com/kevinmsong/DeepDTA-iBAM),
which remains frozen as the archival record of its published work.

> **Current state: curated data and encoder caches, no model.** BindingDB 202610 is
> ingested, standardized, deduplicated, assay-grouped and reported
> (**2,739,920 curated observations**), and all 10,820 protein constructs are
> encoded into a verified 14 GiB ESM-C feature cache. Cross-attention, iBAM,
> retrieval and diffusion are not built, and **no BINDRA model has been trained —
> there are no BINDRA performance numbers.** See
> [`docs/STATUS.md`](docs/STATUS.md) for the component-by-component ledger — it
> is the authoritative answer to "does BINDRA do X yet?" — and
> [`docs/CURATION_REPORT.md`](docs/CURATION_REPORT.md) for what the data actually
> supports.

## Specification

[`BINDRA_Agent_Instructions.md`](BINDRA_Agent_Instructions.md) is the source of
truth for requirements. Every scaffolded module names the clause it owes.

## Architecture

```
sequence ──► ESM-C-600M (frozen) ──┐
                                   ├─► shared adapters ─► bidirectional ─► pKd / pKi / pIC50 heads
SMILES ────► KPGT (frozen) ────────┘                      cross-attention   │
                                                                 │          ├─► iBAM heatmaps
                                                                 │          ├─► retrieval projections
                                                                 └──────────┴─► diffusion conditioning
```

Encoders stay frozen and cached; the adapters and task modules are trained
jointly. Full encoder fine-tuning is a later, separately measured experiment.

## Requirements

- **Python 3.12** (not 3.13 — several chemistry and graph dependencies lack 3.13 wheels)
- **CUDA 12.8+ PyTorch.** The target GPU is an RTX 5060 Ti (Blackwell, sm_120),
  which the default PyPI CPU wheel and any pre-2.7 CUDA wheel cannot drive.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install -e ".[encoders,dev]"
```

Then confirm the environment before anything else:

```bash
python -m bindra.cli.doctor --esmc --verify-weights
```

This must report the RTX 5060 Ti at `sm_120`, not the 4 GB T1200. It exits
non-zero if the environment cannot train, so it is safe to gate scripts on.

```
[  ok   ] selected cuda:1  NVIDIA GeForce RTX 5060 Ti  15.9 GiB  sm_120
[  ok   ] VRAM 14.8 GiB free of 15.9 GiB total
[  ok   ] bf16 matmul 4096x4096: 2.83 ms/iter (48.6 TFLOP/s)
[  ok   ] ESM-C-600M encode: (33, 1152) (torch.bfloat16) for a 33-residue sequence
[  ok   ] kernel backends: none (pure PyTorch)  (part of the cache key)
[  ok   ] pinned revision fcc9d36f5fff97d9bb69a36e3252de61cc78c968
[  ok   ] weights checksum matches the pinned digest
```

The ESM-C check deliberately goes through `ESMCProteinEncoder`, the same adapter
that populates the feature cache. An earlier version called `esm` directly and so
validated a code path nothing else used.

### Why GPU selection is explicit

This machine has two NVIDIA GPUs and CUDA's device order is not `nvidia-smi`'s.
`bindra.runtime.device` selects by compute capability and memory, never by bare
index, and **refuses to fall back to CPU** unless asked. The legacy project ran
`torch 2.9.1+cpu` without anything saying so.

### The KPGT environment is separate

KPGT requires DGL, which stops at Python 3.12 / torch ~2.4 and no longer ships
Windows CUDA wheels. Since KPGT is frozen and only ever needs a forward pass, it
runs in an isolated CPU environment under `envs/kpgt/` and exchanges versioned
cached arrays with the training environment. The training environment never
imports DGL.

## The curation pipeline

Run in this order. Each stage reads the previous stage's output and writes a JSON
side-car beside it; the curation report is assembled from those side-cars, so it
cannot drift from the artifacts it describes.

```bash
bindra prepare-data verify      data/snapshots/bindingdb_202610
bindra prepare-data ingest      --snapshot data/snapshots/bindingdb_202610 --out data/processed/202610
# author bindra/data/bindingdb/source_classes.json from ingest_stats.json, then:
bindra prepare-data standardize --processed data/processed/202610 --workers 14
bindra prepare-data homology    --processed data/processed/202610 --workers 14   # long, checkpointed
bindra prepare-data curate      --processed data/processed/202610
bindra prepare-data report      --processed data/processed/202610

bindra make-splits build   --processed data/processed/202610 --out data/splits/202610
bindra make-splits verify  --processed data/processed/202610 --out data/splits/202610

bindra cache proteins --pilot 200     # measure before committing to the full encode
bindra cache proteins                 # ~14 GiB, resumable
bindra cache verify --sample 0.01     # cached vs live, bit-for-bit
```

`homology` aligns every one of the 58,530,790 construct pairs exactly and
checkpoints each block, so an interrupted run resumes instead of restarting. It is
the only stage measured in hours.

`make-splits` verifies the curated table against the hash recorded in
`qc_report.json` before it starts. A directory listing is not a dataset: `curate`
rewrites `observations/` shard by shard, and a glob run mid-rewrite returns a
partial corpus that looks perfectly healthy.

## Commands

| Command | Status | Purpose |
|---|---|---|
| `bindra doctor` | ready | Verify GPU, torch build, pinned encoder and weights |
| `bindra prepare-data` | ready | Acquire, parse and curate a BindingDB snapshot |
| `bindra make-splits` | ready | Build and audit immutable split manifests |
| `bindra cache` | ready (proteins) | Populate and verify frozen encoder caches |
| `bindra train` | not built | Train the affinity / retrieval / diffusion stack |
| `bindra calibrate` | not built | Fit conformal calibration |
| `bindra predict` | not built | Predict affinity for sequence/SMILES pairs |
| `bindra retrieve` | not built | Target fishing and analog fishing |
| `bindra generate` | not built | Generate scaffold-constrained analogs |
| `bindra plot-ibam` | not built | Export iBAM matrices and heatmaps |
| `bindra evaluate` | not built | Run the locked evaluation panels |

Unbuilt commands exit with a message naming the blocker rather than appearing to
succeed. [`docs/STATUS.md`](docs/STATUS.md) is the authoritative ledger.

## Where the bulk artifacts go

`data/processed/` holds the curated tables (~0.5 GB) and `data/cache/` the frozen
encoder features (**14.0 GiB** for the 10,820 protein constructs). Both are
git-ignored.

This repository lives inside a OneDrive sync root, so the feature cache is 14 GiB
of sync traffic. `BINDRA_CACHE_ROOT` moves it without touching any call site:

```bash
export BINDRA_CACHE_ROOT=C:\bindra\cache
```

## Tests

```bash
pytest                      # everything available on this machine
pytest -m "not gpu"         # skip tests needing a CUDA device
pytest -m "not encoders"    # skip tests needing downloaded weights
```

## Conventions this project holds to

- **pKd, pKi and pIC50 never merge.** IC50 is assay-dependent and is not Kd.
  KIBA is not a pKd label.
- **Censoring is first-class.** A `>` or `<` measurement is a bound, trained
  against with a one-sided penalty, never as an exact value.
- **Computational decoys are presumed controls**, not measured nonbinders, and
  never receive fabricated affinity labels.
- **iBAM is an attention-based explanation**, not a contact map, binding pose or
  established mechanism.
- **Status is tracked separately** from the existence of code. See
  [`docs/STATUS.md`](docs/STATUS.md).
