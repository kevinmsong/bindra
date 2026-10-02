# BINDRA

**B**inding **IN**ference, **D**esign, **R**etrieval, and **A**ttention.

A sequence-based platform for protein–ligand affinity prediction, attention-based
explanation (iBAM), target-conditioned retrieval, and diffusion-based analog
generation. Successor to [DeepDTA-iBAM](https://github.com/kevinmsong/DeepDTA-iBAM),
which remains frozen as the archival record of its published work.

> **Current state: early scaffold.** The GPU environment, endpoint conversion and
> the ESM-C protein encoder are implemented and tested. Most of the platform is
> not built yet, and **no BINDRA model has been trained**. See
> [`docs/STATUS.md`](docs/STATUS.md) for the component-by-component ledger — it
> is the authoritative answer to "does BINDRA do X yet?"

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
python -m bindra.cli.doctor --esmc
```

This must report the RTX 5060 Ti at `sm_120`, not the 4 GB T1200. It exits
non-zero if the environment cannot train, so it is safe to gate scripts on.

```
[  ok   ] selected cuda:1  NVIDIA GeForce RTX 5060 Ti  15.9 GiB  sm_120
[  ok   ] VRAM 14.8 GiB free of 15.9 GiB total
[  ok   ] bf16 matmul 4096x4096: 2.78 ms/iter (49.4 TFLOP/s)
[  ok   ] ESM-C-600M forward pass: (1, 35, 1152) (torch.bfloat16)
```

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

## Commands

| Command | Purpose |
|---|---|
| `bindra doctor` | Verify GPU, torch build, and encoder availability |
| `bindra prepare-data` | Ingest and curate a BindingDB snapshot |
| `bindra make-splits` | Build immutable split manifests |
| `bindra cache` | Populate frozen encoder feature caches |
| `bindra train` | Train the affinity / retrieval / diffusion stack |
| `bindra calibrate` | Fit conformal calibration |
| `bindra predict` | Predict affinity for sequence/SMILES pairs |
| `bindra retrieve` | Target fishing and analog fishing |
| `bindra generate` | Generate scaffold-constrained analogs |
| `bindra plot-ibam` | Export iBAM matrices and heatmaps |
| `bindra evaluate` | Run the locked evaluation panels |

Unimplemented commands exit with a message naming the blocker rather than
appearing to succeed.

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
