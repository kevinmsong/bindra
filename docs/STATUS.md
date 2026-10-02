# BINDRA status ledger

The specification is explicit: *"Do not mark all requested functionality complete
merely because interfaces exist. Separate implemented, smoke-tested, trained,
benchmarked, and experimentally validated status."*

This file is that separation. It is the authoritative answer to "does BINDRA do X
yet?" — not the presence of a module, not a passing import.

**Last updated:** 2026-10-01

## Status vocabulary

| Status | Meaning |
|---|---|
| `absent` | Not written. |
| `scaffolded` | Interface and contract exist; the body raises `NotImplementedError`. |
| `implemented` | Logic written, unit-tested in isolation. Says nothing about real data. |
| `smoke-tested` | Runs end to end on real inputs at small scale. |
| `trained` | A model was actually fit on the curated corpus. |
| `benchmarked` | Measured on locked, prespecified evaluation panels. |
| `validated` | Confirmed by **experimental measurement**, not by another model. |

Nothing below is `validated`. Nothing below is `trained`.

## Priority 0 — environment and legacy preservation

| Component | Status | Evidence |
|---|---|---|
| CUDA environment (torch 2.11.0+cu128) | smoke-tested | `bindra doctor`: selects RTX 5060 Ti, sm_120, 49.4 TFLOP/s bf16 |
| GPU selection by capability, not index | implemented | `bindra/runtime/device.py`; rejects the 4 GB T1200 |
| ESM-C-600M forward pass on GPU | smoke-tested | `bindra doctor --esmc`: `(1, 35, 1152)` bf16 |
| KPGT isolated environment | absent | Blocked, see below |
| Legacy baseline reproduction | absent | Single-member only, see blockers |

## Priority 1 — endpoint-aware BindingDB dataset

| Component | Status | Evidence |
|---|---|---|
| Endpoint / p-value conversion | implemented | `bindra/data/endpoints.py`, 41 tests incl. the spec's required table |
| Censoring relations and reversal | implemented | Round-trip and strictness tests pass |
| BindingDB ingestion | absent | **Blocked: no snapshot downloaded** |
| Standardization and QC | absent | |
| Deduplication / assay comparability | absent | |
| Immutable splits (cold/double-cold/temporal) | absent | |
| Curation report | absent | |

## Priority 2 — cross-attention, iBAM, retrieval

| Component | Status | Evidence |
|---|---|---|
| `ProteinFeatures` / `LigandFeatures` contracts | implemented | `bindra/features/types.py` |
| ESM-C adapter (token stripping, windowing) | implemented | `bindra/encoders/protein_esmc.py`, 28 tests |
| Cache-key discipline (encoder identity in key) | implemented | Tested: any encoder change invalidates the key |
| KPGT adapter | absent | Blocked on the isolated environment |
| Sharded feature cache | absent | Legacy design does not scale, see plan |
| Cross-attention fusion | absent | Port target identified |
| Per-endpoint heads (pKd/pKi/pIC50) | absent | |
| iBAM export and plots | absent | |
| Retrieval projections / ANN index | absent | |

## Priorities 3–5 and Upgrades A–C

| Component | Status |
|---|---|
| Discrete graph diffusion (DiGress-style) | absent |
| Graph-to-molecule decoder | absent |
| Measured analog-delta learning (Upgrade A) | absent |
| Locked stress-test suite (Upgrade B) | absent |
| Prospective study support (Upgrade C) | absent |
| Ensemble + conformal calibration | absent |
| Applicability-domain reporting | absent |

## Active blockers

1. **No BindingDB snapshot.** Everything in Priority 1 beyond endpoint
   conversion is blocked until a snapshot is obtained and its release, date,
   source URL and checksum recorded. This is the first task of the next stage.

2. **KPGT / DGL dependency conflict.** KPGT needs DGL, which stops at Python
   3.12 / torch ~2.4 and no longer ships Windows CUDA wheels; the training
   stack is torch 2.11+cu128. Resolution: a separate CPU-only Python 3.12
   environment runs frozen KPGT and exchanges versioned cached arrays. Not yet
   built. KPGT pretrained weights not yet downloaded.

3. **Legacy ensemble is incomplete.** `checkpoints/` in the legacy repo holds
   only `max_rmse_cluster_diffusion_member_0`, but that profile declares
   `ensemble_size=5`, and the legacy loader requires all five. The reproducible
   legacy baseline is therefore **single-member inference only**. Recorded
   legacy validation metrics, in KIBA units, not reproduced here yet:
   RMSE 0.473, CI 0.868, R² 0.689, epoch 47.

4. **ESM-C API deprecation.** The installed `esm` package warns that `ESMC` is
   deprecated and now wraps `EsmcForMaskedLM`. The specification requires
   distinguishing a newer checkpoint from an older one carrying a similar name.
   Before any feature caching at scale, pin the revision and record a weights
   checksum (`ESMCProteinEncoder.weights_checksum()` exists for this), and
   decide whether to migrate to the native API.

## Deliberately not claimed

- No BINDRA model has been trained. There are no BINDRA performance numbers.
- No retrieval, generation, or iBAM results exist.
- DIP2A is an exploratory application. No DIP2A binding assay is assumed to
  exist, and no compound is treated as a validated DIP2A binder.
